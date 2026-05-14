"""
hierarchical_globe_map.py

Generates three layers of tiles from a simple 3-color equirectangular map.
Saves metadata including point count for each layer.
"""

import math
import json
import struct
import base64
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from enum import Enum

from PIL import Image
from scipy.spatial import cKDTree


# ============================================================
# PROGRESS BAR
# ============================================================

class ProgressBar:
    def __init__(self, total: int, desc: str = "Progress", width: int = 50):
        self.total = total
        self.desc = desc
        self.width = width
        self.current = 0
        self.start_time = time.time()
        self.last_print = 0
        
    def update(self, n: int = 1):
        self.current += n
        if self.current - self.last_print >= max(1, self.total // 100) or self.current >= self.total:
            self.print_bar()
            self.last_print = self.current
    
    def print_bar(self):
        percent = self.current / self.total if self.total > 0 else 1
        filled = int(self.width * percent)
        bar = "█" * filled + "░" * (self.width - filled)
        
        elapsed = time.time() - self.start_time
        if percent > 0 and percent < 1:
            eta = elapsed / percent - elapsed
            eta_str = f"ETA: {eta:.1f}s"
        else:
            eta_str = "Done!    "
        
        print(f"\r{self.desc}: |{bar}| {percent*100:5.1f}% ({self.current:,}/{self.total:,}) {eta_str}", end="", flush=True)
        
        if self.current >= self.total:
            print()
    
    def close(self):
        if self.current < self.total:
            self.current = self.total
            self.print_bar()


# ============================================================
# CONFIGURATION
# ============================================================

LAND_POINTS = 2048 * 1024

# Layer configurations
# count = number of Fibonacci sphere points to generate for this layer
# The actual number of tiles saved will be less (only those matching the type)
LAYER_CONFIGS = {
    "land": {
        "count": int(LAND_POINTS),
        "static_file": "land_static.json",
        "dynamic_file": "land_dynamic.json",
        "metadata_file": "land_metadata.json",
        "color": (239, 239, 239)
    },
    "coastal": {
        "count": int(LAND_POINTS/4),
        "static_file": "coastal_static.json",
        "dynamic_file": "coastal_dynamic.json",
        "metadata_file": "coastal_metadata.json",
        "color": (172, 202, 202)
    },
    "deep_sea": {
        "count": int(LAND_POINTS/16),
        "static_file": "deep_sea_static.json",
        "dynamic_file": "deep_sea_dynamic.json",
        "metadata_file": "deep_sea_metadata.json",
        "color": (105, 165, 165)
    }
}

WORLD_MAP_IMAGE = "terrain_map.png"
NEIGHBOR_COUNT = 6


# ============================================================
# TILE DATA STRUCTURES
# ============================================================

class TileType(Enum):
    LAND = "land"
    COASTAL = "coastal"
    DEEP_SEA = "deep_sea"

@dataclass
class Tile:
    id: int
    layer: TileType
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    neighbors: List[int] = field(default_factory=list)
    temperature: str = ""
    precipitation: str = ""
    terrain: str = ""
    vegetation: str = ""
    population: int = 0
    buildings: List[str] = field(default_factory=list)


# ============================================================
# ENCODING HELPERS
# ============================================================

def encode_neighbors(neighbors):
    if not neighbors:
        return ""
    binary = struct.pack(f"<{len(neighbors)}I", *neighbors)
    return base64.b64encode(binary).decode("ascii")

def decode_neighbors(encoded):
    if not encoded:
        return []
    binary = base64.b64decode(encoded)
    count = len(binary) // 4
    return list(struct.unpack(f"<{count}I", binary))


# ============================================================
# FIBONACCI SPHERE GENERATION
# ============================================================

def fibonacci_sphere(samples: int, progress_desc: str = None) -> List[Tuple[float, float, float]]:
    points = []
    phi = (1 + math.sqrt(5)) / 2
    golden_angle = 2 * math.pi / phi
    
    bar = None
    if progress_desc:
        bar = ProgressBar(samples, progress_desc)
    
    for i in range(samples):
        y = 1 - (2 * i + 1) / samples
        radius = math.sqrt(1 - y * y)
        theta = golden_angle * i
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        points.append((x, y, z))
        
        if bar:
            bar.update(1)
    
    if bar:
        bar.close()
    
    return points

def xyz_to_latlon(x: float, y: float, z: float) -> Tuple[float, float]:
    lat = math.degrees(math.asin(y))
    lon = math.degrees(math.atan2(z, x))
    return lat, lon

def latlon_to_xy(lat: float, lon: float, img_width: int, img_height: int) -> Tuple[int, int]:
    x = int((lon + 180.0) / 360.0 * img_width)
    y = int((90.0 - lat) / 180.0 * img_height)
    return max(0, min(img_width - 1, x)), max(0, min(img_height - 1, y))


# ============================================================
# FAST 3-COLOR CLASSIFICATION
# ============================================================

class ThreeColorClassifier:
    def __init__(self, image_path: str, color_map: Dict[TileType, Tuple[int, int, int]]):
        print(f"Loading world map: {image_path}")
        self.img = Image.open(image_path).convert("RGB")
        print(f"  Size: {self.img.width}x{self.img.height}")
        
        self.color_map = color_map
        self.rgb_to_type = {rgb: tile_type for tile_type, rgb in color_map.items()}
        
        print(f"  Color mapping:")
        for tile_type, rgb in color_map.items():
            print(f"    {tile_type.value}: RGB{rgb}")
    
    def classify_point(self, x: float, y: float, z: float) -> Optional[TileType]:
        lat, lon = xyz_to_latlon(x, y, z)
        px, py = latlon_to_xy(lat, lon, self.img.width, self.img.height)
        rgb = self.img.getpixel((px, py))
        
        for target_rgb, tile_type in self.rgb_to_type.items():
            if self._colors_match(rgb, target_rgb):
                return tile_type
        
        return self._find_closest_type(rgb)
    
    def _colors_match(self, rgb1: Tuple[int, int, int], rgb2: Tuple[int, int, int], tolerance: int = 10) -> bool:
        return all(abs(rgb1[i] - rgb2[i]) <= tolerance for i in range(3))
    
    def _find_closest_type(self, rgb: Tuple[int, int, int]) -> TileType:
        min_dist = float('inf')
        closest_type = TileType.DEEP_SEA
        
        for tile_type, target_rgb in self.color_map.items():
            dist = sum((rgb[i] - target_rgb[i]) ** 2 for i in range(3))
            if dist < min_dist:
                min_dist = dist
                closest_type = tile_type
        
        return closest_type


# ============================================================
# HIERARCHICAL TILE GENERATOR
# ============================================================

class HierarchicalTileGenerator:
    def __init__(self, classifier: ThreeColorClassifier):
        self.classifier = classifier
        self.tiles_by_layer: Dict[TileType, List[Tile]] = {
            TileType.LAND: [],
            TileType.COASTAL: [],
            TileType.DEEP_SEA: []
        }
        self.total_points_processed = 0
        self.type_counts = {TileType.LAND: 0, TileType.COASTAL: 0, TileType.DEEP_SEA: 0}
    
    def generate_layer(self, target_type: TileType, num_points: int) -> List[Tile]:
        print(f"\n{'='*60}")
        print(f"GENERATING {target_type.value.upper()} LAYER")
        print(f"Points to generate: {num_points:,}")
        print(f"ID space: 0 to {num_points - 1:,} (with gaps)")
        print(f"{'='*60}")
        
        # Generate Fibonacci points with progress bar
        all_points = fibonacci_sphere(num_points, f"  Generating points")
        
        # Filter points
        print(f"  Filtering points for {target_type.value}...")
        filtered_tiles = []
        bar = ProgressBar(num_points, "  Filtering")
        
        for original_index, (x, y, z) in enumerate(all_points):
            tile_type = self.classifier.classify_point(x, y, z)
            self.type_counts[tile_type] += 1
            self.total_points_processed += 1
            
            if tile_type == target_type:
                tile = Tile(
                    id=original_index,
                    layer=tile_type,
                    x=x, y=y, z=z
                )
                filtered_tiles.append(tile)
            
            bar.update(1)
        
        bar.close()
        
        print(f"\nGeneration complete for {target_type.value}:")
        print(f"  Points generated: {num_points:,}")
        print(f"  Tiles saved: {len(filtered_tiles):,}")
        print(f"  Retention rate: {len(filtered_tiles)/num_points*100:.1f}%")
        
        if len(filtered_tiles) > 0:
            ids = [t.id for t in filtered_tiles]
            print(f"  ID range: {min(ids)} to {max(ids)}")
            sample_ids = ids[:10] if len(ids) > 10 else ids
            print(f"  Sample IDs: {sample_ids}")
        
        self.tiles_by_layer[target_type] = filtered_tiles
        return filtered_tiles
    
    def build_neighbors_for_layer(self, tiles: List[Tile], neighbor_count: int):
        if len(tiles) < 2:
            print(f"\nSkipping neighbor graph for {len(tiles)} tiles (too few)")
            return
        
        print(f"\nBuilding neighbor graph for {len(tiles):,} tiles...")
        
        # Build KD-tree
        print("  Building KD-tree...")
        positions = [(t.x, t.y, t.z) for t in tiles]
        tree = cKDTree(positions)
        
        # Query neighbors
        print("  Querying nearest neighbors...")
        adjacency = {t.id: set() for t in tiles}
        k = min(neighbor_count + 1, len(tiles))
        
        bar = ProgressBar(len(tiles), "  Processing tiles")
        
        for i, tile in enumerate(tiles):
            distances, indices = tree.query((tile.x, tile.y, tile.z), k=k)
            nearest = indices[1:] if len(indices) > 1 else []
            
            for neighbor_idx in nearest:
                neighbor_id = tiles[neighbor_idx].id
                adjacency[tile.id].add(neighbor_id)
                adjacency[neighbor_id].add(tile.id)
            
            bar.update(1)
        
        bar.close()
        
        # Store neighbors
        print("  Storing neighbor lists...")
        for tile in tiles:
            tile.neighbors = sorted(list(adjacency[tile.id]))
        
        # Statistics
        degrees = [len(t.neighbors) for t in tiles]
        avg_degree = sum(degrees) / len(degrees) if degrees else 0
        print(f"  Neighbor statistics:")
        print(f"    Average: {avg_degree:.1f}")
        print(f"    Min: {min(degrees) if degrees else 0}")
        print(f"    Max: {max(degrees) if degrees else 0}")
    
    def export_layer(self, tile_type: TileType, config: dict):
        """Export a single layer to JSON files including metadata."""
        tiles = self.tiles_by_layer[tile_type]
        
        if not tiles:
            print(f"\nNo {tile_type.value} tiles to export")
            return
        
        static_file = config["static_file"]
        dynamic_file = config["dynamic_file"]
        metadata_file = config["metadata_file"]
        point_count = config["count"]
        
        print(f"\nExporting {tile_type.value} layer:")
        print(f"  Static: {static_file}")
        print(f"  Dynamic: {dynamic_file}")
        print(f"  Metadata: {metadata_file}")
        
        static_data = []
        dynamic_data = []
        
        bar = ProgressBar(len(tiles), "  Exporting tiles")
        
        for tile in tiles:
            static_tile = {
                "id": tile.id,
                "layer": tile.layer.value,
                "neighbors_b64": encode_neighbors(tile.neighbors),
                "temperature": tile.temperature,
                "precipitation": tile.precipitation,
                "terrain": tile.terrain
            }
            static_data.append(static_tile)
            
            dynamic_tile = {
                "id": tile.id,
                "vegetation": tile.vegetation,
                "population": tile.population,
                "buildings": tile.buildings
            }
            dynamic_data.append(dynamic_tile)
            
            bar.update(1)
        
        bar.close()
        
        # Save static data
        with open(static_file, "w") as f:
            json.dump(static_data, f, indent=2)
        
        # Save dynamic data
        with open(dynamic_file, "w") as f:
            json.dump(dynamic_data, f, indent=2)
        
        # Save metadata (critical for rendering!)
        metadata = {
            "layer": tile_type.value,
            "point_count": point_count,
            "tile_count": len(tiles),
            "neighbor_count": NEIGHBOR_COUNT,
            "id_range": {
                "min": min(t.id for t in tiles),
                "max": max(t.id for t in tiles)
            },
            "retention_rate": len(tiles) / point_count
        }
        
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Exported {len(tiles):,} tiles")
        print(f"  Metadata saved (point_count: {point_count:,})")
    
    def generate_all_layers(self):
        print("\n" + "="*60)
        print("HIERARCHICAL TILE GENERATION (3-COLOR MAP)")
        print("="*60)
        print("\nNOTE: Each layer has its own independent ID space starting from 0")
        
        for tile_type in [TileType.LAND, TileType.COASTAL, TileType.DEEP_SEA]:
            config = LAYER_CONFIGS[tile_type.value]
            num_points = int(config["count"])
            tiles = self.generate_layer(tile_type, num_points)
            self.build_neighbors_for_layer(tiles, NEIGHBOR_COUNT)
            self.export_layer(tile_type, config)
        
        self.print_summary()
    
    def print_summary(self):
        print("\n" + "="*60)
        print("GENERATION SUMMARY")
        print("="*60)
        
        print(f"\nMAP COMPOSITION (based on {self.total_points_processed:,} samples):")
        for tile_type in TileType:
            count = self.type_counts[tile_type]
            percentage = (count / self.total_points_processed) * 100 if self.total_points_processed > 0 else 0
            bar_len = int(percentage / 2)
            bar = "█" * bar_len + "░" * (50 - bar_len)
            print(f"  {tile_type.value:10} {bar} {percentage:5.1f}% ({count:>8,})")
        
        print(f"\nLAYER STATISTICS:")
        total_tiles = 0
        
        for tile_type in TileType:
            tiles = self.tiles_by_layer[tile_type]
            count = len(tiles)
            total_tiles += count
            config = LAYER_CONFIGS[tile_type.value]
            target_points = config["count"]
            
            print(f"\n{tile_type.value.upper()}:")
            print(f"  Points generated: {target_points:>8,}")
            print(f"  Tiles saved:      {count:>8,}")
            print(f"  Retention:        {(count/target_points*100) if target_points > 0 else 0:>7.1f}%")
            
            if count > 0:
                print(f"  ID range:         0 to {count-1:,}")
                sample = tiles[0]
                lat, lon = xyz_to_latlon(sample.x, sample.y, sample.z)
                print(f"  Sample location:  {lat:.1f}°, {lon:.1f}°")
        
        uniform_target = 2048 * 1024
        savings = uniform_target - total_tiles
        print(f"\n{'='*60}")
        print(f"STORAGE COMPARISON:")
        print(f"  Uniform (all tiles):   {uniform_target:>8,} tiles")
        print(f"  Hierarchical (3 layers): {total_tiles:>8,} tiles")
        print(f"  Savings:               {savings:>8,} tiles ({(1 - total_tiles/uniform_target)*100:.1f}% less)")


# ============================================================
# MAIN
# ============================================================

def main():
    color_map = {
        TileType.LAND: LAYER_CONFIGS["land"]["color"],
        TileType.COASTAL: LAYER_CONFIGS["coastal"]["color"],
        TileType.DEEP_SEA: LAYER_CONFIGS["deep_sea"]["color"]
    }
    
    classifier = ThreeColorClassifier(WORLD_MAP_IMAGE, color_map)
    generator = HierarchicalTileGenerator(classifier)
    generator.generate_all_layers()
    
    print("\n✓ Hierarchical world generation complete!")
    print("\nMetadata files created for each layer containing point_count.")


if __name__ == "__main__":
    main()
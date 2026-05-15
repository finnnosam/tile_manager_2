"""
Generates a single unified tile set from a 3-color equirectangular map.
Builds neighbor graph on full point set first, then samples, then fixes low-degree nodes.
"""

import math
import json
import struct
import base64
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set

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

# Base resolution - determines detail level for ALL types
BASE_POINTS = 2048 * 2048

# Sampling rates (keep every Nth point of each type)
SAMPLING_RATES = {
    "land": 1,
    "coastal": 5,
    "deep_sea": 20
}

# Map colors (RGB)
COLOR_MAP = {
    "land": (239, 239, 239),
    "coastal": (172, 202, 202),
    "deep_sea": (105, 165, 165)
}

WORLD_MAP_IMAGE = "terrain_map.png"
NEIGHBOR_COUNT = 6
MIN_NEIGHBORS = 3  # Minimum neighbors required after sampling

# Output files
STATIC_FILE = "tiles_static.json"
DYNAMIC_FILE = "tiles_dynamic.json"
METADATA_FILE = "tiles_metadata.json"


# ============================================================
# TILE DATA STRUCTURES
# ============================================================

@dataclass
class Tile:
    id: int
    type: str  # "land", "coastal", or "deep_sea"
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
# MAP CLASSIFIER
# ============================================================

class MapClassifier:
    def __init__(self, image_path: str):
        print(f"Loading world map: {image_path}")
        self.img = Image.open(image_path).convert("RGB")
        print(f"  Size: {self.img.width}x{self.img.height}")
        
        # Convert RGB colors to type mapping
        self.rgb_to_type = {}
        for tile_type, rgb in COLOR_MAP.items():
            self.rgb_to_type[rgb] = tile_type
            print(f"    {tile_type}: RGB{rgb}")
    
    def classify_point(self, x: float, y: float, z: float) -> str:
        lat, lon = xyz_to_latlon(x, y, z)
        px, py = latlon_to_xy(lat, lon, self.img.width, self.img.height)
        rgb = self.img.getpixel((px, py))
        
        # Find closest matching color
        closest_type = "deep_sea"
        min_dist = float('inf')
        
        for target_rgb, tile_type in self.rgb_to_type.items():
            # Check exact or near match
            if all(abs(rgb[i] - target_rgb[i]) <= 10 for i in range(3)):
                return tile_type
            
            # Track closest for fallback
            dist = sum((rgb[i] - target_rgb[i]) ** 2 for i in range(3))
            if dist < min_dist:
                min_dist = dist
                closest_type = tile_type
        
        return closest_type


# ============================================================
# MAIN GENERATOR
# ============================================================

class TileGenerator:
    def __init__(self, classifier: MapClassifier):
        self.classifier = classifier
        self.all_points: List[Tuple[int, str, float, float, float]] = []  # (id, type, x, y, z)
        self.tiles: List[Tile] = []
        self.type_counts = {"land": 0, "coastal": 0, "deep_sea": 0}
    
    def generate(self):
        print("\n" + "="*60)
        print("TILE GENERATION (PRE-SAMPLING NEIGHBORS)")
        print("="*60)
        print(f"\nBase resolution: {BASE_POINTS:,} points")
        print(f"Minimum neighbors per tile: {MIN_NEIGHBORS}")
        print("\nSampling rates:")
        for type_name, rate in SAMPLING_RATES.items():
            print(f"  {type_name:10}: 1/{rate}")
        
        # Step 1: Generate all points
        print("\n" + "="*60)
        print("STEP 1: GENERATING SPHERE")
        print("="*60)
        all_points = fibonacci_sphere(BASE_POINTS, "  Generating points")
        
        # Step 2: Classify all points
        print("\n" + "="*60)
        print("STEP 2: CLASSIFYING ALL POINTS")
        print("="*60)
        
        bar = ProgressBar(BASE_POINTS, "  Classifying")
        for idx, (x, y, z) in enumerate(all_points):
            tile_type = self.classifier.classify_point(x, y, z)
            self.type_counts[tile_type] += 1
            self.all_points.append((idx, tile_type, x, y, z))
            bar.update(1)
        bar.close()
        
        # Print classification distribution
        print("\n  Classification distribution:")
        for tile_type in ["land", "coastal", "deep_sea"]:
            count = self.type_counts[tile_type]
            pct = (count / BASE_POINTS) * 100
            print(f"    {tile_type:10}: {count:>8,} ({pct:5.1f}%)")
        
        # Step 3: Build neighbor graph on ALL points (PRE-SAMPLING)
        print("\n" + "="*60)
        print("STEP 3: BUILDING NEIGHBOR GRAPH (ALL POINTS)")
        print("="*60)
        all_neighbors = self.build_all_neighbors()
        
        # Step 4: Sample points for final tiles
        print("\n" + "="*60)
        print("STEP 4: SAMPLING POINTS")
        print("="*60)
        self.sample_points(all_neighbors)
        
        # Step 5: Fix low-degree nodes
        print("\n" + "="*60)
        print("STEP 5: FIXING LOW-DEGREE NODES")
        print("="*60)
        self.fix_low_degree_nodes()
        
        # Step 6: Export
        print("\n" + "="*60)
        print("STEP 6: EXPORTING FILES")
        print("="*60)
        self.export()
        
        # Summary
        self.print_summary()
    
    def build_all_neighbors(self) -> Dict[int, Set[int]]:
        """Build neighbor graph for ALL points before sampling."""
        print(f"  Building graph for {len(self.all_points):,} points...")
        
        # Build KD-tree with all points
        positions = [(x, y, z) for _, _, x, y, z in self.all_points]
        tree = cKDTree(positions)
        
        # Build adjacency
        k = NEIGHBOR_COUNT + 1
        adjacency = {point_id: set() for point_id, _, _, _, _ in self.all_points}
        
        bar = ProgressBar(len(self.all_points), "  Processing")
        
        for i, (point_id, _, x, y, z) in enumerate(self.all_points):
            distances, indices = tree.query((x, y, z), k=min(k, len(self.all_points)))
            
            # Skip the first index (itself)
            for neighbor_idx in indices[1:]:
                neighbor_id = self.all_points[neighbor_idx][0]
                adjacency[point_id].add(neighbor_id)
                # Neighbor will add this point when processed
            
            bar.update(1)
        
        bar.close()
        
        # Statistics
        degrees = [len(adjacency[pid]) for pid in adjacency]
        avg_degree = sum(degrees) / len(degrees) if degrees else 0
        print(f"  All points graph statistics:")
        print(f"    Average neighbors: {avg_degree:.2f}")
        print(f"    Min neighbors: {min(degrees)}")
        print(f"    Max neighbors: {max(degrees)}")
        
        return adjacency
    
    def sample_points(self, full_adjacency: Dict[int, Set[int]]):
        """Sample points based on rates, preserving neighbor relationships."""
        print("  Sampling points...")
        
        # Track counters for sampling
        counters = {"land": 0, "coastal": 0, "deep_sea": 0}
        
        bar = ProgressBar(len(self.all_points), "  Processing")
        
        for point_id, tile_type, x, y, z in self.all_points:
            counters[tile_type] += 1
            rate = SAMPLING_RATES[tile_type]
            
            # Sample this point?
            if counters[tile_type] % rate == 0:
                tile = Tile(
                    id=point_id,
                    type=tile_type,
                    x=x, y=y, z=z
                )
                # Copy neighbors, filtering to only those that will also be sampled
                # We'll fix this in the next step
                self.tiles.append(tile)
            
            bar.update(1)
        
        bar.close()
        
        # Create lookup for sampled points
        sampled_ids = {t.id for t in self.tiles}
        
        # Filter neighbor lists to only include sampled points
        sampled_adjacency = {}
        for tile in self.tiles:
            # Get all neighbors from full graph
            full_neighbors = full_adjacency.get(tile.id, set())
            # Keep only those that were also sampled
            filtered_neighbors = full_neighbors & sampled_ids
            sampled_adjacency[tile.id] = filtered_neighbors
        
        # Apply filtered neighbors to tiles
        for tile in self.tiles:
            tile.neighbors = sorted(sampled_adjacency.get(tile.id, set()))
        
        # Statistics after sampling
        degrees = [len(t.neighbors) for t in self.tiles]
        print(f"\n  After sampling (before fixing):")
        print(f"    Sampled tiles: {len(self.tiles):,}")
        print(f"    Average neighbors: {sum(degrees)/len(degrees):.2f}")
        print(f"    Min neighbors: {min(degrees)}")
        print(f"    Max neighbors: {max(degrees)}")
        
        low_degree = sum(1 for d in degrees if d < MIN_NEIGHBORS)
        print(f"    Tiles with <{MIN_NEIGHBORS} neighbors: {low_degree:,}")
    
    def fix_low_degree_nodes(self):
        """Optimized: Find tiles with fewer than MIN_NEIGHBORS and add nearest points."""
        if not self.tiles:
            return
        
        # Create position array and ID mapping for fast lookups
        positions = []
        tile_ids = []
        tile_index_map = {}  # id -> index in positions array
        tile_neighbors_map = {t.id: set(t.neighbors) for t in self.tiles}
        
        for idx, tile in enumerate(self.tiles):
            positions.append([tile.x, tile.y, tile.z])
            tile_ids.append(tile.id)
            tile_index_map[tile.id] = idx
        
        # Build KD-tree once
        tree = cKDTree(positions)
        
        # Find tiles that need fixing
        needs_fixing = [t for t in self.tiles if len(t.neighbors) < MIN_NEIGHBORS]
        
        if not needs_fixing:
            print("  No low-degree nodes found!")
            return
        
        print(f"  Fixing {len(needs_fixing):,} low-degree nodes...")
        
        bar = ProgressBar(len(needs_fixing), "  Fixing")
        
        for tile in needs_fixing:
            current_neighbors = tile_neighbors_map[tile.id]
            needed = MIN_NEIGHBORS - len(current_neighbors)
            
            if needed <= 0:
                continue
            
            # Query more candidates than needed to have options
            k = min(needed * 5 + 10, len(self.tiles))
            distances, indices = tree.query((tile.x, tile.y, tile.z), k=k)
            
            added = 0
            for neighbor_idx in indices:
                if added >= needed:
                    break
                
                neighbor_id = tile_ids[neighbor_idx]
                
                # Skip self and existing neighbors
                if neighbor_id == tile.id or neighbor_id in current_neighbors:
                    continue
                
                # Add neighbor relationship
                current_neighbors.add(neighbor_id)
                
                # Add reverse relationship (using the map for O(1) access)
                tile_neighbors_map[neighbor_id].add(tile.id)
                
                added += 1
            
            bar.update(1)
        
        bar.close()
        
        # Convert back to sorted lists
        for tile in self.tiles:
            tile.neighbors = sorted(tile_neighbors_map[tile.id])
        
        # Statistics after fixing
        degrees = [len(t.neighbors) for t in self.tiles]
        print(f"\n  After fixing low-degree nodes:")
        print(f"    Average neighbors: {sum(degrees)/len(degrees):.2f}")
        print(f"    Min neighbors: {min(degrees)}")
        print(f"    Max neighbors: {max(degrees)}")
        
        still_low = sum(1 for d in degrees if d < MIN_NEIGHBORS)
        if still_low > 0:
            print(f"    WARNING: {still_low:,} tiles still have <{MIN_NEIGHBORS} neighbors")
    
    def export(self):
        # Prepare static data
        static_data = []
        dynamic_data = []
        
        bar = ProgressBar(len(self.tiles), "  Exporting")
        
        for tile in self.tiles:
            static_data.append({
                "id": tile.id,
                "type": tile.type,
                "neighbors_b64": encode_neighbors(tile.neighbors),
                "temperature": tile.temperature,
                "precipitation": tile.precipitation,
                "terrain": tile.terrain
            })
            
            dynamic_data.append({
                "id": tile.id,
                "vegetation": tile.vegetation,
                "population": tile.population,
                "buildings": tile.buildings
            })
            
            bar.update(1)
        
        bar.close()
        
        # Save static data
        with open(STATIC_FILE, "w") as f:
            json.dump(static_data, f, indent=2)
        
        # Save dynamic data
        with open(DYNAMIC_FILE, "w") as f:
            json.dump(dynamic_data, f, indent=2)
        
        # Save metadata
        metadata = {
            "total_tiles": len(self.tiles),
            "base_points": BASE_POINTS,
            "neighbor_count": NEIGHBOR_COUNT,
            "min_neighbors": MIN_NEIGHBORS,
            "sampling_rates": SAMPLING_RATES,
            "tiles_by_type": {
                t_type: sum(1 for t in self.tiles if t.type == t_type)
                for t_type in ["land", "coastal", "deep_sea"]
            },
            "total_by_type": self.type_counts
        }
        
        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n  Exported:")
        print(f"    {STATIC_FILE}: {len(static_data):,} tiles")
        print(f"    {DYNAMIC_FILE}: {len(dynamic_data):,} tiles")
        print(f"    {METADATA_FILE}: metadata")
    
    def print_summary(self):
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        total_sampled = len(self.tiles)
        total_possible = BASE_POINTS
        
        print(f"\n  Base points:      {total_possible:>8,}")
        print(f"  Sampled tiles:    {total_sampled:>8,}")
        print(f"  Savings:          {total_possible - total_sampled:>8,} ({(1 - total_sampled/total_possible)*100:.1f}%)")
        
        print(f"\n  Tiles by type:")
        for t_type in ["land", "coastal", "deep_sea"]:
            total = self.type_counts[t_type]
            sampled = sum(1 for t in self.tiles if t.type == t_type)
            rate = SAMPLING_RATES[t_type]
            print(f"    {t_type:10}: {sampled:>6,}/{total:>6,} (1/{rate})")
        
        # Neighbor statistics
        degrees = [len(t.neighbors) for t in self.tiles]
        print(f"\n  Final neighbor statistics:")
        print(f"    Average: {sum(degrees)/len(degrees):.2f}")
        print(f"    Min: {min(degrees)}")
        print(f"    Max: {max(degrees)}")


# ============================================================
# MAIN
# ============================================================

def main():
    classifier = MapClassifier(WORLD_MAP_IMAGE)
    generator = TileGenerator(classifier)
    generator.generate()
    
    print("\n✓ Generation complete!")
    print(f"  Neighbors were built on full point set first")
    print(f"  Low-degree nodes (below {MIN_NEIGHBORS}) were fixed")


if __name__ == "__main__":
    main()
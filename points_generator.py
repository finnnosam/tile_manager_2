"""
Generates a single unified tile set.
Builds neighbor graph on full point set.
"""

import math
import json
import struct
import base64
import time
from dataclasses import dataclass, field
from typing import List, Tuple

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

# Base resolution - determines number of points
BASE_POINTS = 1024 * 512

NEIGHBOR_COUNT = 6

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


# ============================================================
# MAIN GENERATOR
# ============================================================

class TileGenerator:
    def __init__(self):
        self.tiles: List[Tile] = []
    
    def generate(self):
        print("\n" + "="*60)
        print("TILE GENERATION")
        print("="*60)
        print(f"\nTotal points: {BASE_POINTS:,}")
        print(f"Neighbors per tile: {NEIGHBOR_COUNT}")
        
        # Step 1: Generate all points
        print("\n" + "="*60)
        print("STEP 1: GENERATING SPHERE")
        print("="*60)
        all_points = fibonacci_sphere(BASE_POINTS, "  Generating points")
        
        # Step 2: Build tiles
        print("\n" + "="*60)
        print("STEP 2: BUILDING TILES")
        print("="*60)
        
        bar = ProgressBar(BASE_POINTS, "  Processing")
        for idx, (x, y, z) in enumerate(all_points):
            tile = Tile(
                id=idx,
                x=x, y=y, z=z
            )
            self.tiles.append(tile)
            bar.update(1)
        bar.close()
        
        # Step 3: Build neighbor graph
        print("\n" + "="*60)
        print("STEP 3: BUILDING NEIGHBOR GRAPH")
        print("="*60)
        self.build_neighbors()
        
        # Step 4: Export
        print("\n" + "="*60)
        print("STEP 4: EXPORTING FILES")
        print("="*60)
        self.export()
        
        # Summary
        self.print_summary()
    
    def build_neighbors(self):
        """Build neighbor graph for all points."""
        print(f"  Building graph for {len(self.tiles):,} points...")
        
        # Build KD-tree with all points
        positions = [(tile.x, tile.y, tile.z) for tile in self.tiles]
        tree = cKDTree(positions)
        
        # Build adjacency
        k = NEIGHBOR_COUNT + 1
        adjacency = {tile.id: set() for tile in self.tiles}
        
        bar = ProgressBar(len(self.tiles), "  Processing")
        
        for i, tile in enumerate(self.tiles):
            distances, indices = tree.query((tile.x, tile.y, tile.z), k=min(k, len(self.tiles)))
            
            # Skip the first index (itself)
            for neighbor_idx in indices[1:]:
                neighbor_id = self.tiles[neighbor_idx].id
                adjacency[tile.id].add(neighbor_id)
            
            bar.update(1)
        
        bar.close()
        
        # Apply neighbors to tiles
        for tile in self.tiles:
            tile.neighbors = sorted(adjacency[tile.id])
        
        # Statistics
        degrees = [len(tile.neighbors) for tile in self.tiles]
        avg_degree = sum(degrees) / len(degrees) if degrees else 0
        print(f"\n  Graph statistics:")
        print(f"    Average neighbors: {avg_degree:.2f}")
        print(f"    Min neighbors: {min(degrees)}")
        print(f"    Max neighbors: {max(degrees)}")
    
    def export(self):
        # Prepare static data
        static_data = []
        dynamic_data = []
        
        bar = ProgressBar(len(self.tiles), "  Exporting")
        
        for tile in self.tiles:
            static_data.append({
                "id": tile.id,
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
            "neighbor_count": NEIGHBOR_COUNT
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
        
        print(f"\n  Total tiles generated: {len(self.tiles):,}")
        
        # Neighbor statistics
        degrees = [len(t.neighbors) for t in self.tiles]
        print(f"\n  Neighbor statistics:")
        print(f"    Average: {sum(degrees)/len(degrees):.2f}")
        print(f"    Min: {min(degrees)}")
        print(f"    Max: {max(degrees)}")


# ============================================================
# MAIN
# ============================================================

def main():
    generator = TileGenerator()
    generator.generate()
    
    print("\n✓ Generation complete!")


if __name__ == "__main__":
    main()
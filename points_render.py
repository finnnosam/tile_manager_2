"""
render_hierarchical_world_fast.py

Fast renderer that reads point counts from metadata files.
No manual configuration needed!
"""

import json
import math
import struct
import base64
import time
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from typing import List, Dict, Tuple, Optional


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
# CONFIGURATION - Now minimal!
# ============================================================

# Only need to specify the file names - point counts are read from metadata!
LAYER_CONFIGS = {
    "land": {
        "static_file": "land_static.json",
        "dynamic_file": "land_dynamic.json",
        "metadata_file": "land_metadata.json"
    },
    "coastal": {
        "static_file": "coastal_static.json",
        "dynamic_file": "coastal_dynamic.json",
        "metadata_file": "coastal_metadata.json"
    },
    "deep_sea": {
        "static_file": "deep_sea_static.json",
        "dynamic_file": "deep_sea_dynamic.json",
        "metadata_file": "deep_sea_metadata.json"
    }
}

IMAGE_HEIGHT = 4096
IMAGE_WIDTH = IMAGE_HEIGHT * 2
RENDER_MODE = "random"  # "type" or "random"

if RENDER_MODE == "type":
    OUTPUT_FILE = "hierarchical_world_type.png"
elif RENDER_MODE == "random":
    OUTPUT_FILE = "hierarchical_world_random.png"
else:
    import sys
    print("Incorrect RENDER_MODE")
    sys.exit()

RENDER_LAND = True
RENDER_COASTAL = True
RENDER_DEEP_SEA = True

TYPE_COLORS = {
    "land": (34, 139, 34),
    "coastal": (135, 206, 235),
    "deep_sea": (25, 25, 112),
    "unknown": (89, 89, 89)
}


# ============================================================
# METADATA LOADING
# ============================================================

def load_metadata(metadata_file: str) -> Optional[dict]:
    """Load metadata from JSON file."""
    try:
        with open(metadata_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"  Warning: {metadata_file} not found")
        return None
    except json.JSONDecodeError:
        print(f"  Warning: {metadata_file} is corrupted")
        return None


# ============================================================
# UTILITIES
# ============================================================

def decode_neighbors(encoded: str) -> list:
    if not encoded:
        return []
    binary = base64.b64decode(encoded)
    count = len(binary) // 4
    return list(struct.unpack(f"<{count}I", binary))

def load_layer(static_file: str, dynamic_file: str, metadata_file: str, layer_name: str) -> Tuple[List[dict], int]:
    """Load a layer and return (tiles, point_count)."""
    # Load metadata first to get point_count
    metadata = load_metadata(metadata_file)
    if not metadata:
        print(f"  ERROR: Cannot load {metadata_file} - skipping {layer_name}")
        return [], 0
    
    point_count = metadata.get("point_count", 0)
    if point_count == 0:
        print(f"  ERROR: No point_count in {metadata_file} - skipping {layer_name}")
        return [], 0
    
    print(f"  Metadata: point_count={point_count:,}, tile_count={metadata.get('tile_count', 0):,}")
    
    # Load static data
    try:
        with open(static_file, "r") as f:
            static_data = json.load(f)
        with open(dynamic_file, "r") as f:
            dynamic_data = json.load(f)
    except FileNotFoundError as e:
        print(f"  ERROR: {e.filename} not found - skipping {layer_name}")
        return [], 0
    
    dyn_lookup = {item["id"]: item for item in dynamic_data}
    tiles = []
    
    bar = ProgressBar(len(static_data), f"  Loading {layer_name}")
    
    for s in static_data:
        tid = s["id"]
        d = dyn_lookup.get(tid, {})
        tiles.append({
            "id": tid,
            "layer": s.get("layer", layer_name),
            "neighbors": decode_neighbors(s.get("neighbors_b64", "")),
        })
        bar.update(1)
    
    bar.close()
    tiles.sort(key=lambda t: t["id"])
    
    return tiles, point_count

def load_all_layers() -> Tuple[List[dict], Dict[str, int]]:
    """Load all layers and return (tiles, point_counts_by_layer)."""
    all_tiles = []
    point_counts = {}
    
    for layer_name, config in LAYER_CONFIGS.items():
        if layer_name == "land" and not RENDER_LAND:
            continue
        if layer_name == "coastal" and not RENDER_COASTAL:
            continue
        if layer_name == "deep_sea" and not RENDER_DEEP_SEA:
            continue
        
        print(f"\nLoading {layer_name} layer...")
        tiles, point_count = load_layer(
            config["static_file"], 
            config["dynamic_file"],
            config["metadata_file"],
            layer_name
        )
        
        if not tiles:
            print(f"  No tiles loaded for {layer_name}")
            continue
        
        print(f"  Loaded {len(tiles):,} tiles")
        point_counts[layer_name] = point_count
        
        regenerate_positions_for_layer(tiles, point_count, layer_name)
        all_tiles.extend(tiles)
    
    return all_tiles, point_counts


# ============================================================
# FIBONACCI SPHERE & POSITIONS
# ============================================================

def fibonacci_sphere(samples: int) -> np.ndarray:
    """Generate Fibonacci sphere points as numpy array."""
    phi = (1 + math.sqrt(5)) / 2
    golden_angle = 2 * math.pi / phi
    i = np.arange(samples)
    y = 1 - (2*i + 1) / samples
    radius = np.sqrt(1 - y*y)
    theta = golden_angle * i
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius
    return np.column_stack((x, y, z))

def regenerate_positions_for_layer(tiles: List[dict], total_points: int, layer_name: str):
    """Regenerate positions for a single layer."""
    if not tiles:
        return
    
    print(f"  Regenerating positions for {len(tiles):,} tiles...")
    positions = fibonacci_sphere(total_points)
    
    bar = ProgressBar(len(tiles), "  Assigning positions")
    
    missing = 0
    for tile in tiles:
        idx = tile["id"]
        if idx < total_points:
            tile["x"], tile["y"], tile["z"] = positions[idx]
        else:
            missing += 1
            tile["x"], tile["y"], tile["z"] = 0.0, 0.0, 1.0
        bar.update(1)
    
    bar.close()
    
    if missing > 0:
        print(f"    Warning: {missing} tiles had IDs exceeding point count")
    
    if len(tiles) > 0:
        sample_ids = [t["id"] for t in tiles[:5]]
        print(f"    Sample IDs: {sample_ids}")


# ============================================================
# RENDERING (OPTIMIZED)
# ============================================================

def image_xy_to_sphere_batch(width: int, height: int) -> np.ndarray:
    """Generate sphere coordinates for all pixels."""
    print(f"\nGenerating pixel coordinates...")
    bar = ProgressBar(height, "  Building coordinate grid")
    
    x_img = np.arange(width)
    y_img = np.arange(height)
    xx, yy = np.meshgrid(x_img, y_img)
    bar.update(height)
    bar.close()
    
    print("  Converting to sphere coordinates...")
    lon = (xx / width) * 360.0 - 180.0
    lat = 90.0 - (yy / height) * 180.0
    
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    y = np.sin(lat_rad)
    r = np.cos(lat_rad)
    x = np.cos(lon_rad) * r
    z = np.sin(lon_rad) * r
    
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    print(f"  Generated {len(points):,} points")
    return points

def render_world_fast(tiles: List[dict], point_counts: Dict[str, int]):
    """Main rendering function."""
    print("\n" + "="*60)
    print(f"FAST RENDER - Mode: {RENDER_MODE}")
    print(f"Total tiles: {len(tiles):,}")
    print("="*60)
    
    if not tiles:
        print("No tiles to render!")
        return
    
    # Display point counts from metadata
    print("\nPoint counts (from metadata):")
    for layer, count in point_counts.items():
        print(f"  {layer}: {count:,} Fibonacci points")
    
    # Statistics
    layer_counts = {}
    for tile in tiles:
        layer = tile.get("layer", "unknown")
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
    
    print(f"\nTile counts by layer:")
    for layer, count in sorted(layer_counts.items()):
        print(f"  {layer}: {count:,} tiles")
    
    all_ids = [t["id"] for t in tiles]
    print(f"\nID statistics:")
    print(f"  Min ID: {min(all_ids):,}")
    print(f"  Max ID: {max(all_ids):,}")
    print(f"  Actual tiles: {len(tiles):,}")
    
    # Build KD-tree
    print("\nBuilding KD-tree...")
    positions = np.array([[t["x"], t["y"], t["z"]] for t in tiles], dtype=np.float64)
    
    bar = ProgressBar(len(positions), "  Building position array")
    for i in range(len(positions)):
        bar.update(1)
    bar.close()
    
    tree = cKDTree(positions)
    print(f"  KD-tree built with {len(positions):,} points")
    
    # Generate query points
    query_points = image_xy_to_sphere_batch(IMAGE_WIDTH, IMAGE_HEIGHT)
    
    # Batch query
    print("\nQuerying nearest tiles...")
    query_bar = ProgressBar(len(query_points), "  KD-tree query", width=40)
    
    chunk_size = 1000000
    all_indices = []
    
    for i in range(0, len(query_points), chunk_size):
        chunk = query_points[i:i+chunk_size]
        distances, indices = tree.query(chunk, k=1)
        all_indices.extend(indices)
        query_bar.update(len(chunk))
    
    query_bar.close()
    indices = np.array(all_indices, dtype=np.int32)
    
    # Build color array
    print("\nBuilding color array...")
    
    if RENDER_MODE == "type":
        # Create color lookup table indexed by tile position
        print(f"  Creating color lookup table for {len(tiles):,} tiles...")
        tile_colors = np.zeros((len(tiles), 3), dtype=np.uint8)
        
        color_bar = ProgressBar(len(tiles), "  Building color lookup")
        for i, tile in enumerate(tiles):
            layer = tile.get("layer", "unknown")
            rgb = TYPE_COLORS.get(layer, TYPE_COLORS["unknown"])
            tile_colors[i] = rgb
            color_bar.update(1)
        color_bar.close()
        
        # Fast vectorized lookup
        print(f"  Applying colors to {len(indices):,} pixels (vectorized)...")
        img_array = tile_colors[indices]
        
    elif RENDER_MODE == "random":
        # Find max ID across all tiles
        max_id = max(t["id"] for t in tiles)
        print(f"  Creating random colors for IDs 0..{max_id:,}")
        
        random_bar = ProgressBar(max_id + 1, "  Generating random colors", width=40)
        
        rng = np.random.RandomState(42)
        random_colors = np.zeros((max_id + 1, 3), dtype=np.uint8)
        
        chunk = 100000
        for start in range(0, max_id + 1, chunk):
            end = min(start + chunk, max_id + 1)
            random_colors[start:end] = rng.randint(50, 256, size=(end - start, 3), dtype=np.uint8)
            random_bar.update(end - start)
        
        random_bar.close()
        
        # Build tile color lookup
        print(f"  Building tile color lookup...")
        tile_colors = np.zeros((len(tiles), 3), dtype=np.uint8)
        color_bar = ProgressBar(len(tiles), "  Mapping tile IDs to colors")
        
        for i, tile in enumerate(tiles):
            tile_colors[i] = random_colors[tile["id"]]
            color_bar.update(1)
        color_bar.close()
        
        # Apply to pixels
        print(f"  Applying colors to {len(indices):,} pixels...")
        img_array = tile_colors[indices]
    
    else:
        raise ValueError(f"Invalid render mode: {RENDER_MODE}")
    
    # Reshape to 2D image
    print("\nReshaping to final image...")
    img_array = img_array.reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    
    # Create and save
    print("Creating PIL image...")
    image = Image.fromarray(img_array, "RGB")
    
    print(f"Saving {OUTPUT_FILE}...")
    image.save(OUTPUT_FILE)
    
    print("\n" + "="*60)
    print("✓ Render complete!")
    print("="*60)


# ============================================================
# MAIN
# ============================================================

def main():
    print("="*60)
    print("HIERARCHICAL WORLD RENDERER (FAST)")
    print("Reading point counts from metadata files...")
    print("="*60)
    
    start_time = time.time()
    
    tiles, point_counts = load_all_layers()
    if not tiles:
        print("No tiles loaded. Check that layer files exist and RENDER_* flags are set.")
        return
    
    render_world_fast(tiles, point_counts)
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
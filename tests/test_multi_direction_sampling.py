"""
Test multi-directional sampling for medial axis centering on TS2 skeleton.
"""

import numpy as np
from mcf2swc import (
    PolylinesSkeleton,
    MeshManager,
    SkeletonOptimizer,
    SkeletonOptimizerOptions,
)

# Load TS2 mesh and skeleton
print("Loading TS2 mesh and skeleton...")
mesh_mgr = MeshManager(mesh_path="data/mesh/processed/TS2.obj")
skeleton = PolylinesSkeleton.from_txt(
    "data/mcf_skeletons/TS2_qst0.6_mcst5.polylines.txt"
)

# Prune short branches first
print("\nPruning short branches...")
skeleton_pruned = skeleton.prune_short_branches(min_length=10.0, verbose=False)
print(
    f"Pruned skeleton: {len(skeleton_pruned.polylines)} polylines, {skeleton_pruned.total_points()} points"
)

# Test different numbers of probe directions
test_configs = [
    {"name": "2 directions (opposite pair)", "num_dirs": 2},
    {"name": "4 directions (2 perpendicular pairs)", "num_dirs": 4},
    {"name": "8 directions", "num_dirs": 8},
    {"name": "16 directions", "num_dirs": 16},
]

results = []

for config in test_configs:
    print("\n" + "=" * 70)
    print(f"Testing: {config['name']}")
    print("=" * 70)

    options = SkeletonOptimizerOptions(
        centering_method="medial_axis",
        probe_distance=20.0,
        num_probe_directions=config["num_dirs"],
        max_iterations=50,
        step_size=0.1,
        preserve_endpoints=True,
        smoothing_weight=0.3,
        verbose=False,
    )

    optimizer = SkeletonOptimizer(skeleton_pruned, mesh_mgr.mesh, options)
    optimized = optimizer.optimize()

    # Compute statistics
    all_points = np.vstack(optimized.polylines)
    distances = mesh_mgr.mesh.nearest.signed_distance(all_points)
    inside = mesh_mgr.mesh.contains(all_points)

    result = {
        "name": config["name"],
        "num_dirs": config["num_dirs"],
        "points_inside": inside.sum(),
        "total_points": len(inside),
        "mean_distance": np.abs(distances).mean(),
        "max_distance": np.abs(distances).max(),
        "std_distance": np.abs(distances).std(),
    }
    results.append(result)

    print(f"  Points inside mesh: {result['points_inside']}/{result['total_points']}")
    print(f"  Mean distance to surface: {result['mean_distance']:.4f}")
    print(f"  Max distance to surface: {result['max_distance']:.4f}")
    print(f"  Std distance to surface: {result['std_distance']:.4f}")

# Summary comparison
print("\n" + "=" * 70)
print("Summary Comparison")
print("=" * 70)
print(f"{'Configuration':<40} {'Mean Dist':>12} {'Max Dist':>12} {'Std Dist':>12}")
print("-" * 70)
for r in results:
    print(
        f"{r['name']:<40} {r['mean_distance']:>12.4f} {r['max_distance']:>12.4f} {r['std_distance']:>12.4f}"
    )

# Find best configuration
best_idx = np.argmin([r["mean_distance"] for r in results])
print(f"\nBest configuration: {results[best_idx]['name']}")
print(
    f"  Mean distance improvement over 2 directions: {results[0]['mean_distance'] - results[best_idx]['mean_distance']:.4f}"
)

# Save best result
if best_idx > 0:  # If not the first one
    print(f"\nRe-running best configuration and saving...")
    best_config = test_configs[best_idx]
    options = SkeletonOptimizerOptions(
        centering_method="medial_axis",
        probe_distance=20.0,
        num_probe_directions=best_config["num_dirs"],
        max_iterations=50,
        step_size=0.1,
        preserve_endpoints=True,
        smoothing_weight=0.3,
        verbose=False,
    )
    optimizer = SkeletonOptimizer(skeleton_pruned, mesh_mgr.mesh, options)
    optimized_best = optimizer.optimize()
    optimized_best.to_txt("data/mcf_skeletons/TS2_optimized_multi_dir.polylines.txt")
    print("Saved to: data/mcf_skeletons/TS2_optimized_multi_dir.polylines.txt")

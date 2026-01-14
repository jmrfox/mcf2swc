"""
Quick test of multi-directional sampling for medial axis centering.
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
print("Pruning short branches...")
skeleton_pruned = skeleton.prune_short_branches(min_length=10.0, verbose=False)
print(
    f"Pruned skeleton: {len(skeleton_pruned.polylines)} polylines, {skeleton_pruned.total_points()} points\n"
)

# Test 4 directions vs 8 directions
test_configs = [
    {"name": "4 directions", "num_dirs": 4},
    {"name": "8 directions", "num_dirs": 8},
]

results = []

for config in test_configs:
    print("=" * 70)
    print(f"Testing: {config['name']}")
    print("=" * 70)

    options = SkeletonOptimizerOptions(
        centering_method="medial_axis",
        probe_distance=20.0,
        num_probe_directions=config["num_dirs"],
        max_iterations=30,  # Reduced iterations for speed
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
        "mean_distance": np.abs(distances).mean(),
        "max_distance": np.abs(distances).max(),
    }
    results.append(result)

    print(f"Points inside: {inside.sum()}/{len(inside)}")
    print(f"Mean distance: {result['mean_distance']:.4f}")
    print(f"Max distance: {result['max_distance']:.4f}\n")

# Comparison
print("=" * 70)
print("Comparison")
print("=" * 70)
improvement = results[0]["mean_distance"] - results[1]["mean_distance"]
print(f"4 directions mean distance: {results[0]['mean_distance']:.4f}")
print(f"8 directions mean distance: {results[1]['mean_distance']:.4f}")
print(f"Improvement with 8 directions: {improvement:.4f}")
if improvement > 0:
    print("✓ 8 directions is better (more robust sampling)")
else:
    print("✗ 4 directions is sufficient")

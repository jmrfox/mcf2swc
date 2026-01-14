"""
Test ray tracing-based medial axis centering (no probe_distance needed).
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

# Test ray tracing-based optimization
print("=" * 70)
print("Testing ray tracing-based medial axis centering")
print("=" * 70)

options = SkeletonOptimizerOptions(
    centering_method="medial_axis",
    num_probe_directions=4,
    max_iterations=30,
    step_size=0.1,
    preserve_endpoints=True,
    smoothing_weight=0.3,
    verbose=True,
)

optimizer = SkeletonOptimizer(skeleton_pruned, mesh_mgr.mesh, options)
optimized = optimizer.optimize()

# Compute statistics
all_points = np.vstack(optimized.polylines)
distances = mesh_mgr.mesh.nearest.signed_distance(all_points)
inside = mesh_mgr.mesh.contains(all_points)

print(f"\nResults:")
print(f"  Points inside mesh: {inside.sum()}/{len(inside)}")
print(f"  Mean distance to surface: {np.abs(distances).mean():.4f}")
print(f"  Max distance to surface: {np.abs(distances).max():.4f}")
print(f"  Std distance to surface: {np.abs(distances).std():.4f}")

# Save result
optimized.to_txt("data/mcf_skeletons/TS2_optimized_ray_tracing.polylines.txt")
print("\nSaved to: data/mcf_skeletons/TS2_optimized_ray_tracing.polylines.txt")

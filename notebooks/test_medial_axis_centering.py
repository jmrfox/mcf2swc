"""
Test the new medial axis centering method on TS2 skeleton.
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
skeleton_pruned = skeleton.prune_short_branches(min_length=10.0, verbose=True)

print(
    f"\nPruned skeleton: {len(skeleton_pruned.polylines)} polylines, {skeleton_pruned.total_points()} points"
)

# Test 1: Optimize with closest_point method
print("\n" + "=" * 70)
print("Test 1: Optimization with 'closest_point' centering")
print("=" * 70)

options_cp = SkeletonOptimizerOptions(
    centering_method="closest_point",
    max_iterations=50,
    step_size=0.1,
    preserve_endpoints=True,
    smoothing_weight=0.3,
    verbose=True,
)

optimizer_cp = SkeletonOptimizer(skeleton_pruned, mesh_mgr.mesh, options_cp)
optimized_cp = optimizer_cp.optimize()

# Compute statistics
all_points_cp = np.vstack(optimized_cp.polylines)
distances_cp = mesh_mgr.mesh.nearest.signed_distance(all_points_cp)
inside_cp = mesh_mgr.mesh.contains(all_points_cp)

print(f"\nResults (closest_point):")
print(f"  Points inside mesh: {inside_cp.sum()}/{len(inside_cp)}")
print(f"  Mean distance to surface: {np.abs(distances_cp).mean():.4f}")
print(f"  Max distance to surface: {np.abs(distances_cp).max():.4f}")

# Test 2: Optimize with medial_axis method
print("\n" + "=" * 70)
print("Test 2: Optimization with 'medial_axis' centering")
print("=" * 70)

options_ma = SkeletonOptimizerOptions(
    centering_method="medial_axis",
    probe_distance=20.0,  # Adjust based on mesh size
    max_iterations=50,
    step_size=0.1,
    preserve_endpoints=True,
    smoothing_weight=0.3,
    verbose=True,
)

optimizer_ma = SkeletonOptimizer(skeleton_pruned, mesh_mgr.mesh, options_ma)
optimized_ma = optimizer_ma.optimize()

# Compute statistics
all_points_ma = np.vstack(optimized_ma.polylines)
distances_ma = mesh_mgr.mesh.nearest.signed_distance(all_points_ma)
inside_ma = mesh_mgr.mesh.contains(all_points_ma)

print(f"\nResults (medial_axis):")
print(f"  Points inside mesh: {inside_ma.sum()}/{len(inside_ma)}")
print(f"  Mean distance to surface: {np.abs(distances_ma).mean():.4f}")
print(f"  Max distance to surface: {np.abs(distances_ma).max():.4f}")

# Comparison
print("\n" + "=" * 70)
print("Comparison")
print("=" * 70)
print(
    f"Mean distance improvement: {np.abs(distances_cp).mean() - np.abs(distances_ma).mean():.4f}"
)
print(f"  (positive = medial_axis is better)")

# Save results
print("\nSaving optimized skeletons...")
optimized_cp.to_txt("data/mcf_skeletons/TS2_optimized_closest_point.polylines.txt")
optimized_ma.to_txt("data/mcf_skeletons/TS2_optimized_medial_axis.polylines.txt")
print("Saved:")
print("  - data/mcf_skeletons/TS2_optimized_closest_point.polylines.txt")
print("  - data/mcf_skeletons/TS2_optimized_medial_axis.polylines.txt")

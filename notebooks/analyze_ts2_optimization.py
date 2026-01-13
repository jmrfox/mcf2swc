"""
Analyze TS2 skeleton optimization to understand why some regions don't optimize well.
"""

import numpy as np
from mcf2swc import (
    PolylinesSkeleton,
    MeshManager,
    SkeletonOptimizer,
    SkeletonOptimizerOptions,
)

# Load TS2 mesh and skeleton
print("Loading TS2 data...")
mesh_mgr = MeshManager(mesh_path="data/mesh/processed/TS2.obj")
skeleton = PolylinesSkeleton.from_txt(
    "data/mcf_skeletons/TS2_qst0.6_mcst5.polylines.txt"
)

print(f"\nMesh properties:")
print(f"  Vertices: {len(mesh_mgr.mesh.vertices)}")
print(f"  Faces: {len(mesh_mgr.mesh.faces)}")
print(f"  Volume: {mesh_mgr.mesh.volume:.2f}")
print(f"  Watertight: {mesh_mgr.mesh.is_watertight}")
print(f"  Bounds: {mesh_mgr.mesh.bounds}")

print(f"\nSkeleton properties:")
print(f"  Number of polylines: {len(skeleton.polylines)}")
print(f"  Total points: {skeleton.total_points()}")
for i, pl in enumerate(skeleton.polylines):
    print(f"  Polyline {i}: {len(pl)} points")

# Check which points are inside vs outside
print("\n" + "=" * 70)
print("Analyzing point containment...")
print("=" * 70)

all_pts = np.vstack(skeleton.polylines)
inside_mask = mesh_mgr.mesh.contains(all_pts)
num_inside = np.sum(inside_mask)
num_outside = np.sum(~inside_mask)

print(f"Points inside mesh: {num_inside}/{len(all_pts)}")
print(f"Points outside mesh: {num_outside}/{len(all_pts)}")

# Analyze distance to surface for all points
from trimesh.proximity import closest_point

cp, distances, _ = closest_point(mesh_mgr.mesh, all_pts)

print(f"\nDistance to surface statistics:")
print(f"  Mean distance: {distances.mean():.4f}")
print(f"  Median distance: {np.median(distances):.4f}")
print(f"  Max distance: {distances.max():.4f}")
print(f"  Min distance: {distances.min():.4f}")
print(f"  Std deviation: {distances.std():.4f}")

# Analyze per-polyline
print("\n" + "=" * 70)
print("Per-polyline analysis:")
print("=" * 70)

start_idx = 0
for i, pl in enumerate(skeleton.polylines):
    end_idx = start_idx + len(pl)
    pl_inside = inside_mask[start_idx:end_idx]
    pl_distances = distances[start_idx:end_idx]

    print(f"\nPolyline {i} ({len(pl)} points):")
    print(f"  Inside: {np.sum(pl_inside)}/{len(pl)}")
    print(f"  Outside: {np.sum(~pl_inside)}/{len(pl)}")
    print(f"  Mean dist to surface: {pl_distances.mean():.4f}")
    print(f"  Max dist to surface: {pl_distances.max():.4f}")
    print(f"  Min dist to surface: {pl_distances.min():.4f}")

    # Find points far from surface
    far_threshold = 10.0  # Adjust based on mesh scale
    far_points = pl_distances > far_threshold
    if np.any(far_points):
        print(f"  Points >10 units from surface: {np.sum(far_points)}")
        far_indices = np.where(far_points)[0]
        print(f"    Indices: {far_indices.tolist()}")

    start_idx = end_idx

# Run optimization with verbose output
print("\n" + "=" * 70)
print("Running optimization...")
print("=" * 70)

opts = SkeletonOptimizerOptions(
    max_iterations=100,
    step_size=0.1,
    convergence_threshold=1e-4,
    preserve_endpoints=True,
    smoothing_weight=0.5,
    verbose=True,
)

optimizer = SkeletonOptimizer(skeleton, mesh_mgr.mesh, opts)
optimized_skeleton = optimizer.optimize()

# Analyze optimization results
print("\n" + "=" * 70)
print("Optimization results:")
print("=" * 70)

all_pts_opt = np.vstack(optimized_skeleton.polylines)
inside_mask_opt = mesh_mgr.mesh.contains(all_pts_opt)
cp_opt, distances_opt, _ = closest_point(mesh_mgr.mesh, all_pts_opt)

print(f"\nAfter optimization:")
print(f"  Points inside: {np.sum(inside_mask_opt)}/{len(all_pts_opt)}")
print(f"  Points outside: {np.sum(~inside_mask_opt)}/{len(all_pts_opt)}")
print(f"  Mean dist to surface: {distances_opt.mean():.4f}")
print(f"  Max dist to surface: {distances_opt.max():.4f}")

# Calculate movement per point
movement = np.linalg.norm(all_pts_opt - all_pts, axis=1)
print(f"\nPoint movement statistics:")
print(f"  Mean movement: {movement.mean():.4f}")
print(f"  Median movement: {np.median(movement):.4f}")
print(f"  Max movement: {movement.max():.4f}")
print(f"  Min movement: {movement.min():.4f}")

# Find points that didn't move much
low_movement_threshold = 0.01
low_movement = movement < low_movement_threshold
if np.any(low_movement):
    print(f"\nPoints with movement < {low_movement_threshold}: {np.sum(low_movement)}")

    # Check if these are near the surface
    low_movement_distances = distances[low_movement]
    print(f"  Their mean distance to surface: {low_movement_distances.mean():.4f}")
    print(f"  Their max distance to surface: {low_movement_distances.max():.4f}")

# Identify problematic regions
print("\n" + "=" * 70)
print("Identifying problematic regions:")
print("=" * 70)

# Points that are far from surface but didn't move much
far_from_surface = distances > 5.0
didnt_move_much = movement < 0.5
problematic = far_from_surface & didnt_move_much

if np.any(problematic):
    print(
        f"\nFound {np.sum(problematic)} points far from surface that didn't move much:"
    )
    prob_indices = np.where(problematic)[0]
    for idx in prob_indices[:10]:  # Show first 10
        print(
            f"  Point {idx}: dist={distances[idx]:.4f}, movement={movement[idx]:.4f}, inside={inside_mask[idx]}"
        )

# Save results
print("\n" + "=" * 70)
print("Saving optimized skeleton...")
optimized_skeleton.to_txt("data/mcf_skeletons/TS2_optimized.polylines.txt")
print("Saved to: data/mcf_skeletons/TS2_optimized.polylines.txt")

"""
Demo script for skeleton optimization.

This example demonstrates how to use the SkeletonOptimizer to refine
MCF-generated skeleton polylines by pushing points toward the medial axis
of the mesh volume.
"""

import numpy as np

from mcf2swc import (
    MeshManager,
    PolylinesSkeleton,
    SkeletonOptimizer,
    SkeletonOptimizerOptions,
    example_mesh,
)


def demo_cylinder_optimization():
    """Demonstrate skeleton optimization on a cylinder mesh."""
    print("=" * 70)
    print("Cylinder Skeleton Optimization Demo")
    print("=" * 70)

    mesh = example_mesh("cylinder", radius=1.0, height=10.0, sections=32)
    print(f"\nCreated cylinder mesh:")
    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Faces: {len(mesh.faces)}")
    print(f"  Volume: {mesh.volume:.2f}")
    print(f"  Watertight: {mesh.is_watertight}")

    skeleton_points = np.array(
        [
            [0.4, 0.3, -4.0],
            [0.3, 0.2, -2.0],
            [0.2, 0.1, 0.0],
            [0.3, 0.2, 2.0],
            [0.4, 0.3, 4.0],
        ]
    )
    skeleton = PolylinesSkeleton([skeleton_points])
    print(f"\nCreated skeleton with {skeleton.total_points()} points")
    print(f"  Skeleton bounds: {skeleton.bounds()}")

    opts = SkeletonOptimizerOptions(
        max_iterations=100,
        step_size=0.1,
        convergence_threshold=1e-4,
        preserve_endpoints=True,
        smoothing_weight=0.5,
        verbose=True,
    )

    optimizer = SkeletonOptimizer(skeleton, mesh, opts)

    print("\n" + "-" * 70)
    print("Checking for surface crossing...")
    print("-" * 70)
    has_crossing, num_outside, max_dist = optimizer.check_surface_crossing()
    print(f"  Surface crossing detected: {has_crossing}")
    print(f"  Points outside mesh: {num_outside}/{skeleton.total_points()}")
    print(f"  Max distance outside: {max_dist:.4f}")

    print("\n" + "-" * 70)
    print("Running optimization...")
    print("-" * 70)
    optimized_skeleton = optimizer.optimize()

    print("\n" + "-" * 70)
    print("Optimization complete!")
    print("-" * 70)

    optimizer_after = SkeletonOptimizer(optimized_skeleton, mesh, opts)
    has_crossing_after, num_outside_after, max_dist_after = (
        optimizer_after.check_surface_crossing()
    )
    print(
        f"  Points outside mesh (after): {num_outside_after}/{optimized_skeleton.total_points()}"
    )
    print(f"  Max distance outside (after): {max_dist_after:.4f}")

    original_points = skeleton.polylines[0]
    optimized_points = optimized_skeleton.polylines[0]
    movement = np.linalg.norm(optimized_points - original_points, axis=1)
    print(f"\n  Average point movement: {movement.mean():.4f}")
    print(f"  Max point movement: {movement.max():.4f}")
    print(f"  Min point movement: {movement.min():.4f}")

    stats = optimizer.get_optimization_stats()
    print(f"\nOptimization statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def demo_torus_optimization():
    """Demonstrate skeleton optimization on a torus mesh."""
    print("\n\n" + "=" * 70)
    print("Torus Skeleton Optimization Demo")
    print("=" * 70)

    mesh = example_mesh(
        "torus",
        major_radius=4.0,
        minor_radius=1.0,
        major_sections=48,
        minor_sections=24,
    )
    print(f"\nCreated torus mesh:")
    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Faces: {len(mesh.faces)}")
    print(f"  Volume: {mesh.volume:.2f}")
    print(f"  Watertight: {mesh.is_watertight}")

    theta = np.linspace(0, 2 * np.pi, 20)
    major_r = 3.5
    skeleton_points = np.column_stack(
        [major_r * np.cos(theta), major_r * np.sin(theta), np.zeros_like(theta)]
    )
    skeleton = PolylinesSkeleton([skeleton_points])
    print(f"\nCreated circular skeleton with {skeleton.total_points()} points")

    opts = SkeletonOptimizerOptions(
        max_iterations=50,
        step_size=0.05,
        convergence_threshold=1e-5,
        preserve_endpoints=False,
        smoothing_weight=0.7,
        verbose=True,
    )

    optimizer = SkeletonOptimizer(skeleton, mesh, opts)

    print("\n" + "-" * 70)
    print("Running optimization...")
    print("-" * 70)
    optimized_skeleton = optimizer.optimize()

    print("\n" + "-" * 70)
    print("Optimization complete!")
    print("-" * 70)

    original_points = skeleton.polylines[0]
    optimized_points = optimized_skeleton.polylines[0]
    movement = np.linalg.norm(optimized_points - original_points, axis=1)
    print(f"  Average point movement: {movement.mean():.4f}")
    print(f"  Max point movement: {movement.max():.4f}")


def demo_multiple_polylines():
    """Demonstrate optimization with multiple polylines."""
    print("\n\n" + "=" * 70)
    print("Multiple Polylines Optimization Demo")
    print("=" * 70)

    mesh = example_mesh("cylinder", radius=1.5, height=12.0, sections=32)
    print(f"\nCreated cylinder mesh:")
    print(f"  Radius: 1.5, Height: 12.0")

    polyline1 = np.array(
        [
            [0.0, 0.0, -5.0],
            [0.0, 0.0, -2.5],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 2.5],
            [0.0, 0.0, 5.0],
        ]
    )
    polyline2 = np.array([[0.5, 0.5, -3.0], [0.5, 0.5, 0.0], [0.5, 0.5, 3.0]])
    polyline3 = np.array([[-0.4, 0.3, -2.0], [-0.4, 0.3, 0.0], [-0.4, 0.3, 2.0]])

    skeleton = PolylinesSkeleton([polyline1, polyline2, polyline3])
    print(f"\nCreated skeleton with {len(skeleton.polylines)} polylines")
    print(f"  Total points: {skeleton.total_points()}")

    opts = SkeletonOptimizerOptions(
        max_iterations=50,
        step_size=0.08,
        preserve_endpoints=True,
        smoothing_weight=0.6,
        verbose=False,
    )

    optimizer = SkeletonOptimizer(skeleton, mesh, opts)
    optimized_skeleton = optimizer.optimize()

    print("\nOptimization complete!")
    for i, (orig, opt) in enumerate(
        zip(skeleton.polylines, optimized_skeleton.polylines)
    ):
        movement = np.linalg.norm(opt - orig, axis=1).mean()
        print(f"  Polyline {i}: {len(orig)} points, avg movement = {movement:.4f}")


if __name__ == "__main__":
    demo_cylinder_optimization()
    demo_torus_optimization()
    demo_multiple_polylines()
    print("\n" + "=" * 70)
    print("All demos complete!")
    print("=" * 70)

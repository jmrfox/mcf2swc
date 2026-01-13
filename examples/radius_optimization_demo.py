"""
Demonstration of radius optimization for skeleton graphs.

This example shows how to use the RadiusOptimizer to improve radius estimates
by treating them as parameters in an optimization problem rather than computing
them independently using geometric heuristics.
"""

import numpy as np
from mcf2swc import (
    SWCModel,
    example_mesh,
    OptimizerOptions,
    optimize_skeleton_radii,
    RadiusOptimizer,
)


def demo_basic_optimization():
    """Basic example: optimize radii for a simple cylinder skeleton."""
    print("=" * 70)
    print("Demo 1: Basic Radius Optimization")
    print("=" * 70)

    # Create a simple SWC model with poor initial radius estimates
    skeleton = SWCModel()
    skeleton.add_junction(node_id=0, x=0.0, y=0.0, z=0.0, r=0.5)
    skeleton.add_junction(node_id=1, x=0.0, y=0.0, z=5.0, r=0.6)
    skeleton.add_junction(node_id=2, x=0.0, y=0.0, z=10.0, r=0.5)
    skeleton.add_edge(0, 1)
    skeleton.add_edge(1, 2)

    # Target mesh: cylinder with radius 1.0
    mesh = example_mesh("cylinder", radius=1.0, height=10.0)

    print(f"\nMesh properties:")
    print(f"  Surface area: {mesh.area:.4f}")
    print(f"  Volume: {mesh.volume:.4f}")

    # Compute initial SWC model properties
    optimizer = RadiusOptimizer(skeleton, mesh)
    initial_radii = optimizer.get_initial_radii()
    initial_area = optimizer.compute_swc_surface_area(initial_radii)
    initial_volume = optimizer.compute_swc_volume(initial_radii)

    print(f"\nInitial skeleton (before optimization):")
    print(f"  Radii: {initial_radii}")
    print(
        f"  Surface area: {initial_area:.4f} (error: {abs(initial_area - mesh.area) / mesh.area * 100:.1f}%)"
    )
    print(
        f"  Volume: {initial_volume:.4f} (error: {abs(initial_volume - mesh.volume) / mesh.volume * 100:.1f}%)"
    )

    # Optimize using surface area loss
    options = OptimizerOptions(
        loss_function="surface_area", max_iterations=100, verbose=False
    )
    optimized_skeleton = optimize_skeleton_radii(skeleton, mesh, options=options)

    # Compute optimized properties
    optimized_radii = np.array(
        [optimized_skeleton.nodes[i]["radius"] for i in range(3)]
    )
    optimized_area = optimizer.compute_swc_surface_area(optimized_radii)
    optimized_volume = optimizer.compute_swc_volume(optimized_radii)

    print(f"\nOptimized skeleton (after optimization):")
    print(f"  Radii: {optimized_radii}")
    print(
        f"  Surface area: {optimized_area:.4f} (error: {abs(optimized_area - mesh.area) / mesh.area * 100:.1f}%)"
    )
    print(
        f"  Volume: {optimized_volume:.4f} (error: {abs(optimized_volume - mesh.volume) / mesh.volume * 100:.1f}%)"
    )

    print(f"\nImprovement:")
    print(
        f"  Surface area error reduced by {(abs(initial_area - mesh.area) - abs(optimized_area - mesh.area)) / abs(initial_area - mesh.area) * 100:.1f}%"
    )


def demo_loss_functions():
    """Compare different loss functions."""
    print("\n" + "=" * 70)
    print("Demo 2: Comparing Loss Functions")
    print("=" * 70)

    # Create SWC model
    skeleton = SWCModel()
    skeleton.add_junction(id=0, xyz=np.array([0.0, 0.0, 0.0]), radius=0.7)
    skeleton.add_junction(id=1, xyz=np.array([0.0, 0.0, 10.0]), radius=0.7)
    skeleton.add_edge(0, 1)

    mesh = example_mesh("cylinder", radius=1.0, height=10.0)

    loss_functions = ["surface_area", "volume", "combined"]

    for loss_fn in loss_functions:
        print(f"\n--- Loss function: {loss_fn} ---")

        options = OptimizerOptions(
            loss_function=loss_fn, max_iterations=100, verbose=False
        )

        optimized = optimize_skeleton_radii(skeleton.copy(), mesh, options=options)

        r0 = optimized.nodes[0]["radius"]
        r1 = optimized.nodes[1]["radius"]

        print(f"  Optimized radii: [{r0:.4f}, {r1:.4f}]")
        print(f"  Mean radius: {(r0 + r1) / 2:.4f} (target: 1.0)")


def demo_with_constraints():
    """Demonstrate optimization with radius constraints."""
    print("\n" + "=" * 70)
    print("Demo 3: Optimization with Constraints")
    print("=" * 70)

    skeleton = SWCModel()
    skeleton.add_junction(id=0, xyz=np.array([0.0, 0.0, 0.0]), radius=0.5)
    skeleton.add_junction(id=1, xyz=np.array([0.0, 0.0, 10.0]), radius=0.5)
    skeleton.add_edge(0, 1)

    mesh = example_mesh("cylinder", radius=1.0, height=10.0)

    # Optimize with tight constraints
    print("\nWith constraints: min_radius=0.2, max_radius=0.8")
    options = OptimizerOptions(
        loss_function="surface_area",
        min_radius=0.2,
        max_radius=0.8,
        max_iterations=100,
        verbose=False,
    )

    constrained = optimize_skeleton_radii(skeleton.copy(), mesh, options=options)

    r0 = constrained.nodes[0]["radius"]
    r1 = constrained.nodes[1]["radius"]

    print(f"  Optimized radii: [{r0:.4f}, {r1:.4f}]")
    print(f"  All radii within bounds: {0.2 <= r0 <= 0.8 and 0.2 <= r1 <= 0.8}")

    # Compare with unconstrained
    print("\nWithout constraints:")
    options_free = OptimizerOptions(
        loss_function="surface_area",
        min_radius=0.01,
        max_radius=None,
        max_iterations=100,
        verbose=False,
    )

    unconstrained = optimize_skeleton_radii(skeleton.copy(), mesh, options=options_free)

    r0_free = unconstrained.nodes[0]["radius"]
    r1_free = unconstrained.nodes[1]["radius"]

    print(f"  Optimized radii: [{r0_free:.4f}, {r1_free:.4f}]")


def demo_advanced_usage():
    """Advanced usage with the RadiusOptimizer class directly."""
    print("\n" + "=" * 70)
    print("Demo 4: Advanced Usage with RadiusOptimizer Class")
    print("=" * 70)

    skeleton = SWCModel()
    skeleton.add_junction(id=0, xyz=np.array([0.0, 0.0, 0.0]), radius=0.5)
    skeleton.add_junction(id=1, xyz=np.array([0.0, 0.0, 5.0]), radius=0.6)
    skeleton.add_junction(id=2, xyz=np.array([0.0, 0.0, 10.0]), radius=0.5)
    skeleton.add_edge(0, 1)
    skeleton.add_edge(1, 2)

    mesh = example_mesh("cylinder", radius=1.0, height=10.0)

    # Create optimizer with verbose output
    options = OptimizerOptions(
        loss_function="combined",
        loss_weights={"surface_area": 2.0, "volume": 1.0},
        max_iterations=50,
        verbose=True,
    )

    optimizer = RadiusOptimizer(skeleton, mesh, options=options)

    print("\nRunning optimization with combined loss (2x surface area + 1x volume)...")
    optimized = optimizer.optimize()

    print(f"\nOptimization history: {len(optimizer.history)} checkpoints recorded")

    if optimizer.history:
        print("\nLoss progression:")
        for entry in optimizer.history[:5]:  # Show first 5
            print(f"  Iteration {entry['iteration']}: loss = {entry['loss']:.6e}")


if __name__ == "__main__":
    demo_basic_optimization()
    demo_loss_functions()
    demo_with_constraints()
    demo_advanced_usage()

    print("\n" + "=" * 70)
    print("All demos complete!")
    print("=" * 70)

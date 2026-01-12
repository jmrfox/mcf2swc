"""
Demonstration of different constraint modes for radius optimization.

This example shows how regularization and scale-only constraints address
the under-constrained optimization problem.
"""

import numpy as np
from mcf2swc import (
    SkeletonGraph,
    Junction,
    example_mesh,
    OptimizerOptions,
    optimize_skeleton_radii,
)


def demo_constraint_modes():
    """Compare unconstrained, regularization, and scale-only optimization."""
    print("=" * 70)
    print("Constraint Modes Comparison")
    print("=" * 70)

    # Create skeleton with varying initial radii
    skeleton = SkeletonGraph()
    skeleton.add_junction(Junction(id=0, xyz=np.array([0.0, 0.0, 0.0]), radius=0.4))
    skeleton.add_junction(Junction(id=1, xyz=np.array([0.0, 0.0, 5.0]), radius=0.6))
    skeleton.add_junction(Junction(id=2, xyz=np.array([0.0, 0.0, 10.0]), radius=0.4))
    skeleton.add_edge(0, 1)
    skeleton.add_edge(1, 2)

    # Target mesh
    mesh = example_mesh("cylinder", radius=1.0, height=10.0)

    print(f"\nInitial radii: [0.4, 0.6, 0.4]")
    print(f"Initial relative proportions: [1.0, 1.5, 1.0]")
    print(f"\nTarget mesh: radius=1.0, surface_area={mesh.area:.4f}")

    # 1. Unconstrained optimization
    print("\n" + "-" * 70)
    print("1. UNCONSTRAINED MODE")
    print("-" * 70)

    options_unconstrained = OptimizerOptions(
        loss_function="surface_area",
        constraint_mode="unconstrained",
        max_iterations=100,
        verbose=False,
    )

    result_unconstrained = optimize_skeleton_radii(
        skeleton.copy(), mesh, options=options_unconstrained
    )

    radii_unc = np.array([result_unconstrained.nodes[i]["radius"] for i in range(3)])
    proportions_unc = radii_unc / radii_unc[0]

    print(f"Optimized radii: {radii_unc}")
    print(f"Relative proportions: {proportions_unc}")
    print(
        f"Note: Proportions may have changed significantly from initial [1.0, 1.5, 1.0]"
    )

    # 2. Regularization
    print("\n" + "-" * 70)
    print("2. REGULARIZATION MODE")
    print("-" * 70)

    for weight in [0.01, 0.1, 1.0]:
        options_reg = OptimizerOptions(
            loss_function="surface_area",
            constraint_mode="regularization",
            regularization_weight=weight,
            max_iterations=100,
            verbose=False,
        )

        result_reg = optimize_skeleton_radii(skeleton.copy(), mesh, options=options_reg)

        radii_reg = np.array([result_reg.nodes[i]["radius"] for i in range(3)])

        # Compute deviation from initial
        deviation = np.linalg.norm(radii_reg - np.array([0.4, 0.6, 0.4]))

        print(f"\nRegularization weight = {weight}:")
        print(f"  Optimized radii: {radii_reg}")
        print(f"  Deviation from initial: {deviation:.4f}")
        print(f"  (Higher weight → stays closer to initial values)")

    # 3. Scale-only
    print("\n" + "-" * 70)
    print("3. SCALE-ONLY MODE")
    print("-" * 70)

    options_scale = OptimizerOptions(
        loss_function="surface_area",
        constraint_mode="scale_only",
        max_iterations=100,
        verbose=False,
    )

    result_scale = optimize_skeleton_radii(skeleton.copy(), mesh, options=options_scale)

    radii_scale = np.array([result_scale.nodes[i]["radius"] for i in range(3)])
    proportions_scale = radii_scale / radii_scale[0]

    # Compute scale factor
    scale_factor = radii_scale[0] / 0.4

    print(f"Optimized radii: {radii_scale}")
    print(f"Relative proportions: {proportions_scale}")
    print(f"Scale factor: {scale_factor:.4f}")
    print(f"Verification: initial * scale = {np.array([0.4, 0.6, 0.4]) * scale_factor}")
    print(f"Note: Proportions exactly preserved at [1.0, 1.5, 1.0]")


def demo_why_constraints_matter():
    """Demonstrate the under-constrained problem."""
    print("\n" + "=" * 70)
    print("Why Constraints Matter: The Under-Constrained Problem")
    print("=" * 70)

    skeleton = SkeletonGraph()
    skeleton.add_junction(Junction(id=0, xyz=np.array([0.0, 0.0, 0.0]), radius=0.5))
    skeleton.add_junction(Junction(id=1, xyz=np.array([0.0, 0.0, 10.0]), radius=0.5))
    skeleton.add_edge(0, 1)

    mesh = example_mesh("cylinder", radius=1.0, height=10.0)

    print("\nSimple 2-node skeleton, target mesh with radius=1.0")
    print("\nTesting multiple random initializations with unconstrained mode:")

    from mcf2swc.radius_optimizer import RadiusOptimizer

    for trial in range(3):
        # Random initial radii
        np.random.seed(trial)
        r_init = np.random.uniform(0.3, 0.8, size=2)

        skel_trial = skeleton.copy()
        skel_trial.nodes[0]["radius"] = r_init[0]
        skel_trial.nodes[1]["radius"] = r_init[1]

        options = OptimizerOptions(
            constraint_mode="unconstrained", max_iterations=50, verbose=False
        )

        result = optimize_skeleton_radii(skel_trial, mesh, options=options)

        r_final = np.array([result.nodes[0]["radius"], result.nodes[1]["radius"]])

        optimizer = RadiusOptimizer(result, mesh, options=options)
        final_loss = optimizer.compute_data_loss(r_final)

        print(f"\nTrial {trial + 1}:")
        print(f"  Initial: {r_init}")
        print(f"  Final:   {r_final}")
        print(f"  Loss:    {final_loss:.6f}")

    print("\nObservation: Different initializations may converge to different")
    print("solutions with similar loss values (under-constrained problem).")
    print("\nSolution: Use regularization or scale-only constraints!")


if __name__ == "__main__":
    demo_constraint_modes()
    demo_why_constraints_matter()

    print("\n" + "=" * 70)
    print("Recommendation:")
    print("  - Use 'regularization' (default) for general cases")
    print("  - Use 'scale_only' when initial proportions are known to be good")
    print("  - Use 'unconstrained' only for experimentation")
    print("=" * 70)

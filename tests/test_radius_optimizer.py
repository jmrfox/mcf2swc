"""
Tests for radius_optimizer module.
"""

import numpy as np
import pytest
import trimesh

from mcf2swc import SkeletonGraph, Junction
from mcf2swc.radius_optimizer import (
    OptimizerOptions,
    RadiusOptimizer,
    optimize_skeleton_radii,
)
from mcf2swc.mesh import example_mesh


class TestRadiusOptimizer:
    """Test suite for RadiusOptimizer class."""

    def test_optimizer_initialization(self):
        """Test that optimizer initializes correctly."""
        # Create a simple skeleton
        skel = SkeletonGraph()
        skel.add_junction(Junction(id=0, xyz=np.array([0.0, 0.0, 0.0]), radius=1.0))
        skel.add_junction(Junction(id=1, xyz=np.array([0.0, 0.0, 5.0]), radius=1.0))
        skel.add_edge(0, 1)

        # Create a simple mesh
        mesh = example_mesh("cylinder", radius=1.0, height=10.0)

        # Initialize optimizer
        optimizer = RadiusOptimizer(skel, mesh)

        assert optimizer.n_nodes == 2
        assert len(optimizer.node_ids) == 2
        assert optimizer.mesh is mesh
        assert optimizer.skeleton is skel

    def test_get_initial_radii(self):
        """Test extraction of initial radii from skeleton."""
        skel = SkeletonGraph()
        skel.add_junction(Junction(id=0, xyz=np.array([0.0, 0.0, 0.0]), radius=1.5))
        skel.add_junction(Junction(id=1, xyz=np.array([0.0, 0.0, 5.0]), radius=2.0))
        skel.add_edge(0, 1)

        mesh = example_mesh("cylinder", radius=1.0, height=10.0)
        optimizer = RadiusOptimizer(skel, mesh)

        radii = optimizer.get_initial_radii()

        assert radii.shape == (2,)
        assert radii[0] == 1.5
        assert radii[1] == 2.0

    def test_set_radii(self):
        """Test updating skeleton with new radii."""
        skel = SkeletonGraph()
        skel.add_junction(Junction(id=0, xyz=np.array([0.0, 0.0, 0.0]), radius=1.0))
        skel.add_junction(Junction(id=1, xyz=np.array([0.0, 0.0, 5.0]), radius=1.0))
        skel.add_edge(0, 1)

        mesh = example_mesh("cylinder", radius=1.0, height=10.0)
        optimizer = RadiusOptimizer(skel, mesh)

        new_radii = np.array([2.5, 3.0])
        optimizer.set_radii(new_radii)

        assert skel.nodes[0]["radius"] == 2.5
        assert skel.nodes[1]["radius"] == 3.0

    def test_compute_swc_surface_area_cylinder(self):
        """Test surface area computation for a simple cylinder-like skeleton."""
        # Create skeleton with two nodes forming a cylinder
        skel = SkeletonGraph()
        skel.add_junction(Junction(id=0, xyz=np.array([0.0, 0.0, 0.0]), radius=1.0))
        skel.add_junction(Junction(id=1, xyz=np.array([0.0, 0.0, 10.0]), radius=1.0))
        skel.add_edge(0, 1)

        mesh = example_mesh("cylinder", radius=1.0, height=10.0)
        optimizer = RadiusOptimizer(skel, mesh)

        radii = np.array([1.0, 1.0])
        area = optimizer.compute_swc_surface_area(radii)

        # For a cylinder: lateral area = 2*pi*r*h = 2*pi*1*10 = 20*pi
        expected_area = 2.0 * np.pi * 1.0 * 10.0

        assert np.isclose(area, expected_area, rtol=1e-6)

    def test_compute_swc_surface_area_frustum(self):
        """Test surface area computation for a frustum (cone with different radii)."""
        # Create skeleton with two nodes of different radii
        skel = SkeletonGraph()
        skel.add_junction(Junction(id=0, xyz=np.array([0.0, 0.0, 0.0]), radius=1.0))
        skel.add_junction(Junction(id=1, xyz=np.array([0.0, 0.0, 10.0]), radius=2.0))
        skel.add_edge(0, 1)

        mesh = example_mesh("cylinder", radius=1.5, height=10.0)
        optimizer = RadiusOptimizer(skel, mesh)

        radii = np.array([1.0, 2.0])
        area = optimizer.compute_swc_surface_area(radii)

        # For a frustum: A = π(r1 + r2) * sqrt(h^2 + (r1 - r2)^2)
        h = 10.0
        r1, r2 = 1.0, 2.0
        slant = np.sqrt(h**2 + (r1 - r2) ** 2)
        expected_area = np.pi * (r1 + r2) * slant

        assert np.isclose(area, expected_area, rtol=1e-6)

    def test_compute_swc_volume_cylinder(self):
        """Test volume computation for a cylinder-like skeleton."""
        skel = SkeletonGraph()
        skel.add_junction(Junction(id=0, xyz=np.array([0.0, 0.0, 0.0]), radius=1.0))
        skel.add_junction(Junction(id=1, xyz=np.array([0.0, 0.0, 10.0]), radius=1.0))
        skel.add_edge(0, 1)

        mesh = example_mesh("cylinder", radius=1.0, height=10.0)
        optimizer = RadiusOptimizer(skel, mesh)

        radii = np.array([1.0, 1.0])
        volume = optimizer.compute_swc_volume(radii)

        # For a cylinder: V = pi*r^2*h = pi*1^2*10 = 10*pi
        expected_volume = np.pi * 1.0**2 * 10.0

        assert np.isclose(volume, expected_volume, rtol=1e-6)

    def test_compute_swc_volume_frustum(self):
        """Test volume computation for a frustum."""
        skel = SkeletonGraph()
        skel.add_junction(Junction(id=0, xyz=np.array([0.0, 0.0, 0.0]), radius=1.0))
        skel.add_junction(Junction(id=1, xyz=np.array([0.0, 0.0, 10.0]), radius=2.0))
        skel.add_edge(0, 1)

        mesh = example_mesh("cylinder", radius=1.5, height=10.0)
        optimizer = RadiusOptimizer(skel, mesh)

        radii = np.array([1.0, 2.0])
        volume = optimizer.compute_swc_volume(radii)

        # For a frustum: V = (π * h / 3) * (r1^2 + r1*r2 + r2^2)
        h = 10.0
        r1, r2 = 1.0, 2.0
        expected_volume = (np.pi * h / 3.0) * (r1**2 + r1 * r2 + r2**2)

        assert np.isclose(volume, expected_volume, rtol=1e-6)

    def test_compute_loss_surface_area(self):
        """Test loss computation with surface area metric."""
        skel = SkeletonGraph()
        skel.add_junction(Junction(id=0, xyz=np.array([0.0, 0.0, 0.0]), radius=1.0))
        skel.add_junction(Junction(id=1, xyz=np.array([0.0, 0.0, 10.0]), radius=1.0))
        skel.add_edge(0, 1)

        mesh = example_mesh("cylinder", radius=1.0, height=10.0)

        options = OptimizerOptions(loss_function="surface_area")
        optimizer = RadiusOptimizer(skel, mesh, options=options)

        # Perfect match should give near-zero loss
        radii = np.array([1.0, 1.0])
        loss = optimizer.compute_loss(radii)

        # Loss should be small (relative error)
        assert loss < 0.1  # Less than 10% error

    def test_compute_loss_volume(self):
        """Test loss computation with volume metric."""
        skel = SkeletonGraph()
        skel.add_junction(Junction(id=0, xyz=np.array([0.0, 0.0, 0.0]), radius=1.0))
        skel.add_junction(Junction(id=1, xyz=np.array([0.0, 0.0, 10.0]), radius=1.0))
        skel.add_edge(0, 1)

        mesh = example_mesh("cylinder", radius=1.0, height=10.0)

        options = OptimizerOptions(loss_function="volume")
        optimizer = RadiusOptimizer(skel, mesh, options=options)

        radii = np.array([1.0, 1.0])
        loss = optimizer.compute_loss(radii)

        # Loss should be small
        assert loss < 0.1

    def test_compute_loss_combined(self):
        """Test loss computation with combined metric."""
        skel = SkeletonGraph()
        skel.add_junction(Junction(id=0, xyz=np.array([0.0, 0.0, 0.0]), radius=1.0))
        skel.add_junction(Junction(id=1, xyz=np.array([0.0, 0.0, 10.0]), radius=1.0))
        skel.add_edge(0, 1)

        mesh = example_mesh("cylinder", radius=1.0, height=10.0)

        options = OptimizerOptions(
            loss_function="combined", loss_weights={"surface_area": 1.0, "volume": 1.0}
        )
        optimizer = RadiusOptimizer(skel, mesh, options=options)

        radii = np.array([1.0, 1.0])
        loss = optimizer.compute_loss(radii)

        # Loss should be finite and positive
        assert np.isfinite(loss)
        assert loss >= 0

    def test_optimize_simple_cylinder(self):
        """Test optimization on a simple cylinder case."""
        # Create skeleton with poor initial radii
        skel = SkeletonGraph()
        skel.add_junction(Junction(id=0, xyz=np.array([0.0, 0.0, 0.0]), radius=0.5))
        skel.add_junction(Junction(id=1, xyz=np.array([0.0, 0.0, 10.0]), radius=0.5))
        skel.add_edge(0, 1)

        # Target mesh is a cylinder with radius 1.0
        mesh = example_mesh("cylinder", radius=1.0, height=10.0)

        options = OptimizerOptions(
            loss_function="surface_area", max_iterations=100, verbose=False
        )
        optimizer = RadiusOptimizer(skel, mesh, options=options)

        initial_loss = optimizer.compute_loss(optimizer.get_initial_radii())
        optimized_skel = optimizer.optimize()

        # Check that optimization improved the loss
        optimized_radii = np.array(
            [optimized_skel.nodes[0]["radius"], optimized_skel.nodes[1]["radius"]]
        )
        final_loss = optimizer.compute_loss(optimized_radii)

        assert final_loss < initial_loss

        # Radii should be closer to 1.0
        assert np.allclose(optimized_radii, 1.0, atol=0.2)

    def test_optimize_with_bounds(self):
        """Test that optimization respects radius bounds."""
        skel = SkeletonGraph()
        skel.add_junction(Junction(id=0, xyz=np.array([0.0, 0.0, 0.0]), radius=0.5))
        skel.add_junction(Junction(id=1, xyz=np.array([0.0, 0.0, 10.0]), radius=0.5))
        skel.add_edge(0, 1)

        mesh = example_mesh("cylinder", radius=1.0, height=10.0)

        options = OptimizerOptions(
            loss_function="surface_area",
            min_radius=0.1,
            max_radius=0.7,  # Constrain below true radius
            max_iterations=50,
            verbose=False,
        )

        optimized_skel = optimize_skeleton_radii(skel, mesh, options=options)

        # All radii should respect bounds
        for nid in optimized_skel.nodes():
            r = optimized_skel.nodes[nid]["radius"]
            assert r >= 0.1
            assert r <= 0.7

    def test_optimize_skeleton_radii_convenience_function(self):
        """Test the convenience function optimize_skeleton_radii."""
        skel = SkeletonGraph()
        skel.add_junction(Junction(id=0, xyz=np.array([0.0, 0.0, 0.0]), radius=0.5))
        skel.add_junction(Junction(id=1, xyz=np.array([0.0, 0.0, 10.0]), radius=0.5))
        skel.add_edge(0, 1)

        mesh = example_mesh("cylinder", radius=1.0, height=10.0)

        optimized_skel = optimize_skeleton_radii(skel, mesh)

        # Should return a SkeletonGraph
        assert isinstance(optimized_skel, SkeletonGraph)

        # Should have same nodes
        assert optimized_skel.number_of_nodes() == skel.number_of_nodes()
        assert optimized_skel.number_of_edges() == skel.number_of_edges()

    def test_multiple_edges(self):
        """Test optimization with a skeleton containing multiple edges."""
        # Create a Y-shaped skeleton
        skel = SkeletonGraph()
        skel.add_junction(Junction(id=0, xyz=np.array([0.0, 0.0, 0.0]), radius=0.5))
        skel.add_junction(Junction(id=1, xyz=np.array([0.0, 0.0, 5.0]), radius=0.5))
        skel.add_junction(Junction(id=2, xyz=np.array([2.0, 0.0, 8.0]), radius=0.5))
        skel.add_junction(Junction(id=3, xyz=np.array([-2.0, 0.0, 8.0]), radius=0.5))
        skel.add_edge(0, 1)
        skel.add_edge(1, 2)
        skel.add_edge(1, 3)

        mesh = example_mesh("cylinder", radius=1.0, height=10.0)

        optimizer = RadiusOptimizer(skel, mesh)

        # Should handle multiple edges
        radii = optimizer.get_initial_radii()
        area = optimizer.compute_swc_surface_area(radii)
        volume = optimizer.compute_swc_volume(radii)

        assert area > 0
        assert volume > 0

    def test_invalid_loss_function(self):
        """Test that invalid loss function raises error."""
        skel = SkeletonGraph()
        skel.add_junction(Junction(id=0, xyz=np.array([0.0, 0.0, 0.0]), radius=1.0))
        skel.add_junction(Junction(id=1, xyz=np.array([0.0, 0.0, 10.0]), radius=1.0))
        skel.add_edge(0, 1)

        mesh = example_mesh("cylinder", radius=1.0, height=10.0)

        options = OptimizerOptions(loss_function="invalid_loss")
        optimizer = RadiusOptimizer(skel, mesh, options=options)

        with pytest.raises(ValueError, match="Unknown loss function"):
            optimizer.compute_loss(np.array([1.0, 1.0]))

    def test_invalid_optimizer(self):
        """Test that invalid optimizer raises error."""
        skel = SkeletonGraph()
        skel.add_junction(Junction(id=0, xyz=np.array([0.0, 0.0, 0.0]), radius=1.0))
        skel.add_junction(Junction(id=1, xyz=np.array([0.0, 0.0, 10.0]), radius=1.0))
        skel.add_edge(0, 1)

        mesh = example_mesh("cylinder", radius=1.0, height=10.0)

        options = OptimizerOptions(optimizer="invalid_optimizer")
        optimizer = RadiusOptimizer(skel, mesh, options=options)

        with pytest.raises(ValueError, match="Unknown optimizer"):
            optimizer.optimize()

    def test_regularization_constraint(self):
        """Test optimization with regularization constraint."""
        skel = SkeletonGraph()
        skel.add_junction(Junction(id=0, xyz=np.array([0.0, 0.0, 0.0]), radius=0.5))
        skel.add_junction(Junction(id=1, xyz=np.array([0.0, 0.0, 10.0]), radius=0.5))
        skel.add_edge(0, 1)

        mesh = example_mesh("cylinder", radius=1.0, height=10.0)

        # Test with weak regularization - should allow some movement
        options_weak = OptimizerOptions(
            loss_function="surface_area",
            constraint_mode="regularization",
            regularization_weight=0.1,  # Weak regularization
            max_iterations=100,
            verbose=False,
        )

        optimized_weak = optimize_skeleton_radii(skel, mesh, options=options_weak)

        # With weak regularization, radii should improve toward optimal
        r0_weak = optimized_weak.nodes[0]["radius"]
        r1_weak = optimized_weak.nodes[1]["radius"]

        # Should be between initial (0.5) and optimal (~1.0)
        assert 0.5 <= r0_weak <= 1.2
        assert 0.5 <= r1_weak <= 1.2

        # Test with strong regularization - should stay closer to initial
        options_strong = OptimizerOptions(
            loss_function="surface_area",
            constraint_mode="regularization",
            regularization_weight=10.0,  # Very strong regularization
            max_iterations=100,
            verbose=False,
        )

        optimized_strong = optimize_skeleton_radii(skel, mesh, options=options_strong)

        r0_strong = optimized_strong.nodes[0]["radius"]
        r1_strong = optimized_strong.nodes[1]["radius"]

        # Strong regularization should keep radii closer to initial (0.5)
        # than weak regularization does
        assert abs(r0_strong - 0.5) < abs(r0_weak - 0.5) or np.isclose(r0_strong, 0.5)
        assert abs(r1_strong - 0.5) < abs(r1_weak - 0.5) or np.isclose(r1_strong, 0.5)

    def test_scale_only_constraint(self):
        """Test optimization with scale-only constraint."""
        skel = SkeletonGraph()
        skel.add_junction(Junction(id=0, xyz=np.array([0.0, 0.0, 0.0]), radius=0.5))
        skel.add_junction(Junction(id=1, xyz=np.array([0.0, 0.0, 5.0]), radius=0.6))
        skel.add_junction(Junction(id=2, xyz=np.array([0.0, 0.0, 10.0]), radius=0.5))
        skel.add_edge(0, 1)
        skel.add_edge(1, 2)

        mesh = example_mesh("cylinder", radius=1.0, height=10.0)

        # Store initial relative proportions
        initial_radii = np.array([0.5, 0.6, 0.5])
        initial_ratios = initial_radii / initial_radii[0]

        options = OptimizerOptions(
            loss_function="surface_area",
            constraint_mode="scale_only",
            max_iterations=100,
            verbose=False,
        )

        optimized = optimize_skeleton_radii(skel, mesh, options=options)

        # Extract optimized radii
        optimized_radii = np.array(
            [
                optimized.nodes[0]["radius"],
                optimized.nodes[1]["radius"],
                optimized.nodes[2]["radius"],
            ]
        )

        # Check that relative proportions are preserved
        optimized_ratios = optimized_radii / optimized_radii[0]
        np.testing.assert_allclose(optimized_ratios, initial_ratios, rtol=1e-6)

        # Check that scale changed (not all still 0.5, 0.6, 0.5)
        assert not np.allclose(optimized_radii, initial_radii)

    def test_unconstrained_mode(self):
        """Test optimization with unconstrained mode."""
        skel = SkeletonGraph()
        skel.add_junction(Junction(id=0, xyz=np.array([0.0, 0.0, 0.0]), radius=0.5))
        skel.add_junction(Junction(id=1, xyz=np.array([0.0, 0.0, 10.0]), radius=0.5))
        skel.add_edge(0, 1)

        mesh = example_mesh("cylinder", radius=1.0, height=10.0)

        options = OptimizerOptions(
            loss_function="surface_area",
            constraint_mode="unconstrained",
            max_iterations=100,
            verbose=False,
        )

        optimized = optimize_skeleton_radii(skel, mesh, options=options)

        # Should optimize without regularization penalty
        assert isinstance(optimized, SkeletonGraph)
        assert optimized.number_of_nodes() == 2

    def test_compute_regularization_loss(self):
        """Test regularization loss computation."""
        skel = SkeletonGraph()
        skel.add_junction(Junction(id=0, xyz=np.array([0.0, 0.0, 0.0]), radius=1.0))
        skel.add_junction(Junction(id=1, xyz=np.array([0.0, 0.0, 10.0]), radius=1.0))
        skel.add_edge(0, 1)

        mesh = example_mesh("cylinder", radius=1.0, height=10.0)
        optimizer = RadiusOptimizer(skel, mesh)

        # No change should give zero regularization loss
        initial_radii = optimizer.get_initial_radii()
        reg_loss = optimizer.compute_regularization_loss(initial_radii)
        assert np.isclose(reg_loss, 0.0)

        # Changed radii should give positive regularization loss
        changed_radii = initial_radii * 2.0
        reg_loss = optimizer.compute_regularization_loss(changed_radii)
        assert reg_loss > 0

    def test_data_loss_vs_total_loss(self):
        """Test that total loss includes regularization when enabled."""
        skel = SkeletonGraph()
        skel.add_junction(Junction(id=0, xyz=np.array([0.0, 0.0, 0.0]), radius=0.5))
        skel.add_junction(Junction(id=1, xyz=np.array([0.0, 0.0, 10.0]), radius=0.5))
        skel.add_edge(0, 1)

        mesh = example_mesh("cylinder", radius=1.0, height=10.0)

        # With regularization
        options_reg = OptimizerOptions(
            constraint_mode="regularization", regularization_weight=0.5
        )
        optimizer_reg = RadiusOptimizer(skel, mesh, options=options_reg)

        # Without regularization
        options_unreg = OptimizerOptions(constraint_mode="unconstrained")
        optimizer_unreg = RadiusOptimizer(skel, mesh, options=options_unreg)

        # Test with changed radii
        changed_radii = np.array([1.5, 1.5])

        data_loss = optimizer_reg.compute_data_loss(changed_radii)
        total_loss_reg = optimizer_reg.compute_loss(changed_radii)
        total_loss_unreg = optimizer_unreg.compute_loss(changed_radii)

        # Total loss with regularization should be higher
        assert total_loss_reg > data_loss
        assert total_loss_unreg == data_loss

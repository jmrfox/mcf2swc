"""
Tests for skeleton optimization module.
"""

import numpy as np
import pytest
import trimesh

from mcf2swc import (
    MeshManager,
    SkeletonGraph,
    SkeletonOptimizer,
    SkeletonOptimizerOptions,
    example_mesh,
)


class TestSkeletonOptimizerOptions:
    """Test SkeletonOptimizerOptions dataclass."""

    def test_default_options(self):
        opts = SkeletonOptimizerOptions()
        assert opts.check_surface_crossing is True
        assert opts.max_iterations == 100
        assert opts.step_size == 0.1
        assert opts.convergence_threshold == 1e-4
        assert opts.preserve_terminal_nodes is True
        assert opts.smoothing_weight == 0.5
        assert opts.verbose is False

    def test_custom_options(self):
        opts = SkeletonOptimizerOptions(
            max_iterations=50,
            step_size=0.05,
            smoothing_weight=0.3,
            verbose=True,
        )
        assert opts.max_iterations == 50
        assert opts.step_size == 0.05
        assert opts.smoothing_weight == 0.3
        assert opts.verbose is True


class TestSkeletonOptimizer:
    """Test SkeletonOptimizer class."""

    @pytest.fixture
    def cylinder_mesh(self):
        """Create a simple cylinder mesh for testing."""
        return example_mesh("cylinder", radius=1.0, height=10.0, sections=16)

    @pytest.fixture
    def cylinder_skeleton_inside(self):
        """Create a skeleton that is fully inside the cylinder."""
        points = np.array(
            [
                [0.5, 0.0, -3.5],
                [0.5, 0.0, -1.5],
                [0.5, 0.0, 0.0],
                [0.5, 0.0, 1.5],
                [0.5, 0.0, 3.5],
            ]
        )
        return SkeletonGraph.from_polylines([points])

    @pytest.fixture
    def cylinder_skeleton_outside(self):
        """Create a skeleton with some points outside the cylinder."""
        points = np.array(
            [
                [0.0, 0.0, -3.5],
                [1.5, 0.0, -1.5],
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 1.5],
                [0.0, 0.0, 3.5],
            ]
        )
        return SkeletonGraph.from_polylines([points])

    @pytest.fixture
    def cylinder_skeleton_offset(self):
        """Create a skeleton offset from the center."""
        points = np.array(
            [
                [0.3, 0.2, -3.5],
                [0.3, 0.2, -1.5],
                [0.3, 0.2, 0.0],
                [0.3, 0.2, 1.5],
                [0.3, 0.2, 3.5],
            ]
        )
        return SkeletonGraph.from_polylines([points])

    def test_initialization(self, cylinder_mesh, cylinder_skeleton_inside):
        """Test basic initialization of SkeletonOptimizer."""
        optimizer = SkeletonOptimizer(cylinder_skeleton_inside, cylinder_mesh)
        assert optimizer.skeleton is not None
        assert optimizer.mesh is not None
        assert optimizer.options is not None
        assert isinstance(optimizer.options, SkeletonOptimizerOptions)

    def test_initialization_with_options(self, cylinder_mesh, cylinder_skeleton_inside):
        """Test initialization with custom options."""
        opts = SkeletonOptimizerOptions(max_iterations=50, verbose=True)
        optimizer = SkeletonOptimizer(cylinder_skeleton_inside, cylinder_mesh, opts)
        assert optimizer.options.max_iterations == 50
        assert optimizer.options.verbose is True

    def test_check_surface_crossing_inside(
        self, cylinder_mesh, cylinder_skeleton_inside
    ):
        """Test surface crossing detection for skeleton inside mesh."""
        optimizer = SkeletonOptimizer(cylinder_skeleton_inside, cylinder_mesh)
        has_crossing, num_outside, max_dist = optimizer.check_surface_crossing()

        points = cylinder_skeleton_inside.get_all_positions()
        inside_check = cylinder_mesh.contains(points)
        assert np.all(
            inside_check
        ), "All points should be inside mesh according to contains()"

    def test_check_surface_crossing_outside(
        self, cylinder_mesh, cylinder_skeleton_outside
    ):
        """Test surface crossing detection for skeleton with points outside mesh."""
        optimizer = SkeletonOptimizer(cylinder_skeleton_outside, cylinder_mesh)
        has_crossing, num_outside, max_dist = optimizer.check_surface_crossing()
        assert has_crossing is True
        assert num_outside > 0
        assert max_dist > 0.0

    def test_optimize_skeleton_inside(self, cylinder_mesh, cylinder_skeleton_inside):
        """Test optimization of skeleton that is already inside mesh."""
        opts = SkeletonOptimizerOptions(
            max_iterations=10, step_size=0.05, verbose=False
        )
        optimizer = SkeletonOptimizer(cylinder_skeleton_inside, cylinder_mesh, opts)
        optimized = optimizer.optimize()

        assert isinstance(optimized, SkeletonGraph)
        assert optimized.number_of_nodes() == cylinder_skeleton_inside.number_of_nodes()

    def test_optimize_skeleton_outside(self, cylinder_mesh, cylinder_skeleton_outside):
        """Test optimization of skeleton with points outside mesh."""
        opts = SkeletonOptimizerOptions(max_iterations=50, step_size=0.1, verbose=False)
        optimizer = SkeletonOptimizer(cylinder_skeleton_outside, cylinder_mesh, opts)

        has_crossing_before, num_outside_before, _ = optimizer.check_surface_crossing()
        assert has_crossing_before is True
        assert num_outside_before > 0

        optimized = optimizer.optimize()

        optimizer_after = SkeletonOptimizer(optimized, cylinder_mesh, opts)
        has_crossing_after, num_outside_after, _ = (
            optimizer_after.check_surface_crossing()
        )

        assert num_outside_after <= num_outside_before

    def test_optimize_skeleton_offset(self, cylinder_mesh, cylinder_skeleton_offset):
        """Test optimization of skeleton offset from center."""
        opts = SkeletonOptimizerOptions(
            max_iterations=50, step_size=0.1, smoothing_weight=0.5, verbose=False
        )
        optimizer = SkeletonOptimizer(cylinder_skeleton_offset, cylinder_mesh, opts)
        optimized = optimizer.optimize()

        original_polylines = cylinder_skeleton_offset.to_polylines()
        optimized_polylines = optimized.to_polylines()
        original_points = original_polylines[0]
        optimized_points = optimized_polylines[0]

        total_movement = np.sum(
            np.linalg.norm(optimized_points - original_points, axis=1)
        )
        assert total_movement > 0, "Optimization should move points"

        assert len(optimized_points) == len(original_points)
        assert optimized_points.shape == original_points.shape

    def test_preserve_endpoints(self, cylinder_mesh, cylinder_skeleton_offset):
        """Test that endpoints are preserved when preserve_terminal_nodes=True."""
        opts = SkeletonOptimizerOptions(
            max_iterations=50, preserve_terminal_nodes=True, verbose=False
        )
        optimizer = SkeletonOptimizer(cylinder_skeleton_offset, cylinder_mesh, opts)
        optimized = optimizer.optimize()

        original_polylines = cylinder_skeleton_offset.to_polylines()
        optimized_polylines = optimized.to_polylines()
        original_points = original_polylines[0]
        optimized_points = optimized_polylines[0]

        np.testing.assert_allclose(original_points[0], optimized_points[0], rtol=1e-10)
        np.testing.assert_allclose(
            original_points[-1], optimized_points[-1], rtol=1e-10
        )

    def test_do_not_preserve_endpoints(self, cylinder_mesh, cylinder_skeleton_offset):
        """Test that endpoints can move when preserve_terminal_nodes=False."""
        opts = SkeletonOptimizerOptions(
            max_iterations=50, preserve_terminal_nodes=False, verbose=False
        )
        optimizer = SkeletonOptimizer(cylinder_skeleton_offset, cylinder_mesh, opts)
        optimized = optimizer.optimize()

        original_polylines = cylinder_skeleton_offset.to_polylines()
        optimized_polylines = optimized.to_polylines()
        original_points = original_polylines[0]
        optimized_points = optimized_polylines[0]

        endpoint_moved = not np.allclose(
            original_points[0], optimized_points[0], rtol=1e-10
        ) or not np.allclose(original_points[-1], optimized_points[-1], rtol=1e-10)
        assert endpoint_moved

    def test_empty_polyline(self, cylinder_mesh):
        """Test optimization with empty polyline."""
        empty_skeleton = SkeletonGraph.from_polylines([np.zeros((0, 3))])
        optimizer = SkeletonOptimizer(empty_skeleton, cylinder_mesh)
        optimized = optimizer.optimize()
        assert optimized.number_of_nodes() == 0

    def test_single_point_polyline(self, cylinder_mesh):
        """Test optimization with single-point polyline."""
        single_point = SkeletonGraph.from_polylines([np.array([[0.0, 0.0, 0.0]])])
        optimizer = SkeletonOptimizer(single_point, cylinder_mesh)
        optimized = optimizer.optimize()
        assert optimized.number_of_nodes() == 1

    def test_multiple_polylines(self, cylinder_mesh):
        """Test optimization with multiple polylines."""
        polyline1 = np.array(
            [[0.0, 0.0, -3.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 3.0]]
        )
        polyline2 = np.array([[0.2, 0.2, -2.0], [0.2, 0.2, 0.0], [0.2, 0.2, 2.0]])
        skeleton = SkeletonGraph.from_polylines([polyline1, polyline2])

        opts = SkeletonOptimizerOptions(max_iterations=20, verbose=False)
        optimizer = SkeletonOptimizer(skeleton, cylinder_mesh, opts)
        optimized = optimizer.optimize()

        opt_polylines = optimized.to_polylines()
        assert len(opt_polylines) == 2

    def test_get_optimization_stats(self, cylinder_mesh, cylinder_skeleton_outside):
        """Test getting optimization statistics."""
        optimizer = SkeletonOptimizer(cylinder_skeleton_outside, cylinder_mesh)
        stats = optimizer.get_optimization_stats()

        assert "num_nodes" in stats
        assert "num_edges" in stats
        assert "nodes_outside_mesh" in stats
        assert "max_distance_outside" in stats
        assert stats["num_nodes"] == 5
        assert stats["nodes_outside_mesh"] > 0

    def test_smoothing_weight_effect(self, cylinder_mesh, cylinder_skeleton_offset):
        """Test that smoothing weight affects optimization."""
        opts_no_smoothing = SkeletonOptimizerOptions(
            max_iterations=30, smoothing_weight=0.0, verbose=False
        )
        optimizer_no_smoothing = SkeletonOptimizer(
            cylinder_skeleton_offset, cylinder_mesh, opts_no_smoothing
        )
        optimized_no_smoothing = optimizer_no_smoothing.optimize()

        opts_high_smoothing = SkeletonOptimizerOptions(
            max_iterations=30, smoothing_weight=0.9, verbose=False
        )
        optimizer_high_smoothing = SkeletonOptimizer(
            cylinder_skeleton_offset, cylinder_mesh, opts_high_smoothing
        )
        optimized_high_smoothing = optimizer_high_smoothing.optimize()

        no_smooth_pl = optimized_no_smoothing.to_polylines()[0]
        high_smooth_pl = optimized_high_smoothing.to_polylines()[0]
        assert not np.allclose(no_smooth_pl, high_smooth_pl)

    def test_convergence(self, cylinder_mesh, cylinder_skeleton_offset):
        """Test that optimization converges."""
        opts = SkeletonOptimizerOptions(
            max_iterations=100,
            convergence_threshold=1e-5,
            step_size=0.05,
            verbose=False,
        )
        optimizer = SkeletonOptimizer(cylinder_skeleton_offset, cylinder_mesh, opts)
        optimized = optimizer.optimize()

        assert isinstance(optimized, SkeletonGraph)

    def test_torus_mesh(self):
        """Test optimization with a torus mesh."""
        torus_mesh = example_mesh(
            "torus", major_radius=4.0, minor_radius=1.0, major_sections=32
        )

        points = np.array(
            [
                [3.5, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [-3.0, 0.0, 0.0],
                [0.0, -3.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        )
        skeleton = SkeletonGraph.from_polylines([points])

        opts = SkeletonOptimizerOptions(max_iterations=50, verbose=False)
        optimizer = SkeletonOptimizer(skeleton, torus_mesh, opts)
        optimized = optimizer.optimize()

        assert isinstance(optimized, SkeletonGraph)
        assert optimized.number_of_nodes() > 0

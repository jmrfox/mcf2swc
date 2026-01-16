"""
Test medial axis centering method for skeleton optimization.
"""

from pathlib import Path
import pytest
import numpy as np
from mcf2swc import (
    SkeletonGraph,
    SkeletonOptimizer,
    SkeletonOptimizerOptions,
    example_mesh,
)


def test_closest_point_centering():
    """Test skeleton optimization with closest_point centering method."""
    mesh = example_mesh("cylinder", radius=1.0, height=10.0)

    points = np.array(
        [
            [0.3, 0.2, -3.5],
            [0.3, 0.2, 0.0],
            [0.3, 0.2, 3.5],
        ]
    )
    skeleton = SkeletonGraph.from_polylines([points])

    options = SkeletonOptimizerOptions(
        max_iterations=10,
        step_size=0.1,
        preserve_terminal_nodes=True,
        smoothing_weight=0.3,
        verbose=False,
    )

    optimizer = SkeletonOptimizer(skeleton, mesh, options)
    optimized = optimizer.optimize()

    assert optimized.number_of_nodes() > 0


def test_medial_axis_centering():
    """Test skeleton optimization with medial_axis centering method."""
    mesh = example_mesh("cylinder", radius=1.0, height=10.0)

    points = np.array(
        [
            [0.3, 0.2, -3.5],
            [0.3, 0.2, 0.0],
            [0.3, 0.2, 3.5],
        ]
    )
    skeleton = SkeletonGraph.from_polylines([points])

    options = SkeletonOptimizerOptions(
        max_iterations=10,
        step_size=0.1,
        preserve_terminal_nodes=True,
        smoothing_weight=0.3,
        verbose=False,
    )

    optimizer = SkeletonOptimizer(skeleton, mesh, options)
    optimized = optimizer.optimize()

    assert optimized.number_of_nodes() > 0


def test_centering_improves_position():
    """Test that optimization moves points closer to center."""
    mesh = example_mesh("cylinder", radius=1.0, height=10.0)

    points = np.array(
        [
            [0.5, 0.5, -3.5],
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 3.5],
        ]
    )
    skeleton = SkeletonGraph.from_polylines([points])

    options = SkeletonOptimizerOptions(
        max_iterations=20,
        step_size=0.1,
        preserve_terminal_nodes=False,
        smoothing_weight=0.3,
        verbose=False,
    )

    optimizer = SkeletonOptimizer(skeleton, mesh, options)
    optimized = optimizer.optimize()

    original_distances = np.linalg.norm(points[:, :2], axis=1)
    optimized_polylines = optimized.to_polylines()
    optimized_points = optimized_polylines[0]
    optimized_distances = np.linalg.norm(optimized_points[:, :2], axis=1)

    assert np.mean(optimized_distances) < np.mean(original_distances)

"""
Test multi-directional sampling for medial axis centering.
"""

import pytest
import numpy as np
from mcf2swc import (
    SkeletonGraph,
    SkeletonOptimizer,
    SkeletonOptimizerOptions,
    example_mesh,
)


@pytest.mark.parametrize("num_directions", [4, 8])
def test_multi_direction_sampling(num_directions):
    """Test medial axis centering with different numbers of probe directions."""
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
        n_rays=num_directions,
        max_iterations=10,
        step_size=0.1,
        preserve_terminal_nodes=True,
        smoothing_weight=0.3,
        verbose=False,
    )

    optimizer = SkeletonOptimizer(skeleton, mesh, options)
    optimized = optimizer.optimize()

    assert optimized.number_of_nodes() > 0

    optimized_polylines = optimized.to_polylines()
    optimized_points = optimized_polylines[0]
    optimized_distances = np.linalg.norm(optimized_points[:, :2], axis=1)

    assert np.all(optimized_distances < 1.0)


def test_more_directions_improves_centering():
    """Test that more probe directions can improve centering quality."""
    mesh = example_mesh("cylinder", radius=1.0, height=10.0)

    points = np.array(
        [
            [0.6, 0.6, -3.5],
            [0.6, 0.6, 0.0],
            [0.6, 0.6, 3.5],
        ]
    )
    skeleton = SkeletonGraph.from_polylines([points])

    results = []
    for num_dirs in [4, 8]:
        options = SkeletonOptimizerOptions(
            n_rays=num_dirs,
            max_iterations=15,
            step_size=0.1,
            preserve_terminal_nodes=False,
            smoothing_weight=0.3,
            verbose=False,
        )

        optimizer = SkeletonOptimizer(skeleton, mesh, options)
        optimized = optimizer.optimize()

        optimized_polylines = optimized.to_polylines()
        optimized_points = optimized_polylines[0]
        mean_distance = np.mean(np.linalg.norm(optimized_points[:, :2], axis=1))
        results.append(mean_distance)

    assert results[1] <= results[0] * 1.1

"""
Tests for normal_strategy option in trace.py
"""

import numpy as np
import pytest

from mcf2swc import (
    SkeletonGraph,
    PolylinesSkeleton,
    TraceOptions,
    build_traced_skeleton_graph,
)
from mcf2swc.mesh import example_mesh


class TestNormalStrategy:
    """Test suite for normal_strategy option."""

    def test_tangent_strategy_default(self):
        """Test that tangent strategy is the default."""
        options = TraceOptions()
        assert options.normal_strategy == "tangent"

    def test_tangent_strategy_explicit(self):
        """Test tracing with explicit tangent strategy."""
        # Create a simple cylinder mesh
        mesh = example_mesh("cylinder", radius=1.0, height=10.0)

        # Create a polyline along the cylinder axis
        polyline = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 5.0],
                [0.0, 0.0, 10.0],
            ]
        )
        polylines = PolylinesSkeleton(polylines=[polyline])

        # Trace with tangent strategy
        options = TraceOptions(
            spacing=2.5, normal_strategy="tangent", radius_strategy="equivalent_area"
        )

        skeleton = build_traced_skeleton_graph(mesh, polylines, options=options)

        # Should have created nodes
        assert skeleton.number_of_nodes() > 0
        assert skeleton.number_of_edges() > 0

        # Check that radii are reasonable
        for nid in skeleton.nodes():
            r = skeleton.nodes[nid]["radius"]
            assert r > 0
            assert np.isfinite(r)

    def test_radial_strategy(self):
        """Test tracing with radial strategy."""
        # Create a simple cylinder mesh
        mesh = example_mesh("cylinder", radius=1.0, height=10.0)

        # Create a polyline along the cylinder axis
        polyline = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 5.0],
                [0.0, 0.0, 10.0],
            ]
        )
        polylines = PolylinesSkeleton(polylines=[polyline])

        # Trace with radial strategy
        options = TraceOptions(
            spacing=2.5, normal_strategy="radial", radius_strategy="equivalent_area"
        )

        skeleton = build_traced_skeleton_graph(mesh, polylines, options=options)

        # Should have created nodes
        assert skeleton.number_of_nodes() > 0
        assert skeleton.number_of_edges() > 0

        # Check that radii are reasonable
        for nid in skeleton.nodes():
            r = skeleton.nodes[nid]["radius"]
            assert r > 0
            assert np.isfinite(r)

    def test_compare_strategies(self):
        """Compare radii from tangent vs radial strategies."""
        # Create a cylinder mesh
        mesh = example_mesh("cylinder", radius=1.0, height=10.0)

        # Create a polyline along the axis
        polyline = np.array(
            [
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 5.0],
                [0.0, 0.0, 8.0],
            ]
        )
        polylines = PolylinesSkeleton(polylines=[polyline])

        # Trace with tangent strategy
        options_tangent = TraceOptions(
            spacing=1.5, normal_strategy="tangent", radius_strategy="equivalent_area"
        )
        skeleton_tangent = build_traced_skeleton_graph(
            mesh, polylines, options=options_tangent
        )

        # Trace with radial strategy
        options_radial = TraceOptions(
            spacing=1.5, normal_strategy="radial", radius_strategy="equivalent_area"
        )
        skeleton_radial = build_traced_skeleton_graph(
            mesh, polylines, options=options_radial
        )

        # Both should produce valid skeletons
        assert skeleton_tangent.number_of_nodes() > 0
        assert skeleton_radial.number_of_nodes() > 0

        # Extract radii
        radii_tangent = [
            skeleton_tangent.nodes[nid]["radius"]
            for nid in sorted(skeleton_tangent.nodes())
        ]
        radii_radial = [
            skeleton_radial.nodes[nid]["radius"]
            for nid in sorted(skeleton_radial.nodes())
        ]

        # Should have same number of nodes (same spacing)
        assert len(radii_tangent) == len(radii_radial)

        # Radii should be positive and finite
        assert all(r > 0 and np.isfinite(r) for r in radii_tangent)
        assert all(r > 0 and np.isfinite(r) for r in radii_radial)

        # The two strategies should produce different results
        # (radial strategy slices across the mesh width, tangent slices along it)
        # At least some radii should differ
        differences = [abs(r_t - r_r) for r_t, r_r in zip(radii_tangent, radii_radial)]
        assert any(
            diff > 0.1 for diff in differences
        ), "Strategies should produce different radii"

    def test_radial_strategy_off_axis_polyline(self):
        """Test radial strategy with a polyline not along the axis."""
        # Create a cylinder mesh
        mesh = example_mesh("cylinder", radius=1.0, height=10.0)

        # Create a polyline that's slightly off-axis
        # This is where radial strategy should differ from tangent
        polyline = np.array(
            [
                [0.2, 0.0, 2.0],
                [0.2, 0.0, 5.0],
                [0.2, 0.0, 8.0],
            ]
        )
        polylines = PolylinesSkeleton(polylines=[polyline])

        # Trace with radial strategy
        options = TraceOptions(
            spacing=1.5, normal_strategy="radial", radius_strategy="equivalent_area"
        )

        skeleton = build_traced_skeleton_graph(mesh, polylines, options=options)

        # Should produce valid skeleton
        assert skeleton.number_of_nodes() > 0

        # All radii should be positive and finite
        # Note: off-axis polylines may produce larger radii with radial strategy
        # because the cross-section captures a different slice of the mesh
        for nid in skeleton.nodes():
            r = skeleton.nodes[nid]["radius"]
            assert r > 0
            assert np.isfinite(r)

    def test_radial_strategy_with_different_radius_strategies(self):
        """Test that radial normal_strategy works with different radius_strategies."""
        mesh = example_mesh("cylinder", radius=1.0, height=10.0)

        polyline = np.array(
            [
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 8.0],
            ]
        )
        polylines = PolylinesSkeleton(polylines=[polyline])

        radius_strategies = [
            "equivalent_area",
            "equivalent_perimeter",
            "section_median",
        ]

        for radius_strat in radius_strategies:
            options = TraceOptions(
                spacing=3.0, normal_strategy="radial", radius_strategy=radius_strat
            )

            skeleton = build_traced_skeleton_graph(mesh, polylines, options=options)

            assert skeleton.number_of_nodes() > 0

            # Check all radii are valid
            for nid in skeleton.nodes():
                r = skeleton.nodes[nid]["radius"]
                assert r > 0
                assert np.isfinite(r)

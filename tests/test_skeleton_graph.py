"""
Test SkeletonGraph functionality.
"""

from pathlib import Path
import pytest
import numpy as np
from mcf2swc import SkeletonGraph


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "mcf_skeletons"


@pytest.fixture
def ts2_skeleton():
    """Load TS2 skeleton for testing."""
    skeleton_path = DATA / "TS2_qst0.6_mcst5.polylines.txt"
    if not skeleton_path.exists():
        pytest.skip(f"TS2 skeleton not found at {skeleton_path}")
    return SkeletonGraph.from_txt(str(skeleton_path), tolerance=1e-6)


def test_skeleton_graph_loading(ts2_skeleton):
    """Test that SkeletonGraph loads correctly from file."""
    assert ts2_skeleton is not None
    assert ts2_skeleton.number_of_nodes() > 0
    assert ts2_skeleton.number_of_edges() > 0
    print(f"\nLoaded: {ts2_skeleton}")


def test_skeleton_graph_structure(ts2_skeleton):
    """Test basic graph structure properties."""
    # Should have nodes and edges
    assert ts2_skeleton.number_of_nodes() > 0
    assert ts2_skeleton.number_of_edges() > 0

    # Number of edges should be reasonable relative to nodes
    # For a skeleton, typically edges ≈ nodes - num_components
    assert ts2_skeleton.number_of_edges() <= ts2_skeleton.number_of_nodes()


def test_node_classification(ts2_skeleton):
    """Test node classification (terminal, continuation, branch)."""
    terminal_nodes = ts2_skeleton.get_terminal_nodes()
    branch_nodes = ts2_skeleton.get_branch_nodes()

    # Should have at least some terminal nodes (endpoints)
    assert len(terminal_nodes) > 0

    # Terminal and branch nodes should be disjoint
    assert len(set(terminal_nodes) & set(branch_nodes)) == 0

    # All nodes should have degree >= 1
    for node in ts2_skeleton.nodes():
        degree = ts2_skeleton.degree(node)
        assert degree >= 1

        # Check classification is correct
        if degree == 1:
            assert node in terminal_nodes
        elif degree >= 3:
            assert node in branch_nodes

    print(f"\nTerminal nodes: {len(terminal_nodes)}")
    print(f"Branch nodes: {len(branch_nodes)}")


def test_node_positions(ts2_skeleton):
    """Test that all nodes have valid 3D positions."""
    for node in ts2_skeleton.nodes():
        pos = ts2_skeleton.get_node_position(node)

        # Should be a 3D coordinate
        assert pos.shape == (3,)
        assert pos.dtype == np.float64

        # Should have finite values
        assert np.all(np.isfinite(pos))


def test_skeleton_bounds(ts2_skeleton):
    """Test bounding box calculation using node positions."""
    all_positions = ts2_skeleton.get_all_positions()

    bounds_min = all_positions.min(axis=0)
    bounds_max = all_positions.max(axis=0)

    assert bounds_min.shape == (3,)
    assert bounds_max.shape == (3,)

    # Min should be less than max in all dimensions
    assert np.all(bounds_min < bounds_max)

    print(f"\nBounds: min={bounds_min}, max={bounds_max}")


def test_skeleton_centroid(ts2_skeleton):
    """Test centroid calculation using node positions."""
    all_positions = ts2_skeleton.get_all_positions()
    centroid = all_positions.mean(axis=0)

    assert centroid.shape == (3,)
    assert np.all(np.isfinite(centroid))

    # Centroid should be within bounds
    bounds_min = all_positions.min(axis=0)
    bounds_max = all_positions.max(axis=0)
    assert np.all(centroid >= bounds_min)
    assert np.all(centroid <= bounds_max)

    print(f"\nCentroid: {centroid}")


def test_total_length(ts2_skeleton):
    """Test total edge length calculation."""
    total_length = ts2_skeleton.get_total_length()

    assert total_length > 0
    assert np.isfinite(total_length)

    print(f"\nTotal length: {total_length:.2f}")


def test_to_polylines_conversion(ts2_skeleton):
    """Test conversion back to polylines format."""
    polylines = ts2_skeleton.to_polylines()

    # Should return a list of arrays
    assert isinstance(polylines, list)
    assert len(polylines) > 0

    # Each polyline should be a 2D array with shape (N, 3)
    for pl in polylines:
        assert isinstance(pl, np.ndarray)
        assert pl.ndim == 2
        assert pl.shape[1] == 3
        assert pl.shape[0] >= 2  # At least 2 points per polyline


def test_copy_skeleton(ts2_skeleton):
    """Test skeleton copying."""
    copy = ts2_skeleton.copy_skeleton()

    # Should have same structure
    assert copy.number_of_nodes() == ts2_skeleton.number_of_nodes()
    assert copy.number_of_edges() == ts2_skeleton.number_of_edges()

    # Should be independent (modifying copy doesn't affect original)
    original_pos = ts2_skeleton.get_node_position(0)
    copy.set_node_position(0, original_pos + np.array([1.0, 1.0, 1.0]))

    # Original should be unchanged
    assert np.allclose(ts2_skeleton.get_node_position(0), original_pos)


def test_save_and_reload(ts2_skeleton, tmp_path):
    """Test saving and reloading skeleton."""
    output_path = tmp_path / "test_skeleton.graphml"

    # Save
    ts2_skeleton.to_txt(str(output_path))
    assert output_path.exists()

    # Reload
    reloaded = SkeletonGraph.from_txt(str(output_path), tolerance=1e-6)

    # Should have similar structure (node count may differ slightly due to endpoint merging)
    # but edge count should be the same
    assert reloaded.number_of_edges() == ts2_skeleton.number_of_edges()

    # Node count should be close (within a few nodes due to endpoint handling)
    node_diff = abs(reloaded.number_of_nodes() - ts2_skeleton.number_of_nodes())
    assert node_diff <= 5, f"Node count difference too large: {node_diff}"

    # Both should have valid total lengths
    assert reloaded.get_total_length() > 0
    assert ts2_skeleton.get_total_length() > 0


def test_total_points(ts2_skeleton):
    """Test total_points method."""
    total_points = ts2_skeleton.total_points()

    # Should equal number of nodes
    assert total_points == ts2_skeleton.number_of_nodes()
    assert total_points > 0


def test_edge_lengths(ts2_skeleton):
    """Test that edges have valid length attributes."""
    for u, v in ts2_skeleton.edges():
        edge_data = ts2_skeleton.edges[u, v]

        # Should have a length attribute
        assert "length" in edge_data
        length = edge_data["length"]

        # Length should be positive and finite
        assert length > 0
        assert np.isfinite(length)

        # Verify length matches distance between nodes
        pos_u = ts2_skeleton.get_node_position(u)
        pos_v = ts2_skeleton.get_node_position(v)
        computed_length = np.linalg.norm(pos_v - pos_u)

        assert np.isclose(length, computed_length, rtol=1e-6)


def test_skeleton_statistics(ts2_skeleton):
    """Print comprehensive statistics about the skeleton."""
    print("\n" + "=" * 70)
    print("TS2 Skeleton Statistics")
    print("=" * 70)
    print(f"Representation: {ts2_skeleton}")
    print(f"Total nodes: {ts2_skeleton.number_of_nodes()}")
    print(f"Total edges: {ts2_skeleton.number_of_edges()}")
    print(f"Terminal nodes: {len(ts2_skeleton.get_terminal_nodes())}")
    print(f"Branch nodes: {len(ts2_skeleton.get_branch_nodes())}")
    print(f"Total length: {ts2_skeleton.get_total_length():.2f}")

    # Compute centroid from positions
    all_positions = ts2_skeleton.get_all_positions()
    centroid = all_positions.mean(axis=0)
    print(f"Centroid: {centroid}")

    # Compute bounds from positions
    bounds_min = all_positions.min(axis=0)
    bounds_max = all_positions.max(axis=0)
    print(
        f"Bounds: [{bounds_min[0]:.2f}, {bounds_max[0]:.2f}] x "
        f"[{bounds_min[1]:.2f}, {bounds_max[1]:.2f}] x "
        f"[{bounds_min[2]:.2f}, {bounds_max[2]:.2f}]"
    )

    # Degree distribution
    degrees = [ts2_skeleton.degree(node) for node in ts2_skeleton.nodes()]
    print(f"\nDegree distribution:")
    print(f"  Min degree: {min(degrees)}")
    print(f"  Max degree: {max(degrees)}")
    print(f"  Mean degree: {np.mean(degrees):.2f}")

    print("=" * 70)

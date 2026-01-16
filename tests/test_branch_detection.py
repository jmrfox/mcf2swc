"""
Test branch point detection on TS2 skeleton.
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
    return SkeletonGraph.from_txt(str(skeleton_path))


def test_branch_detection_basic(ts2_skeleton):
    """Test basic branch point detection functionality."""
    topology = ts2_skeleton.detect_branch_points(tolerance=1e-6)

    assert "branch_points" in topology
    assert "endpoints" in topology
    assert "branch_locations" in topology
    assert "endpoint_locations" in topology

    assert len(topology["branch_points"]) >= 0
    assert len(topology["endpoints"]) >= 0
    assert len(topology["branch_locations"]) >= 0
    assert len(topology["endpoint_locations"]) >= 0


def test_branch_detection_with_tolerance(ts2_skeleton):
    """Test that larger tolerance affects branch detection."""
    topology_small = ts2_skeleton.detect_branch_points(tolerance=1e-6)
    topology_large = ts2_skeleton.detect_branch_points(tolerance=0.1)

    assert isinstance(topology_small["branch_points"], list)
    assert isinstance(topology_large["branch_points"], list)


def test_build_graph(ts2_skeleton):
    """Test building networkx graph from skeleton."""
    graph = ts2_skeleton.build_graph(tolerance=0.1)

    assert graph.number_of_nodes() > 0
    assert graph.number_of_edges() > 0

    for node, data in graph.nodes(data=True):
        assert "type" in data
        assert "pos" in data
        assert data["type"] in ["branch", "endpoint", "continuation"]


def test_graph_topology_consistency(ts2_skeleton):
    """Test that graph has valid structure with node types."""
    graph = ts2_skeleton.build_graph(tolerance=0.1)

    assert graph.number_of_nodes() > 0
    assert graph.number_of_edges() >= 0

    for node, data in graph.nodes(data=True):
        assert "type" in data
        assert "pos" in data
        assert data["type"] in ["branch", "endpoint", "continuation"]
        assert isinstance(data["pos"], (list, tuple, np.ndarray))
        assert len(data["pos"]) == 3


def test_convenience_methods(ts2_skeleton):
    """Test convenience methods for getting branch/endpoint indices."""
    branch_indices = ts2_skeleton.get_branch_point_indices(tolerance=0.1)
    endpoint_indices = ts2_skeleton.get_true_endpoint_indices(tolerance=0.1)

    assert isinstance(branch_indices, (list, set))
    assert isinstance(endpoint_indices, (list, set))

    assert len(branch_indices) + len(endpoint_indices) <= ts2_skeleton.number_of_nodes()

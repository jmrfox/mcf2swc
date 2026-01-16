"""
Test branch pruning on TS2 skeleton.
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


def test_prune_short_branches_basic(ts2_skeleton):
    """Test basic branch pruning functionality."""
    original_count = ts2_skeleton.number_of_nodes()

    pruned = ts2_skeleton.prune_short_branches(min_length=10.0, verbose=False)

    assert pruned.number_of_nodes() <= original_count
    assert pruned.total_points() <= ts2_skeleton.total_points()


def test_prune_short_branches_length_threshold(ts2_skeleton):
    """Test that pruning removes branches below length threshold."""
    min_length = 10.0
    pruned = ts2_skeleton.prune_short_branches(min_length=min_length, verbose=False)

    terminal_nodes = pruned.get_terminal_nodes()
    branch_lengths = pruned.compute_branch_lengths()
    for (u, v), length in branch_lengths.items():
        if u in terminal_nodes or v in terminal_nodes:
            assert length >= min_length or np.isclose(length, min_length, rtol=0.1)


def test_prune_percentile_based(ts2_skeleton):
    """Test percentile-based pruning."""
    original_n = ts2_skeleton.number_of_nodes()
    pruned = ts2_skeleton.prune_short_branches(min_length_percentile=20, verbose=False)

    assert isinstance(pruned, SkeletonGraph)
    assert pruned.number_of_nodes() <= original_n


def test_pruning_preserves_topology(ts2_skeleton):
    """Test that pruning maintains valid topology."""
    pruned = ts2_skeleton.prune_short_branches(min_length=10.0, verbose=False)

    topology = pruned.detect_branch_points(tolerance=1e-6)

    assert "branch_points" in topology
    assert "endpoints" in topology
    assert isinstance(topology["branch_points"], list)
    assert isinstance(topology["endpoints"], list)

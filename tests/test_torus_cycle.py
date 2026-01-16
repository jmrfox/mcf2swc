"""Test that torus polyline creates a cycle in MorphologyGraph."""

from pathlib import Path
import tempfile

import numpy as np
import networkx as nx

from mcf2swc import (
    fit_morphology,
    FitOptions,
    SkeletonGraph,
    example_mesh,
    SWCModel,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "demo"


def test_torus_creates_cycle():
    """Test that fitting a closed torus polyline creates a cycle in the MorphologyGraph."""
    # Load torus mesh and polylines
    mesh = example_mesh("torus")
    skel = SkeletonGraph.from_txt(str(DATA / "torus.polylines.txt"))

    # Fit morphology
    morph = fit_morphology(mesh, skel, options=FitOptions(spacing=1.0))

    # The torus polyline is closed, so we should have a cycle
    cycle_basis = nx.cycle_basis(morph)

    # Verify we have at least one cycle
    assert (
        len(cycle_basis) > 0
    ), f"Expected at least 1 cycle in torus, got {len(cycle_basis)}"

    # Verify the graph structure
    assert morph.number_of_nodes() > 0
    assert morph.number_of_edges() > 0

    # Export to SWC and verify cycle breaking
    with tempfile.NamedTemporaryFile(mode="w", suffix=".swc", delete=False) as f:
        temp_path = f.name

    try:
        swc_text = morph.to_swc_file(temp_path)

        # Should have cycle break annotation
        assert (
            "CYCLE_BREAK" in swc_text
        ), "Expected CYCLE_BREAK annotation in SWC output"

        # Verify the SWC file is a valid tree
        model = SWCModel.from_swc_file(temp_path)
        assert nx.is_tree(model), "SWC model should be a tree after cycle breaking"

    finally:
        Path(temp_path).unlink(missing_ok=True)

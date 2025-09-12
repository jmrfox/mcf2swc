"""
Visualization utilities for SWC morphology files (Jupyter-focused).

This module provides a simple 3D viewer for SWC files using PyVista/VTK.
It is designed to work inside Jupyter notebooks and supports toggling
between simple line rendering and radius-aware tubes.

Public API:
- show_swc(path_or_obj, use_radii=True, ...)

Notes
-----
- This viewer is meant for exploratory visualization in notebooks, not as a
  full-featured GUI. It handles morphologies with up to several thousand
  points comfortably.
- Radii rendering uses a tube filter; when radii are disabled, segments are
  shown as simple lines with configurable width.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pyvista as pv

# Optional: use navis if present, but we don't require it
try:  # pragma: no cover - exercised in runtime environments with navis
    import navis as _navis  # type: ignore
    _HAS_NAVIS = True
except Exception:  # pragma: no cover - avoid import errors in tests
    _HAS_NAVIS = False


ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]
PathLike = Union[str, Path]


def _parse_swc_text(lines: Iterable[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse SWC lines into arrays.

    Returns
    -------
    points : (N, 3) float64
        XYZ coordinates.
    edges : (M, 2) int64
        Child-parent index pairs (0-based indices into ``points``).
    radii : (N,) float64
        Node radii.
    types : (N,) int64
        SWC type for each node (1=soma, 2=axon, 3=dendrite, 4=apical, ...).
    """
    ids: List[int] = []
    types: List[int] = []
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    rs: List[float] = []
    parents: List[int] = []

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 7:
            # Ignore malformed rows silently; viewer is best-effort
            continue
        try:
            nid = int(float(parts[0]))
            ntype = int(float(parts[1]))
            x = float(parts[2])
            y = float(parts[3])
            z = float(parts[4])
            r = float(parts[5])
            parent = int(float(parts[6]))
        except Exception:
            # Skip lines that cannot be parsed
            continue
        ids.append(nid)
        types.append(ntype)
        xs.append(x)
        ys.append(y)
        zs.append(z)
        rs.append(r)
        parents.append(parent)

    if not ids:
        raise ValueError("No valid SWC nodes parsed from input")

    # Map SWC ids to contiguous 0..N-1 indices
    order = np.argsort(np.asarray(ids, dtype=np.int64))
    ids_sorted = np.asarray(ids, dtype=np.int64)[order]
    id_to_idx = {int(i): int(k) for k, i in enumerate(ids_sorted.tolist())}

    points = np.column_stack(
        [np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), np.asarray(zs, dtype=float)]
    )[order]
    radii = np.asarray(rs, dtype=float)[order]
    types_arr = np.asarray(types, dtype=np.int64)[order]
    parents_sorted = np.asarray(parents, dtype=np.int64)[order]

    # Build edges from child to parent (skip parent == -1)
    edges: List[Tuple[int, int]] = []
    for k, pid in enumerate(parents_sorted.tolist()):
        if int(pid) == -1:
            continue
        pidx = id_to_idx.get(int(pid))
        if pidx is None:
            # Parent missing in file; skip this edge
            continue
        edges.append((k, pidx))

    if not edges:
        # Degenerate but still valid: single node soma?
        edges_arr = np.empty((0, 2), dtype=np.int64)
    else:
        edges_arr = np.asarray(edges, dtype=np.int64)

    return points, edges_arr, radii, types_arr


def _read_swc(path: PathLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read an SWC file from disk and return arrays.

    If ``navis`` is available and can read the file, we will rely on our
    lightweight parser anyway for consistency with edge definitions.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"SWC file not found: {p}")
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        return _parse_swc_text(f)


def _build_polydata(points: ArrayLike, edges: ArrayLike, radii: ArrayLike, types: ArrayLike) -> pv.PolyData:
    """Create a PyVista PolyData with line cells and point data.

    Parameters
    ----------
    points : (N, 3)
    edges : (M, 2)
    radii : (N,)
    types : (N,)
    """
    pts = np.asarray(points, dtype=float)
    e = np.asarray(edges, dtype=np.int64)
    r = np.asarray(radii, dtype=float)
    t = np.asarray(types, dtype=np.int64)

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be (N, 3)")
    if r.shape[0] != pts.shape[0] or t.shape[0] != pts.shape[0]:
        raise ValueError("radii/types length must match number of points")
    if e.size == 0:
        # Create a single vertex-only polydata to show e.g. a soma-only SWC
        poly = pv.PolyData(pts)
        poly["radius"] = r
        poly["type"] = t
        return poly

    # VTK lines cell array format: [2, i0, j0, 2, i1, j1, ...]
    lines_list: List[int] = []
    for i, j in e.tolist():
        lines_list.extend([2, int(i), int(j)])
    lines = np.asarray(lines_list, dtype=np.int64)

    poly = pv.PolyData(pts, lines=lines)
    # Store scalars as point data
    poly["radius"] = r
    poly["type"] = t
    return poly


def show_swc(
    swc: PathLike,
    *,
    use_radii: bool = True,
    background: str = "white",
    line_color: str = "#2a5599",
    line_width: float = 2.0,
    tube_sides: int = 16,
    min_radius: Optional[float] = None,
    show_axes: bool = True,
    show: bool = True,
    return_plotter: bool = False,
) -> pv.Plotter:
    """Display an SWC morphology in 3D using PyVista (Jupyter-friendly).

    Parameters
    ----------
    swc : str or Path
        Path to an SWC file.
    use_radii : bool, default True
        If True, render segments as tubes using the SWC radii. If False, render
        as simple lines with ``line_width``.
    background : str, default 'white'
        Plotter background color.
    line_color : str, default '#2a5599'
        Color for lines/tubes.
    line_width : float, default 2.0
        Screen-space line width when ``use_radii=False``.
    tube_sides : int, default 16
        Number of sides for the tube filter when ``use_radii=True``.
    min_radius : float, optional
        Clamp radii to at least this value (useful for near-zero radii).
        If None, an automatic epsilon based on the morphology scale is used.
    show_axes : bool, default True
        Show a small axes actor.
    show : bool, default True
        If True, call ``plotter.show()``; set False in tests or to embed later.
    return_plotter : bool, default False
        If True, return the ``pyvista.Plotter`` instance for further tweaking.

    Returns
    -------
    plotter : pv.Plotter
        The configured plotter. Returned regardless of ``show``.
    """
    points, edges, radii, types = _read_swc(swc)

    # Clamp radii to avoid degenerate zero-radius tubes
    if use_radii:
        r = np.asarray(radii, dtype=float)
        if min_radius is None:
            # Use ~0.2% of the spatial extent as a fallback epsilon
            extent = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
            eps = max(1e-6, 0.002 * extent)
        else:
            eps = float(min_radius)
        radii = np.where(np.asarray(r) > eps, r, eps)

    poly = _build_polydata(points, edges, radii, types)

    # Create a Jupyter-friendly plotter
    plotter = pv.Plotter(notebook=True)
    plotter.set_background(background)

    if edges.size == 0:
        # No edges: show points only
        plotter.add_points(poly, color=line_color, point_size=max(3.0, line_width * 2))
    elif use_radii:
        # Use PyVista's tube filter with varying radius by point scalars
        try:
            tubed = poly.tube(scalars="radius", n_sides=int(tube_sides), capping=True)
        except Exception:
            # Fallback: render as lines if tube filter fails
            tubed = None
        if tubed is not None:
            plotter.add_mesh(tubed, color=line_color, smooth_shading=True)
        else:
            plotter.add_mesh(poly, color=line_color, line_width=line_width)
    else:
        # Simple lines without radii
        plotter.add_mesh(poly, color=line_color, line_width=line_width)

    if show_axes:
        plotter.add_axes(line_width=1, labels_off=False)

    if show:
        # In notebooks, this returns a display object and keeps the plotter alive
        plotter.show()

    if return_plotter:
        return plotter
    return plotter


__all__ = [
    "show_swc",
]

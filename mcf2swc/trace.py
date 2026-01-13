"""
This module builds an SWC model by tracing along user-provided skeleton polylines and
estimating radii from local mesh cross-sections.

Terminology:
- "skeleton": The mesh centroid (polylines format, result of MCF calculation) without radii
- "SWC model" or "swc": Skeleton with radii information attached to each node

High level:
- Resample each input skeleton polyline at approximately fixed arc-length spacing.
- For each resampled point, fit an equivalent circle radius:
    * equivalent_area: r = sqrt(A / pi)
    * equivalent_perimeter: r = L / (2*pi), using the exterior boundary length
    * section_median: median ray-to-boundary distance in the local section
      plane from the sample origin (robust for irregular/partial sections).
    * section_circle_fit: algebraic circle fit (Kåsa) to the section boundary.
    * nearest_surface: distance from the sample point to nearest mesh surface
      (bypasses sectioning entirely, useful as a robust fallback).
- Create an SWC model with:
    - one node per sample at position P (xyz, the exact polyline coordinate) with the
      fitted radius
    - edges connecting consecutive samples along each polyline

Notes:
- This module intentionally does not change connectivity beyond linking samples
  along each polyline.
- Coordinates are taken directly from the skeleton polylines (no snapping by default).
  An optional snap-to-mesh pass is provided via PolylinesSkeleton if desired.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import shapely.geometry as sgeom
import trimesh

from swctools import SWCModel

from .polylines import PolylinesSkeleton
from .mesh import MeshManager

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class TraceOptions:
    """
    Configuration options for polyline-guided tracing and local radius estimation.

    These options control how input polylines are sampled, how local
    cross-sections are probed and selected, and how radii are estimated from those
    sections (or from the surface directly).

    Attributes:
        spacing: Sampling step along polylines in mesh units. Resampling keeps
            endpoints and inserts additional samples at approximately fixed arc-length.
        radius_strategy: Strategy for estimating node radii at each sample. One of:
            - "equivalent_area" (default): r = sqrt(A/pi) using cross-section area.
            - "equivalent_perimeter": r = L/(2*pi) using exterior boundary length.
            - "section_median": median ray-to-boundary distance in the local section
              plane from the sample origin (robust for irregular/partial sections).
            - "section_circle_fit": algebraic circle fit (Kåsa) to the section boundary.
            - "nearest_surface": distance from the sample point to nearest mesh surface
              (bypasses sectioning entirely, useful as a robust fallback).
        section_probe_eps: Step size (scaled by mesh bbox) for offsetting the section
            plane origin along the local normal when the exact plane yields no curves.
        section_probe_tries: Number of +/- k*eps offsets to try when seeking a section.
        snap_polylines_to_mesh: If True, project polylines to nearest surface before
            tracing. Useful when user-provided lines are slightly off the surface.
        max_snap_distance: Optional distance threshold for snapping; larger moves are
            ignored if provided.
    """

    spacing: float = 1.0  # sampling step along polylines (mesh units)
    radius_strategy: str = (
        "equivalent_area"  # {"equivalent_area", "equivalent_perimeter", "section_median", "section_circle_fit", "nearest_surface"}
    )
    # When the exact plane P,t yields an empty section, try small offsets
    # along the normal by +/- k * eps until found or max_tries.
    section_probe_eps: float = 1e-4
    section_probe_tries: int = 3
    # Optional: snap polylines to mesh surface prior to tracing
    snap_polylines_to_mesh: bool = False
    max_snap_distance: Optional[float] = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_traced_skeleton_graph(
    mesh: trimesh.Trimesh | MeshManager,
    polylines: PolylinesSkeleton,
    *,
    options: Optional[TraceOptions] = None,
) -> SWCModel:
    """
    Trace skeleton polylines over a mesh and build an SWC model with nodes at sampled
    polyline points and radii estimated from local mesh cross-sections.

    - Resamples each input polyline with spacing `options.spacing`.
    - For each sample P with tangent T, intersects the mesh with the plane
      (origin=P, normal=T) and selects the polygon that covers/contains the local
      origin, or the one whose boundary is closest to it.
    - Estimates a radius using `options.radius_strategy` (equivalent area, equivalent
      perimeter, section median, section circle fit, or nearest surface).
    - If no section is found near the exact plane, tries small offsets along ±T
      controlled by `section_probe_eps` and `section_probe_tries`.
    - If still no section is found, falls back to nearest surface distance.

    Records per-node diagnostics:
      - `radius_strategy`: which strategy actually provided the radius
      - `inside_mesh`: whether the sample point appears inside the mesh (signed distance <= 0)

    Args:
        mesh: The mesh to intersect against (`trimesh.Trimesh`)
        polylines: Skeleton polyline guidance, in the same frame as the mesh.
        options: `TraceOptions` controlling sampling, radius strategy, and section probing.

    Returns:
        A populated SWC model (`swctools.SWCModel`).
    """
    if options is None:
        options = TraceOptions()

    # Resolve mesh
    if not isinstance(mesh, (trimesh.Trimesh, MeshManager)):
        raise TypeError("mesh must be a trimesh.Trimesh or MeshManager")

    if isinstance(mesh, MeshManager):
        mesh = mesh.mesh

    if len(mesh.vertices) == 0:
        raise ValueError("Mesh is empty or not provided")

    # Start logging summary
    try:
        n_pl = len(getattr(polylines, "polylines", []) or [])
        total_pts = 0
        try:
            total_pts = int(polylines.total_points())
        except Exception:
            pass
        logger.info(
            "Tracing start: mesh[V=%d,F=%d], polylines=%d (pts=%d), spacing=%.3g, radius_strategy=%s",
            len(mesh.vertices),
            len(mesh.faces),
            n_pl,
            total_pts,
            float(options.spacing),
            str(options.radius_strategy),
        )
    except Exception:
        pass

    # Optionally snap polylines to the mesh surface
    pls = polylines.copy()
    if options.snap_polylines_to_mesh:
        try:
            moved, mean = pls.snap_to_mesh_surface(
                mesh,
                project_outside_only=True,
                max_distance=options.max_snap_distance,
            )
            logger.info(
                "Polylines snapped to mesh surface: moved=%d, mean=%.4g", moved, mean
            )
        except Exception as e:
            logger.warning("Failed snapping polylines to mesh: %s", e)

    # Pre-compute mesh scale for section probing epsilon
    V = np.asarray(mesh.vertices, dtype=float)
    bbox_size = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0))) if V.size else 1.0
    eps = max(
        1e-12, float(options.section_probe_eps) * (bbox_size if bbox_size > 0 else 1.0)
    )

    # Build a KDTree over mesh vertices for robust radius fallback (if SciPy available)
    v_kdtree = None
    try:
        if V.size > 0:
            from scipy.spatial import cKDTree  # type: ignore

            v_kdtree = cKDTree(V)
    except Exception:
        v_kdtree = None

    # Build SWC model
    skel = SWCModel()

    # Helper to allocate node ids
    next_id = 0

    def alloc_id() -> int:
        nonlocal next_id
        nid = next_id
        next_id += 1
        return nid

    # Diagnostics counters
    total_samples = 0
    used_section = 0
    used_fallback = 0

    # Spatial index for deduplicating nodes by overlapping coordinates
    # Use quantization by tolerance cell to find candidates, then check true distance
    quant_tol = max(1e-9, 1e-3 * float(options.spacing))

    def _quant_key(P: np.ndarray) -> tuple[int, int, int]:
        return tuple(np.round(np.asarray(P, dtype=float) / quant_tol).astype(int))  # type: ignore[return-value]

    pos_index: dict[tuple[int, int, int], tuple[int, np.ndarray]] = {}

    # Process each polyline
    for pl_index, pl in enumerate(pls.as_arrays()):
        if pl is None or pl.size == 0 or pl.shape[0] < 2:
            continue
        # Resample
        samples = _resample_polyline(pl, float(options.spacing))
        try:
            logger.info(
                "Polyline %d: input_pts=%d -> samples=%d",
                int(pl_index),
                int(pl.shape[0]),
                int(samples.shape[0]),
            )
        except Exception:
            pass
        if samples.shape[0] == 0:
            continue
        # Precompute tangents on the resampled curve
        tangents = _estimate_tangents(samples)

        prev_node: Optional[int] = None
        first_node: Optional[int] = None
        for i in range(samples.shape[0]):
            P = samples[i]
            tangent = tangents[i]
            if (
                not np.all(np.isfinite(tangent))
                or float(np.linalg.norm(tangent)) <= 1e-12
            ):
                # Fallback tangent from original segment if degenerate
                if i + 1 < samples.shape[0]:
                    tangent = samples[i + 1] - samples[i]
                elif i > 0:
                    tangent = samples[i] - samples[i - 1]
                else:
                    tangent = np.array([0.0, 0.0, 1.0], dtype=float)
                tangent = tangent / (np.linalg.norm(tangent) + 1e-12)

            # Use tangent as normal for cross-section plane
            n = tangent

            # Fit local radius according to selected mode
            radius = 0.0
            radius_strategy = "unknown"
            inside_mesh = None

            # Inside/outside diagnostic (does not alter logic; used for debugging)
            try:
                from trimesh.proximity import signed_distance  # type: ignore

                sd = float(signed_distance(mesh, P.reshape(1, 3))[0])
                # Convention: positive outside in trimesh; <= 0 means inside/on
                inside_mesh = bool(sd <= 0.0)
            except Exception:
                inside_mesh = None

            # Special case: explicitly request nearest surface distance
            if options.radius_strategy == "nearest_surface":
                radius = _nearest_surface_distance(P, mesh, V, v_kdtree)
                radius_strategy = "nearest_surface"
            else:
                # Try cross-section first
                poly2d = _cross_section_polygon_near_point(
                    mesh=mesh,
                    origin=P,
                    normal=n,
                    eps=eps,
                    max_tries=int(options.section_probe_tries),
                )
                if poly2d is not None:
                    used_section += 1
                    area = float(poly2d.area)
                    mode = str(options.radius_strategy)
                    if mode == "equivalent_perimeter":
                        perim = float(poly2d.exterior.length)
                        radius = perim / (2.0 * math.pi) if perim > 0 else 0.0
                        radius_strategy = "equivalent_perimeter"
                    elif mode == "section_median":
                        radius = _radius_from_section_median(poly2d)
                        radius_strategy = "section_median"
                    elif mode == "section_circle_fit":
                        r_fit = _radius_from_section_circle_fit(poly2d)
                        if not np.isfinite(r_fit) or r_fit <= 0:
                            # conservative fallback to equivalent area
                            radius = math.sqrt(area / math.pi) if area > 0 else 0.0
                            radius_strategy = "equivalent_area_fallback"
                        else:
                            radius = float(r_fit)
                            radius_strategy = "section_circle_fit"
                    else:  # "equivalent_area" (default) or unknown => area-based
                        radius = math.sqrt(area / math.pi) if area > 0 else 0.0
                        radius_strategy = "equivalent_area"
                else:
                    # No section found; robust fallback = nearest surface distance
                    area = 0.0
                    used_fallback += 1
                    radius = _nearest_surface_distance(P, mesh, V, v_kdtree)
                    radius_strategy = "nearest_surface_fallback"

            total_samples += 1
            # Per-sample diagnostics (DEBUG)
            try:
                logger.debug(
                    "pl=%d i=%d P=(%.4g,%.4g,%.4g) r=%.4g source=%s inside=%s",
                    int(pl_index),
                    int(i),
                    float(P[0]),
                    float(P[1]),
                    float(P[2]),
                    float(radius),
                    str(radius_strategy),
                    str(inside_mesh),
                )
            except Exception:
                pass

            # Add or reuse node if overlapping an existing coordinate
            qk = _quant_key(P)
            nid: int
            reused = False
            if qk in pos_index:
                existing_id, existing_xyz = pos_index[qk]
                if (
                    float(np.linalg.norm(np.asarray(P, dtype=float) - existing_xyz))
                    <= quant_tol
                ):
                    nid = int(existing_id)
                    reused = True
                else:
                    nid = alloc_id()
                    skel.add_junction(
                        node_id=nid,
                        x=float(P[0]),
                        y=float(P[1]),
                        z=float(P[2]),
                        r=float(radius),
                    )
                    pos_index[qk] = (nid, np.asarray(P, dtype=float))
            else:
                nid = alloc_id()
                skel.add_junction(
                    node_id=nid,
                    x=float(P[0]),
                    y=float(P[1]),
                    z=float(P[2]),
                    r=float(radius),
                )
                pos_index[qk] = (nid, np.asarray(P, dtype=float))

            if not reused:
                # Attach source metadata on the graph node for diagnostics
                try:
                    skel.nodes[nid]["radius_strategy"] = radius_strategy
                    if inside_mesh is not None:
                        skel.nodes[nid]["inside_mesh"] = bool(inside_mesh)
                except Exception:
                    pass
            else:
                try:
                    logger.debug(
                        "pl=%d i=%d: reused existing node %d for overlapping position",
                        int(pl_index),
                        int(i),
                        int(nid),
                    )
                except Exception:
                    pass

            # Edge from previous sample
            if prev_node is not None and prev_node != nid:
                try:
                    skel.add_edge(
                        prev_node, nid, kind="trace", polyline_index=int(pl_index)
                    )
                except Exception:
                    pass
            prev_node = nid
            if first_node is None:
                first_node = nid

        # No explicit closure here; cycles emerge naturally when endpoints overlap

    # Final summary
    try:
        logger.info(
            "Tracing done: nodes=%d, edges=%d, samples=%d, section=%d, fallback=%d",
            int(skel.number_of_nodes()),
            int(skel.number_of_edges()),
            int(total_samples),
            int(used_section),
            int(used_fallback),
        )
    except Exception:
        pass

    return skel


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _resample_polyline(pl: np.ndarray, spacing: float) -> np.ndarray:
    """Resample a polyline at approximately constant arc-length spacing.

    Includes the first and last vertex; inserts intermediate points every
    multiple of `spacing` along cumulative arclength.
    """
    P = np.asarray(pl, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3 or P.shape[0] == 0:
        return np.zeros((0, 3), dtype=float)
    if P.shape[0] == 1:
        return P.copy()

    seg = np.linalg.norm(P[1:] - P[:-1], axis=1)
    L = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(L[-1])

    if total <= 0.0:
        return P[[0], :].copy()

    step = float(max(spacing, 1e-12))
    # Always include start and end
    targets = list(np.arange(0.0, total, step))
    if targets[-1] != total:
        targets.append(total)

    out: List[np.ndarray] = []
    si = 0  # segment index
    for t in targets:
        # advance si until L[si] <= t <= L[si+1]
        while si < len(seg) and L[si + 1] < t:
            si += 1
        if si >= len(seg):
            out.append(P[-1])
            continue
        t0 = L[si]
        t1 = L[si + 1]
        if t1 <= t0:
            out.append(P[si])
            continue
        alpha = (t - t0) / (t1 - t0)
        Q = (1.0 - alpha) * P[si] + alpha * P[si + 1]
        out.append(Q)

    return np.vstack(out) if out else P[[0], :].copy()


def _estimate_tangents(P: np.ndarray) -> np.ndarray:
    """Estimate unit tangents for a polyline represented by points P."""
    n = P.shape[0]
    T = np.zeros_like(P)
    if n == 1:
        T[0] = np.array([0.0, 0.0, 1.0], dtype=float)
        return T
    for i in range(n):
        if i == 0:
            v = P[1] - P[0]
        elif i == n - 1:
            v = P[n - 1] - P[n - 2]
        else:
            v = P[i + 1] - P[i - 1]
        norm = float(np.linalg.norm(v))
        if norm <= 1e-12 or not np.isfinite(norm):
            T[i] = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            T[i] = v / norm
    return T


def _plane_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return orthonormal (u, v, n) basis for a plane with normal n.

    u and v span the plane; n is the normalized input normal.
    """
    n = np.asarray(normal, dtype=float)
    n = n / (np.linalg.norm(n) + 1e-12)
    # pick a not-near-parallel axis
    ax = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, ax)
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(n, u)
    v /= np.linalg.norm(v) + 1e-12
    return u, v, n


def _world_to_local_plane(P: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Return 4x4 transform that maps world -> local plane coords centered at P.

    In local coordinates, the plane is z=0; x,y span the plane.
    """
    u, v, n = _plane_basis(normal)
    R = np.column_stack([u, v, n])  # world axes -> local
    # We want X_local = R^T * (X_world - P)
    M = np.eye(4, dtype=float)
    M[:3, :3] = R.T
    M[:3, 3] = -R.T @ np.asarray(P, dtype=float)
    return M


def _compose_polygons_with_holes(polys: List[sgeom.Polygon]) -> List[sgeom.Polygon]:
    """Compose simple polygons into polygons-with-holes by containment parity.

    Mirrors mcf2swc.skeleton._compose_polygons_with_holes but kept local to avoid
    tight coupling.
    """
    if not polys:
        return []
    # Precompute bounds and pairwise containment depth
    n = len(polys)
    contains = [[False] * n for _ in range(n)]
    depths = [0] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            try:
                contains[i][j] = polys[i].contains(polys[j])
            except Exception:
                contains[i][j] = False

    def depth_of(idx: int) -> int:
        d = 0
        for k in range(n):
            if contains[k][idx]:
                d += 1
        return d

    depths = [depth_of(i) for i in range(n)]
    # Group children under each even-depth parent
    result: List[sgeom.Polygon] = []
    for i in range(n):
        if depths[i] % 2 != 0:
            continue  # skip holes at odd depth
        exterior_poly = polys[i]
        holes_coords: List[List[Tuple[float, float]]] = []
        for j in range(n):
            if i == j:
                continue
            if contains[i][j] and depths[j] == depths[i] + 1:
                try:
                    ring = list(polys[j].exterior.coords)
                    holes_coords.append([(float(x), float(y)) for x, y in ring])
                except Exception:
                    continue
        try:
            composed = sgeom.Polygon(exterior_poly.exterior.coords, holes=holes_coords)
            if composed.is_valid and composed.area > 0:
                result.append(composed)
        except Exception:
            # Fallback to original exterior only
            if exterior_poly.is_valid and exterior_poly.area > 0:
                result.append(exterior_poly)
    return result


def _cross_section_polygon_near_point(
    mesh: trimesh.Trimesh,
    origin: np.ndarray,
    normal: np.ndarray,
    *,
    eps: float,
    max_tries: int,
) -> Optional[sgeom.Polygon]:
    """
    Intersect `mesh` with the plane through `origin` and normal `normal`, returning
    a single 2D polygon in the local plane most consistent with `origin`.

    Selection preference:
    1) A polygon that contains the local origin (0,0)
    2) Otherwise, polygon with minimal distance from its boundary to the origin

    If no section is found exactly at the plane, try small offsets along the
    normal by +/- k*eps for k=1..max_tries. Returns None if no polygons found.
    """
    P = np.asarray(origin, dtype=float)
    n = np.asarray(normal, dtype=float)
    n = n / (np.linalg.norm(n) + 1e-12)

    # Try origin plane and probe offsets; if none found, auto-expand eps
    for scale in (1.0, 2.0, 4.0):
        eps_s = float(eps) * scale
        offsets = [0.0]
        for k in range(1, int(max_tries) + 1):
            offsets.extend([+k * eps_s, -k * eps_s])

        for off in offsets:
            o = P + off * n
            try:
                path = mesh.section(plane_origin=o, plane_normal=n)
            except Exception:
                path = None
            if path is None:
                continue
            # Accept either entities or discrete loops; skip only if both are absent/empty
            entities_candidate = getattr(path, "entities", None)
            has_entities = (
                entities_candidate is not None and len(entities_candidate) > 0
            )
            loops_candidate = getattr(path, "discrete", None)
            has_loops = loops_candidate is not None and len(loops_candidate) > 0
            if not (has_entities or has_loops):
                continue

        # Map 3D curve points to local plane 2D
        M = _world_to_local_plane(P, n)
        polys_2d: List[sgeom.Polygon] = []
        try:
            # Prefer discrete polylines assembled by trimesh over raw entities
            loops = getattr(path, "discrete", None) or []
            if loops:
                src_iter = loops
            else:
                # Fall back to raw entities' points
                src_iter = [
                    np.asarray(getattr(ent, "points", None), dtype=float)
                    for ent in getattr(path, "entities", [])
                ]
            for pts3 in src_iter:
                pts3 = np.asarray(pts3, dtype=float)
                if pts3.ndim != 2 or pts3.shape[1] != 3 or pts3.shape[0] < 2:
                    continue
                # homogeneous transform into local plane centered at P
                ones = np.ones((pts3.shape[0], 1), dtype=float)
                v2 = (M @ np.hstack([pts3, ones]).T).T[:, :3]
                XY = v2[:, :2]
                if XY.shape[0] < 3:
                    continue
                if not np.allclose(XY[0], XY[-1]):
                    XY = np.vstack([XY, XY[0]])
                poly = sgeom.Polygon(XY)
                if poly.is_valid and poly.area > 0:
                    polys_2d.append(poly)
        except Exception:
            polys_2d = []

        if not polys_2d:
            continue

        # Compose holes by containment parity
        try:
            composed = _compose_polygons_with_holes(polys_2d)
        except Exception:
            composed = polys_2d
        if not composed:
            continue

        origin_pt = sgeom.Point(0.0, 0.0)
        # Prefer polygons that contain or cover the origin (treat boundary as inside)
        containing: List[sgeom.Polygon] = []
        for poly in composed:
            try:
                # Shapely 2: covers includes boundary; fallback to contains
                if hasattr(poly, "covers") and poly.covers(origin_pt):  # type: ignore[attr-defined]
                    containing.append(poly)
                elif poly.contains(origin_pt):
                    containing.append(poly)
            except Exception:
                continue
        if containing:
            # If multiple, pick the one with smallest area to be conservative
            containing.sort(key=lambda p: float(p.area))
            return containing[0]
        # Else pick polygon with minimum distance to origin
        composed.sort(key=lambda p: float(p.exterior.distance(origin_pt)))
        return composed[0]

    return None


def _nearest_surface_distance(
    P: np.ndarray,
    mesh: trimesh.Trimesh,
    V: np.ndarray,
    v_kdtree: Optional[object],
) -> float:
    """Distance from point P to nearest point on mesh surface.

    Uses trimesh.proximity.closest_point when available; falls back to KDTree over
    vertices or a brute-force vertex distance as a last resort.
    """
    try:
        # Prefer trimesh closest_point for accurate surface distance
        try:
            from trimesh.proximity import closest_point  # type: ignore

            CP, dist, _tri = closest_point(mesh, P.reshape(1, 3))
            _ = CP  # unused
            return float(dist[0])
        except Exception:
            pass
        # Fallback to KDTree over vertices if available
        if v_kdtree is not None and V.size > 0:
            dd, _ = v_kdtree.query(P, k=1)  # type: ignore[attr-defined]
            return float(dd)
        # Last resort: brute-force to vertices
        if V.size > 0:
            return float(np.min(np.linalg.norm(V - P, axis=1)))
        return 0.0
    except Exception:
        return 0.0


def _radius_from_section_median(poly: sgeom.Polygon, *, n_rays: int = 64) -> float:
    """Robust radius estimate as the median distance from origin to exterior.

    Samples n_rays uniformly spaced directions and intersects rays from the origin
    (0,0) with the polygon exterior boundary, ignoring holes. Returns the median of
    the first intersection distances. If intersections fail, returns 0.0.
    """
    try:
        if poly is None or not poly.is_valid or poly.area <= 0:
            return 0.0
        origin = sgeom.Point(0.0, 0.0)
        # Prefer only when origin lies inside polygon (in local plane)
        if not poly.contains(origin):
            # fall back to equivalent area notion if outside; be conservative
            A = float(poly.area)
            return math.sqrt(A / math.pi) if A > 0 else 0.0

        # Determine an adequate ray length based on bounds
        minx, miny, maxx, maxy = poly.bounds
        R = float(max(maxx - minx, maxy - miny)) * 2.0
        if not np.isfinite(R) or R <= 0:
            R = float(max(1.0, math.sqrt(poly.area / math.pi) * 4.0))

        distances: List[float] = []
        for k in range(int(max(8, n_rays))):
            th = (2.0 * math.pi) * (k / float(n_rays))
            dx = math.cos(th)
            dy = math.sin(th)
            ray = sgeom.LineString([(0.0, 0.0), (dx * R, dy * R)])
            try:
                inter = ray.intersection(poly.exterior)
            except Exception:
                continue
            # inter may be MultiPoint or Point
            pts: List[Tuple[float, float]] = []
            if inter.is_empty:
                continue
            if hasattr(inter, "geoms"):
                for g in inter.geoms:  # type: ignore[attr-defined]
                    if hasattr(g, "x") and hasattr(g, "y"):
                        pts.append((float(g.x), float(g.y)))
            else:
                if hasattr(inter, "x") and hasattr(inter, "y"):
                    pts.append((float(inter.x), float(inter.y)))
            if not pts:
                continue
            # Choose the nearest positive distance along the ray
            best = None
            for x, y in pts:
                d = math.hypot(x, y)
                # ensure same direction as (dx,dy) via dot >= 0
                if x * dx + y * dy >= -1e-9:
                    if best is None or d < best:
                        best = d
            if best is not None and np.isfinite(best):
                distances.append(float(best))
        if not distances:
            return 0.0
        distances.sort()
        m = distances[len(distances) // 2]
        return float(m)
    except Exception:
        return 0.0


def _radius_from_section_circle_fit(poly: sgeom.Polygon) -> float:
    """Fit an algebraic circle to the exterior ring and return its radius.

    Uses a simple least-squares Kåsa fit on the exterior coords. Returns 0 on
    failure or degeneracy.
    """
    try:
        if poly is None or not poly.is_valid or poly.area <= 0:
            return 0.0
        coords = np.asarray(poly.exterior.coords, dtype=float)
        if coords.ndim != 2 or coords.shape[0] < 3:
            return 0.0
        XY = coords[:, :2]
        # Remove duplicate final point if closed
        if XY.shape[0] >= 2 and np.allclose(XY[0], XY[-1]):
            XY = XY[:-1]
        if XY.shape[0] < 3:
            return 0.0
        x = XY[:, 0]
        y = XY[:, 1]
        A = np.column_stack([x, y, np.ones_like(x)])
        b = -(x * x + y * y)
        # Solve A*[a,b,c]^T ~ b
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        a, b2, c = sol
        cx = -0.5 * a
        cy = -0.5 * b2
        r2 = cx * cx + cy * cy - c
        if not np.isfinite(r2) or r2 <= 0:
            return 0.0
        r = math.sqrt(float(r2))
        return float(r)
    except Exception:
        return 0.0

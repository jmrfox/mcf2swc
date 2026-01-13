"""
Polylines-based skeleton handler.

Provides a `PolylinesSkeleton` class to:
- Load/save skeleton polylines from/to text format lines: `N x1 y1 z1 x2 y2 z2 ...`
- Manage a transform stack similar to `MeshManager` (apply, track, undo, reset).
- Copy transforms from a `MeshManager` (either composite matrix or full stack).
- Snap/project polyline points to the nearest mesh surface to handle small
  outside deviations.

Note: This class is designed to guide the skeletonization process optionally.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np
import trimesh

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class PolylinesSkeleton:
    """
    Container for a set of 3D polylines with transform management.

    Attributes:
        polylines: List of (N_i, 3) arrays of float64
    """

    def __init__(
        self,
        polylines: Optional[Sequence[np.ndarray]] = None,
        polylines_path: Optional[str] = None,
    ):
        self.polylines: List[np.ndarray] = []
        self.polylines_path = polylines_path
        if polylines is not None:
            for pl in polylines:
                arr = np.asarray(pl, dtype=float)
                if arr.ndim != 2 or arr.shape[1] != 3:
                    raise ValueError("Each polyline must be an (N,3) array")
                self.polylines.append(arr.copy())

        if polylines_path is not None:
            self.load_txt(polylines_path)

    # ---------------------------------------------------------------------
    # IO
    # ---------------------------------------------------------------------
    @staticmethod
    def from_txt(path: str) -> "PolylinesSkeleton":
        """
        Load polylines from a `.polylines.txt` file where each line encodes one
        polyline as: `N x1 y1 z1 x2 y2 z2 ... xN yN zN`.
        """
        polylines: List[np.ndarray] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                s = line.strip()
                if not s:
                    continue
                parts = s.split()
                try:
                    n = int(float(parts[0]))
                except Exception as e:
                    raise ValueError(f"Invalid header count on line {line_no}") from e
                coords = parts[1:]
                if len(coords) != 3 * n:
                    raise ValueError(
                        f"Line {line_no}: expected {3*n} coordinate values, got {len(coords)}"
                    )
                vals = np.array([float(c) for c in coords], dtype=float)
                pl = vals.reshape(n, 3)
                polylines.append(pl)
        return PolylinesSkeleton(polylines)

    def load_txt(self, path: str) -> None:
        """Load polylines into this instance from a `.polylines.txt` file.

        This mirrors `from_txt` but mutates the current instance.

        Example:
            pls = PolylinesSkeleton()
            pls.load_txt("data/polylines/cylinder.polylines.txt")
        """
        loaded = PolylinesSkeleton.from_txt(path)
        self.polylines = loaded.polylines

    def to_txt(self, path: str) -> None:
        """Save polylines to the `.polylines.txt` format described above."""
        with open(path, "w", encoding="utf-8") as f:
            for pl in self.polylines:
                n = int(pl.shape[0])
                flat = " ".join(f"{v:.17g}" for v in pl.reshape(-1))
                f.write(f"{n} {flat}\n")

    # ---------------------------------------------------------------------
    # Basic properties
    # ---------------------------------------------------------------------
    def copy(self) -> "PolylinesSkeleton":
        """Deep-copy the contained polylines and return a new instance."""
        return PolylinesSkeleton([pl.copy() for pl in self.polylines])

    def as_arrays(self) -> List[np.ndarray]:
        return [pl.copy() for pl in self.polylines]

    def total_points(self) -> int:
        return int(sum(pl.shape[0] for pl in self.polylines))

    def bounds(self) -> Optional[Dict[str, Tuple[float, float]]]:
        if not self.polylines:
            return None
        all_pts = np.vstack(self.polylines)
        lo = all_pts.min(axis=0)
        hi = all_pts.max(axis=0)
        return {
            "x": (float(lo[0]), float(hi[0])),
            "y": (float(lo[1]), float(hi[1])),
            "z": (float(lo[2]), float(hi[2])),
        }

    def centroid(self) -> Optional[np.ndarray]:
        if not self.polylines:
            return None
        all_pts = np.vstack(self.polylines)
        return all_pts.mean(axis=0)

    # ---------------------------------------------------------------------
    # Topology and branch point detection
    # ---------------------------------------------------------------------
    def detect_branch_points(self, tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        Detect branch points where multiple polylines meet.

        A branch point is a location where 3 or more polyline segments meet.
        Endpoints of polylines are checked for proximity to other polyline endpoints.

        Args:
            tolerance: Distance threshold for considering points as coincident

        Returns:
            Dictionary containing:
                - 'branch_points': List of (polyline_idx, point_idx) tuples for branch points
                - 'endpoints': List of (polyline_idx, point_idx) tuples for true endpoints
                - 'branch_locations': List of 3D coordinates of branch point locations
                - 'endpoint_locations': List of 3D coordinates of true endpoint locations
        """
        if not self.polylines:
            return {
                "branch_points": [],
                "endpoints": [],
                "branch_locations": [],
                "endpoint_locations": [],
            }

        # Collect all endpoints (first and last point of each polyline)
        endpoint_info = []
        for poly_idx, pl in enumerate(self.polylines):
            if len(pl) > 0:
                endpoint_info.append((poly_idx, 0, pl[0]))  # first point
                if len(pl) > 1:
                    endpoint_info.append((poly_idx, len(pl) - 1, pl[-1]))  # last point

        if len(endpoint_info) == 0:
            return {
                "branch_points": [],
                "endpoints": [],
                "branch_locations": [],
                "endpoint_locations": [],
            }

        # Group endpoints by proximity
        n_endpoints = len(endpoint_info)
        groups = []  # Each group is a list of (poly_idx, point_idx, coord) tuples
        used = set()

        for i in range(n_endpoints):
            if i in used:
                continue

            group = [endpoint_info[i]]
            used.add(i)

            # Find all other endpoints within tolerance
            for j in range(i + 1, n_endpoints):
                if j in used:
                    continue

                dist = np.linalg.norm(endpoint_info[i][2] - endpoint_info[j][2])
                if dist < tolerance:
                    group.append(endpoint_info[j])
                    used.add(j)

            groups.append(group)

        # Classify groups as branch points (3+ connections) or endpoints (1-2 connections)
        branch_points = []
        branch_locations = []
        endpoints = []
        endpoint_locations = []

        for group in groups:
            if len(group) >= 3:
                # Branch point: 3 or more polylines meet here
                for poly_idx, point_idx, coord in group:
                    branch_points.append((poly_idx, point_idx))
                # Use centroid of all points in group as branch location
                coords = np.array([item[2] for item in group])
                branch_locations.append(coords.mean(axis=0))
            else:
                # True endpoint: only 1 or 2 polylines meet (2 means they share an endpoint)
                for poly_idx, point_idx, coord in group:
                    endpoints.append((poly_idx, point_idx))
                coords = np.array([item[2] for item in group])
                endpoint_locations.append(coords.mean(axis=0))

        return {
            "branch_points": branch_points,
            "endpoints": endpoints,
            "branch_locations": branch_locations,
            "endpoint_locations": endpoint_locations,
        }

    def build_graph(self, tolerance: float = 1e-6) -> nx.Graph:
        """
        Build a networkx graph representation of the skeleton topology.

        Nodes represent branch points and endpoints.
        Edges represent polyline segments connecting nodes.

        Args:
            tolerance: Distance threshold for considering points as coincident

        Returns:
            NetworkX Graph where:
                - Nodes have attributes: 'pos' (3D coordinate), 'type' ('branch' or 'endpoint')
                - Edges have attributes: 'polyline_idx', 'points' (array of intermediate points)
        """
        graph = nx.Graph()

        if not self.polylines:
            return graph

        # Detect branch points and endpoints
        topology = self.detect_branch_points(tolerance=tolerance)

        # Create a mapping from (poly_idx, point_idx) to node_id
        point_to_node = {}
        node_id = 0

        # Add branch point nodes
        for (poly_idx, point_idx), location in zip(
            topology["branch_points"], topology["branch_locations"]
        ):
            if (poly_idx, point_idx) not in point_to_node:
                # Check if this location already has a node (for grouped points)
                existing_node = None
                for nid, data in graph.nodes(data=True):
                    if np.linalg.norm(data["pos"] - location) < tolerance:
                        existing_node = nid
                        break

                if existing_node is not None:
                    point_to_node[(poly_idx, point_idx)] = existing_node
                else:
                    graph.add_node(node_id, pos=location, type="branch")
                    point_to_node[(poly_idx, point_idx)] = node_id
                    node_id += 1

        # Add endpoint nodes
        for (poly_idx, point_idx), location in zip(
            topology["endpoints"], topology["endpoint_locations"]
        ):
            if (poly_idx, point_idx) not in point_to_node:
                # Check if this location already has a node
                existing_node = None
                for nid, data in graph.nodes(data=True):
                    if np.linalg.norm(data["pos"] - location) < tolerance:
                        existing_node = nid
                        break

                if existing_node is not None:
                    point_to_node[(poly_idx, point_idx)] = existing_node
                else:
                    graph.add_node(node_id, pos=location, type="endpoint")
                    point_to_node[(poly_idx, point_idx)] = node_id
                    node_id += 1

        # Add edges for each polyline
        for poly_idx, pl in enumerate(self.polylines):
            if len(pl) < 2:
                continue

            # Get node IDs for start and end of this polyline
            start_key = (poly_idx, 0)
            end_key = (poly_idx, len(pl) - 1)

            if start_key in point_to_node and end_key in point_to_node:
                start_node = point_to_node[start_key]
                end_node = point_to_node[end_key]

                # Store intermediate points (excluding endpoints)
                intermediate = pl[1:-1] if len(pl) > 2 else np.array([])

                graph.add_edge(
                    start_node,
                    end_node,
                    polyline_idx=poly_idx,
                    points=intermediate,
                    length=self._compute_polyline_length(pl),
                )

        return graph

    def get_branch_point_indices(self, tolerance: float = 1e-6) -> Set[Tuple[int, int]]:
        """
        Get a set of (polyline_idx, point_idx) tuples for all branch points.

        This is useful for optimization to identify which points should not move
        or should move in a coordinated fashion.

        Args:
            tolerance: Distance threshold for considering points as coincident

        Returns:
            Set of (polyline_idx, point_idx) tuples identifying branch points
        """
        topology = self.detect_branch_points(tolerance=tolerance)
        return set(topology["branch_points"])

    def get_true_endpoint_indices(
        self, tolerance: float = 1e-6
    ) -> Set[Tuple[int, int]]:
        """
        Get a set of (polyline_idx, point_idx) tuples for true endpoints.

        True endpoints are polyline ends that don't connect to other polylines.

        Args:
            tolerance: Distance threshold for considering points as coincident

        Returns:
            Set of (polyline_idx, point_idx) tuples identifying true endpoints
        """
        topology = self.detect_branch_points(tolerance=tolerance)
        return set(topology["endpoints"])

    @staticmethod
    def _compute_polyline_length(pl: np.ndarray) -> float:
        """Compute the total arc length of a polyline."""
        if len(pl) < 2:
            return 0.0
        seg_lengths = np.linalg.norm(pl[1:] - pl[:-1], axis=1)
        return float(np.sum(seg_lengths))

    def prune_short_branches(
        self,
        min_length: Optional[float] = None,
        min_length_percentile: Optional[float] = None,
        tolerance: float = 1e-6,
        iterative: bool = True,
        verbose: bool = False,
    ) -> "PolylinesSkeleton":
        """
        Remove short branches from the skeleton.

        A branch is considered "short" if it's a terminal branch (connects to only
        one branch point) and its length is below the threshold. Branch points and
        internal branches are preserved.

        Args:
            min_length: Absolute minimum length threshold. Branches shorter than
                this will be removed. If None, use min_length_percentile.
            min_length_percentile: Percentile-based threshold (0-100). Branches
                shorter than this percentile of all branch lengths will be removed.
                Default: None (use min_length instead)
            tolerance: Distance threshold for detecting branch points
            iterative: If True, repeat pruning until no more branches can be removed.
                This handles cases where removing one branch causes another to become
                a short terminal branch. Default: True
            verbose: If True, print information about removed branches

        Returns:
            New PolylinesSkeleton with short branches removed

        Example:
            >>> # Remove branches shorter than 10 units
            >>> pruned = skeleton.prune_short_branches(min_length=10.0)
            >>> # Remove branches in the shortest 10%
            >>> pruned = skeleton.prune_short_branches(min_length_percentile=10)
        """
        if not self.polylines:
            return self.copy()

        # Determine threshold from original skeleton
        original_lengths = [self._compute_polyline_length(pl) for pl in self.polylines]

        if min_length is None and min_length_percentile is None:
            raise ValueError("Must specify either min_length or min_length_percentile")

        if min_length is not None:
            threshold = float(min_length)
        else:
            threshold = float(np.percentile(original_lengths, min_length_percentile))

        if verbose:
            logger.info("Pruning branches with length < %.4f", threshold)
            logger.info(
                "Original branch lengths: min=%.4f, max=%.4f, mean=%.4f",
                min(original_lengths),
                max(original_lengths),
                np.mean(original_lengths),
            )

        # Iteratively prune until no more branches are removed
        current = self.copy()
        total_removed = 0
        iteration = 0

        while True:
            iteration += 1

            # Compute lengths of current polylines
            lengths = [self._compute_polyline_length(pl) for pl in current.polylines]

            # Detect topology
            topology = current.detect_branch_points(tolerance=tolerance)
            branch_point_set = set(topology["branch_points"])
            endpoint_set = set(topology["endpoints"])

            # Identify which polylines to keep
            keep_indices = []
            removed_this_iteration = 0

            for poly_idx, (pl, length) in enumerate(zip(current.polylines, lengths)):
                if len(pl) < 2:
                    # Keep single-point polylines
                    keep_indices.append(poly_idx)
                    continue

                # Check if this is a terminal branch (one end is a true endpoint)
                start_key = (poly_idx, 0)
                end_key = (poly_idx, len(pl) - 1)

                start_is_endpoint = start_key in endpoint_set
                end_is_endpoint = end_key in endpoint_set

                is_terminal = start_is_endpoint or end_is_endpoint
                is_isolated = start_is_endpoint and end_is_endpoint

                # Remove if: (1) short terminal branch, or (2) isolated branch
                should_remove = False
                reason = ""

                if is_isolated:
                    # Isolated branch - not connected to anything
                    should_remove = True
                    reason = "isolated"
                elif is_terminal and length < threshold:
                    # Short terminal branch
                    should_remove = True
                    reason = f"short terminal (length={length:.4f})"

                if should_remove:
                    if verbose:
                        logger.info(
                            "Iteration %d: Removing polyline %d (%s)",
                            iteration,
                            poly_idx,
                            reason,
                        )
                    removed_this_iteration += 1
                else:
                    # Keep this branch
                    keep_indices.append(poly_idx)

            if removed_this_iteration == 0:
                # No more branches to remove
                break

            total_removed += removed_this_iteration

            # Create new skeleton with kept polylines
            new_polylines = [current.polylines[i].copy() for i in keep_indices]
            current = PolylinesSkeleton(new_polylines)

            if not iterative:
                # Only do one iteration
                break

        if verbose:
            logger.info(
                "Removed %d short branches total in %d iteration(s), kept %d branches",
                total_removed,
                iteration,
                len(current.polylines),
            )

        return current

    def prune_short_branches_inplace(
        self,
        min_length: Optional[float] = None,
        min_length_percentile: Optional[float] = None,
        tolerance: float = 1e-6,
        verbose: bool = False,
    ) -> int:
        """
        In-place version of prune_short_branches.

        Returns:
            Number of branches removed
        """
        original_count = len(self.polylines)
        pruned = self.prune_short_branches(
            min_length=min_length,
            min_length_percentile=min_length_percentile,
            tolerance=tolerance,
            verbose=verbose,
        )
        self.polylines = pruned.polylines
        return original_count - len(self.polylines)

    # ---------------------------------------------------------------------
    # Resampling
    # ---------------------------------------------------------------------
    def resample(self, spacing: float) -> "PolylinesSkeleton":
        """Return a new PolylinesSkeleton with polylines resampled by spacing.

        Includes endpoints; inserts intermediate points every ~`spacing` units
        along arclength.
        """
        out: List[np.ndarray] = []
        for pl in self.polylines:
            out.append(_resample_polyline(pl, float(spacing)))
        return PolylinesSkeleton(out)

    def resample_inplace(self, spacing: float) -> None:
        """In-place version of `resample`.

        Modifies `self.polylines` by resampling each polyline with the given spacing.
        """
        self.polylines = [
            _resample_polyline(pl, float(spacing)) for pl in self.polylines
        ]

    # ---------------------------------------------------------------------
    # Projection / snapping to mesh surface
    # ---------------------------------------------------------------------
    def snap_to_mesh_surface(
        self,
        mesh: trimesh.Trimesh,
        *,
        project_outside_only: bool = True,
        max_distance: Optional[float] = None,
    ) -> Tuple[int, float]:
        """
        Project points to the nearest surface point on `mesh`.

        Args:
            project_outside_only: If True, use signed distance to project only
                points outside (positive sign). If signed distance is unavailable,
                falls back to projecting all points.
            max_distance: If provided, only move points whose distance to the
                surface exceeds this threshold. Units are mesh units.

        Returns:
            (n_moved, mean_move_distance)
        """
        if mesh is None or len(getattr(mesh, "vertices", [])) == 0:
            return 0, 0.0

        all_pts = [pl for pl in self.polylines if pl.size > 0]
        if not all_pts:
            return 0, 0.0

        # Flatten to a single array and track mapping back
        counts = [pl.shape[0] for pl in self.polylines]
        P = (
            np.vstack(self.polylines)
            if self.polylines
            else np.zeros((0, 3), dtype=float)
        )

        # Try signed distance to detect outside points
        use_mask = None
        if project_outside_only:
            try:
                from trimesh.proximity import signed_distance  # type: ignore

                d = signed_distance(mesh, P)
                use_mask = d > 0  # outside
            except Exception:
                use_mask = None

        try:
            from trimesh.proximity import closest_point  # type: ignore

            CP, dist, tri = closest_point(mesh, P)
            _ = tri  # unused
        except Exception:
            # Fallback: KDTree over vertices only
            V = np.asarray(mesh.vertices, dtype=float)
            if V.size == 0:
                return 0, 0.0
            from scipy.spatial import cKDTree  # type: ignore

            tree = cKDTree(V)
            dd, idx = tree.query(P, k=1)
            CP = V[idx]
            dist = dd

        if use_mask is None:
            mask = np.ones(P.shape[0], dtype=bool)
        else:
            mask = use_mask

        if max_distance is not None:
            mask = mask & (dist >= float(max_distance))

        moved = 0
        total_move = 0.0
        # Re-split CP back to polylines and update selectively
        start = 0
        new_polylines: List[np.ndarray] = []
        for cnt in counts:
            segP = P[start : start + cnt]
            segCP = CP[start : start + cnt]
            segDist = dist[start : start + cnt]
            segMask = mask[start : start + cnt]
            start += cnt
            out = segP.copy()
            sel = np.nonzero(segMask)[0]
            if len(sel) > 0:
                out[sel] = segCP[sel]
                moved += int(len(sel))
                total_move += float(np.sum(segDist[sel]))
            new_polylines.append(out)
        self.polylines = new_polylines
        mean_move = (total_move / moved) if moved > 0 else 0.0
        return moved, mean_move

    # ---------------------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------------------
    def visualize_3d(
        self,
        title: str = "3D Polylines",
        color: str = "crimson",
        *,
        backend: str = "auto",
        show_axes: bool = True,
        width: int = 800,
        height: int = 600,
        line_width: float = 3.0,
        opacity: float = 0.95,
    ) -> Optional[object]:
        """Visualize the polylines in 3D.

        Args:
            title: Figure title
            color: Line color for polylines
            backend: 'plotly', 'matplotlib', or 'auto'
            show_axes: Whether to display axes
            width: Figure width (pixels for plotly)
            height: Figure height (pixels for plotly)
            line_width: Line width for polylines
            opacity: Opacity for plotly lines

        Returns:
            Backend-specific figure/axes object or None if backend unavailable.
        """
        # Determine backend if auto
        if backend == "auto":
            try:
                import plotly.graph_objects as go  # noqa: F401

                backend = "plotly"
            except Exception:
                try:
                    import matplotlib.pyplot as plt  # noqa: F401

                    backend = "matplotlib"
                except Exception:
                    backend = "plotly"

        if backend == "plotly":
            try:
                import plotly.graph_objects as go

                fig = go.Figure()
                # Add a trace per polyline
                for idx, pl in enumerate(self.polylines):
                    if pl.size == 0:
                        continue
                    fig.add_trace(
                        go.Scatter3d(
                            x=pl[:, 0],
                            y=pl[:, 1],
                            z=pl[:, 2],
                            mode="lines",
                            line=dict(color=color, width=float(line_width)),
                            opacity=float(opacity),
                            name=f"Polyline {idx}",
                        )
                    )

                fig.update_layout(
                    title=title,
                    autosize=False,
                    width=int(width),
                    height=int(height),
                    scene=dict(
                        aspectmode="data",
                        xaxis=dict(visible=show_axes),
                        yaxis=dict(visible=show_axes),
                        zaxis=dict(visible=show_axes),
                    ),
                    showlegend=False,
                )
                return fig
            except Exception as e:
                logger.warning("Plotly visualization failed: %s", e)
                return None

        if backend == "matplotlib":
            try:
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

                fig = plt.figure(figsize=(max(4, width / 100), max(3, height / 100)))
                ax = fig.add_subplot(111, projection="3d")

                # Plot each polyline
                for pl in self.polylines:
                    if pl.size == 0:
                        continue
                    ax.plot(
                        pl[:, 0],
                        pl[:, 1],
                        pl[:, 2],
                        color=color,
                        linewidth=float(line_width),
                    )

                # Axis labels and bounds
                try:
                    if self.polylines:
                        P = np.vstack(self.polylines)
                        ax.set_xlim(P[:, 0].min(), P[:, 0].max())
                        ax.set_ylim(P[:, 1].min(), P[:, 1].max())
                        ax.set_zlim(P[:, 2].min(), P[:, 2].max())
                except Exception:
                    pass

                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                ax.set_title(title)
                if not show_axes:
                    ax.set_axis_off()
                fig.tight_layout()
                return fig
            except Exception as e:
                logger.warning("Matplotlib visualization failed: %s", e)
                return None

        raise ValueError(f"Unknown backend: {backend}")


# ---------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------
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

import logging
import math
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import shapely.geometry as sgeom
import trimesh

from .mesh import MeshManager
from .polylines import PolylinesSkeleton

# Module-level logger
logger = logging.getLogger(__name__)
# Ensure no "No handler" warnings in library usage; applications can configure handlers.
logger.addHandler(logging.NullHandler())

# ============================================================================
# Data models
# ============================================================================


@dataclass
class Junction:
    """
    A disk fitted to a closed cross-section area at a cut plane (constant z).

    Attributes:
        id: Unique identifier across all junctions.
        z: Plane z-value where this junction lies.
        center: 3D center (x, y, z) of the fitted disk.
        radius: Radius of the fitted disk (default: equivalent-area circle).
        area: Area of the cross-section polygon this disk represents.
        slice_index: Index of the slice interval below this cut (0..n_slices-1).
        cross_section_index: Index of the polygon within the cut's cross-section.
    """

    id: int
    z: float
    center: np.ndarray
    radius: float
    area: float
    slice_index: int
    cross_section_index: int


@dataclass
class CrossSection:
    """Cross-section at a cut plane with potentially multiple closed polygons."""

    z: float
    slice_index: int
    polygons: List[sgeom.Polygon]
    junction_ids: List[int]


@dataclass
class Segment:
    """
    A 3D segment (band component) between two adjacent cuts.

    Each segment induces one or more edges in the skeleton graph, connecting
    junctions at the lower cut to junctions at the upper cut.
    """

    id: int
    slice_index: int
    z_lower: float
    z_upper: float
    volume: float
    surface_area: float
    lower_junction_ids: List[int]
    upper_junction_ids: List[int]


class SkeletonGraph:
    """
    Graph where nodes are `Junction`s and edges correspond to cylinder-like
    segments connecting nodes on adjacent cuts only.
    """

    def __init__(self):
        self.G = nx.Graph()
        self.junctions: Dict[int, Junction] = {}
        self.cross_sections: List[CrossSection] = []
        self.segments: List[Segment] = []
        # Snapshot of transforms applied to the mesh (for selective undo at export)
        # Each entry: {"name": str, "M": np.ndarray(4x4), "is_uniform_scale": bool, "uniform_scale": Optional[float]}
        self.transforms_applied: List[Dict[str, Any]] = []

    # ----------------------------- Node/edge API -----------------------------
    def add_junction(self, j: Junction):
        self.junctions[j.id] = j
        self.G.add_node(
            j.id,
            kind="junction",
            z=float(j.z),
            center=j.center.astype(float),
            radius=float(j.radius),
            area=float(j.area),
            slice_index=int(j.slice_index),
            cross_section_index=int(j.cross_section_index),
        )

    def add_segment_edges(self, seg: Segment):
        self.segments.append(seg)
        # Connect all lower to all upper (supports branching). Adjacent slices only.
        # To guarantee no self-connections and strict adjacency, only connect
        # junctions that lie on this band's lower/upper z-planes respectively.
        tol = max(1e-9, 1e-6 * abs(float(seg.z_upper) - float(seg.z_lower)))
        for jl in seg.lower_junction_ids:
            for ju in seg.upper_junction_ids:
                if jl == ju:
                    # Never allow self-connections
                    continue
                # Fetch z of each junction node
                try:
                    zl = float(self.G.nodes[jl].get("z", float("nan")))
                    zu = float(self.G.nodes[ju].get("z", float("nan")))
                except Exception:
                    continue
                # Require nodes to correspond to the band's bounding planes
                if not (
                    abs(zl - float(seg.z_lower)) <= tol
                    and abs(zu - float(seg.z_upper)) <= tol
                ):
                    continue
                # Add edge only if not present
                if not self.G.has_edge(jl, ju):
                    # Derive polyline support by intersecting polyline ids
                    def _ids_for(nid: int) -> List[int]:
                        try:
                            hints = self.G.nodes[nid].get("polyline_hints", [])
                            return [
                                int(h.get("polyline_id", -1))
                                for h in hints
                                if h.get("polyline_id") is not None
                            ]
                        except Exception:
                            return []

                    ids_l = set(_ids_for(jl))
                    ids_u = set(_ids_for(ju))
                    support = (
                        sorted(list(ids_l.intersection(ids_u)))
                        if ids_l and ids_u
                        else []
                    )

                    self.G.add_edge(
                        jl,
                        ju,
                        kind="segment",
                        segment_id=int(seg.id),
                        slice_index=int(seg.slice_index),
                        z_lower=float(seg.z_lower),
                        z_upper=float(seg.z_upper),
                        volume=float(seg.volume),
                        surface_area=float(seg.surface_area),
                        polyline_support=support,
                        chosen_by="unguided",
                    )

    def to_networkx(self) -> nx.Graph:
        return self.G

    # visualization
    def draw(
        self,
        axis: str = "x",
        ax: Any = None,
        figsize: Optional[Tuple[float, float]] = None,
        with_labels: bool = False,
        node_size: int = 50,
        node_color: str = "C0",
        edge_color: str = "0.6",
        min_hv_ratio: float = 0.3,
        pad_frac: float = 0.05,
        **kwargs: Any,
    ) -> Any:
        """Draw the skeleton graph in 2D using (x,z) or (y,z) coordinates.

        Args:
            axis: Horizontal axis to use ("x" or "y"). Vertical axis is always z.
            ax: Optional matplotlib Axes to draw into. If None, a new figure/axes is created.
            figsize: Optional (width, height) in inches when creating a new figure.
            with_labels: Whether to render node labels.
            node_size: Node marker size passed to networkx.draw.
            node_color: Node color.
            edge_color: Edge color.
            min_hv_ratio: Minimum desired horizontal-to-vertical data range ratio.
                If the horizontal data span is less than this ratio times the
                vertical (z) span, x/y limits are expanded around the mean to
                meet this minimum. Helps avoid an overly thin vertical line.
            pad_frac: Fractional padding added to both axes limits for readability.
            **kwargs: Additional kwargs forwarded to networkx.draw.

        Returns:
            The matplotlib Axes used for drawing.
        """
        axis = axis.lower()
        if axis not in ("x", "y"):
            raise ValueError("axis must be 'x' or 'y'")

        # Build 2D positions from node 3D centers
        idx = 0 if axis == "x" else 1
        pos: Dict[Union[int, str], Tuple[float, float]] = {}
        for n, attrs in self.G.nodes(data=True):
            c = attrs.get("center")
            if c is None or len(c) != 3:
                continue
            pos[n] = (float(c[idx]), float(c[2]))  # (x|y, z)

        # Compute data bounds
        if len(pos) == 0:
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            ax.set_xlabel(f"{axis} (horizontal)")
            ax.set_ylabel("z (vertical)")
            return ax

        xs = np.array([p[0] for p in pos.values()], dtype=float)
        zs = np.array([p[1] for p in pos.values()], dtype=float)
        xmin, xmax = float(xs.min()), float(xs.max())
        zmin, zmax = float(zs.min()), float(zs.max())
        xmid = 0.5 * (xmin + xmax)
        zmid = 0.5 * (zmin + zmax)
        xr = max(xmax - xmin, 1e-12)
        zr = max(zmax - zmin, 1e-12)

        # Enforce a minimum horizontal span relative to vertical span
        min_xr = max(min_hv_ratio * zr, xr)
        if min_xr > xr:
            xmin = xmid - 0.5 * min_xr
            xmax = xmid + 0.5 * min_xr
            xr = min_xr

        # Apply padding
        xmin -= pad_frac * xr
        xmax += pad_frac * xr
        zmin -= pad_frac * zr
        zmax += pad_frac * zr

        # Prepare axes
        if ax is None:
            # Auto figsize that respects data aspect for clarity
            if figsize is None:
                # Base height and width with reasonable minimums
                base_h = 4.0
                # width scaled by data aspect (horizontal over vertical)
                aspect = xr / zr if zr > 0 else 1.0
                base_w = max(4.0, base_h * max(aspect, min_hv_ratio))
                figsize = (base_w, base_h)
            fig, ax = plt.subplots(figsize=figsize)

        # Draw using networkx helper
        nx.draw(
            self.G,
            pos=pos,
            ax=ax,
            with_labels=with_labels,
            node_size=node_size,
            node_color=node_color,
            edge_color=edge_color,
            **kwargs,
        )

        ax.set_xlabel(f"{axis} (horizontal)")
        ax.set_ylabel("z (vertical)")
        # Set limits and aspect for a clear, readable plot
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(zmin, zmax)
        try:
            ax.set_aspect("equal", adjustable="box")
        except Exception:
            # Fallback to auto if backend cannot honor equal aspect
            ax.set_aspect("auto")

        # Return provided or newly-created axes
        return ax

    def plot_cross_section(
        self,
        node_id: Union[int, str],
        ax: Any = None,
        boundary_color: str = "k",
        circle_color: str = "C1",
        center_color: str = "C2",
        linewidth: float = 1.5,
        alpha_boundary: float = 0.9,
        alpha_circle: float = 0.8,
        show_center: bool = True,
        title: Optional[str] = None,
    ) -> Any:
        """Plot the cross-section boundary curve with its fitted disk.

        Expects node attributes `boundary_2d`, `center`, and `radius`.

        Args:
            node_id: Graph node identifier (e.g., "cut3_cs0").
            ax: Optional matplotlib Axes; if None, a new one is created.
            boundary_color: Color for the boundary polyline.
            circle_color: Color for the fitted circle.
            center_color: Color for the center marker.
            linewidth: Line width for boundary and circle.
            alpha_boundary: Alpha for the boundary.
            alpha_circle: Alpha for the circle.
            show_center: Whether to draw the center point.
            title: Optional title; defaults to node id with cut/index.

        Returns:
            The matplotlib Axes used for drawing.
        """
        if node_id not in self.G:
            raise KeyError(f"Node '{node_id}' not in SkeletonGraph")

        attrs = self.G.nodes[node_id]
        boundary_2d = attrs.get("boundary_2d")
        center = attrs.get("center")
        radius = attrs.get("radius")

        if boundary_2d is None or len(boundary_2d) < 2:
            logger.warning(
                "Node '%s' has no boundary_2d; cannot plot boundary.", node_id
            )
        if center is None or radius is None:
            raise ValueError(
                f"Node '{node_id}' missing center/radius required for plotting"
            )

        # Prepare axes
        if ax is None:
            fig, ax = plt.subplots()

        # Plot boundary polyline
        if boundary_2d is not None and len(boundary_2d) >= 2:
            bx = boundary_2d[:, 0]
            by = boundary_2d[:, 1]
            ax.plot(bx, by, color=boundary_color, lw=linewidth, alpha=alpha_boundary)

        # Plot fitted circle
        cx, cy = float(center[0]), float(center[1])
        r = float(radius)
        theta = np.linspace(0.0, 2.0 * math.pi, 256)
        ax.plot(
            cx + r * np.cos(theta),
            cy + r * np.sin(theta),
            color=circle_color,
            lw=linewidth,
            alpha=alpha_circle,
        )
        if show_center:
            ax.plot([cx], [cy], marker="o", color=center_color, ms=4)

        # Cosmetics
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        default_title = title
        if default_title is None:
            ci = attrs.get("slice_index")
            ii = attrs.get("cross_section_index")
            default_title = f"{node_id} (slice {ci}, cs {ii})"
        ax.set_title(default_title)

        return ax

    def plot_all_cross_sections(
        self,
        sort_by: Tuple[str, str] = ("slice_index", "cross_section_index"),
        max_cols: int = 4,
        figsize: Optional[Tuple[float, float]] = None,
        share_axes: bool = False,
        boundary_color: str = "k",
        circle_color: str = "C1",
    ) -> Any:
        """Plot every cross-section boundary with its fitted disk in a grid.

        Args:
            sort_by: Tuple of node attribute names to sort nodes (primary, secondary).
            max_cols: Max number of subplot columns.
            figsize: Figure size; auto if None.
            share_axes: Whether to share x/y among subplots.
            boundary_color: Color for boundaries.
            circle_color: Color for circles.

        Returns:
            The matplotlib Figure created.
        """
        # Gather nodes that have boundary_2d and required attributes
        nodes = []
        for nid, attrs in self.G.nodes(data=True):
            if attrs.get("center") is None or attrs.get("radius") is None:
                continue
            nodes.append((nid, attrs))

        # Sort nodes by provided attributes if present
        def sort_key(item: Tuple[str, Dict[str, Any]]):
            _, a = item
            return tuple(a.get(k, 0) for k in sort_by)

        nodes.sort(key=sort_key)
        n = len(nodes)
        if n == 0:
            logger.info("No nodes available to plot")
            return None

        ncols = max(1, int(max_cols))
        nrows = int(math.ceil(n / ncols))
        if figsize is None:
            figsize = (4 * ncols, 4 * nrows)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=figsize,
            squeeze=False,
            sharex=share_axes,
            sharey=share_axes,
        )

        for idx, (nid, attrs) in enumerate(nodes):
            r = idx // ncols
            c = idx % ncols
            ax = axes[r][c]
            # Determine z from node attributes; fall back to center[2]
            z_val = attrs.get("z")
            if z_val is None:
                ctr = attrs.get("center")
                if ctr is not None and len(ctr) == 3:
                    z_val = float(ctr[2])
            # Build a title that includes slice, cross-section index, and z-level
            slice_idx = attrs.get("slice_index")
            cs_idx = attrs.get("cross_section_index")
            title = (
                f"{nid} (slice {slice_idx}, cs {cs_idx}, z={z_val:.3f})"
                if z_val is not None
                else f"{nid} (slice {slice_idx}, cs {cs_idx})"
            )
            self.plot_cross_section(
                nid,
                ax=ax,
                boundary_color=boundary_color,
                circle_color=circle_color,
                show_center=True,
                title=title,
            )

        # Hide any unused subplots
        for k in range(n, nrows * ncols):
            r = k // ncols
            c = k % ncols
            axes[r][c].axis("off")

        fig.tight_layout()
        return fig

    def to_swc(
        self,
        filepath: str,
        *,
        type_index: int = 3,
        annotate_cycles: bool = True,
        cycle_mode: str = "duplicate_junction",
        undo_transforms: Optional[List[str]] = None,
        force_single_tree: bool = True,
    ) -> None:
        """Write this skeleton graph to an SWC file.

        SWC format: each row has 7 columns: ``n T x y z R p``
          - ``n``: 1-based sample index
          - ``T``: sample type (default 5, overridable via ``type_index``)
          - ``x,y,z``: coordinates (from node ``center``)
          - ``R``: radius (from node ``radius``)
          - ``p``: parent index (or -1 for root)

        Important: SWC requires a tree (no cycles). If this graph contains
        cycles, we can break them in one of two ways (selectable via
        ``cycle_mode``):
          - ``"remove_edge"`` (default): remove one edge per cycle to obtain a
            spanning forest. Prefer removing an edge touching a branching node
            (degree > 2) to preserve simpler chains. Removed edges are
            annotated in the SWC header.
          - ``"duplicate_junction"``: duplicate a branching node within the
            cycle and rewire one incident cycle edge to the duplicate instead
            of removing it. This preserves all segments as edges in SWC while
            yielding a tree. Duplications are annotated in the SWC header so
            downstream tools can reconnect duplicate pairs if needed.

        Args:
            filepath: Destination SWC file path.
            type_index: Type code to write in column T (default 5). You can
                change this to match your downstream tooling.
            annotate_cycles: If True, write header lines describing the cycle
                breaking operations (removed edges or duplicated junctions).
            cycle_mode: "remove_edge" (default) or "duplicate_junction".
                When "duplicate_junction", a branching node on the cycle is
                duplicated and one of its incident cycle edges is rewired to
                the duplicate.
            force_single_tree: If True, after cycle-breaking, connect any
                remaining disconnected components into a single tree by adding
                minimal bridging edges between components (nearest center pairs).
                Bridges are annotated in the SWC header.
        """
        # Validate type index
        try:
            t_code = int(type_index)
        except Exception as e:
            raise ValueError("type_index must be an integer") from e

        G = self.G

        # Build optional undo transform matrix for centers and radius scale factor
        M_undo: Optional[np.ndarray] = None
        radius_scale_factor: float = 1.0
        if undo_transforms:
            # Compose selected transforms in order applied, then invert
            selected: List[np.ndarray] = []
            scale_prod: float = 1.0
            for t in self.transforms_applied:
                name = t.get("name")
                if name in undo_transforms:
                    M = np.asarray(t.get("M"), dtype=float)
                    if M.shape == (4, 4):
                        selected.append(M)
                    if t.get("is_uniform_scale") and t.get("uniform_scale") is not None:
                        try:
                            scale_prod *= float(t.get("uniform_scale"))
                        except Exception:
                            pass
            if selected:
                M_comp = np.eye(4, dtype=float)
                for M in selected:
                    M_comp = M @ M_comp
                try:
                    M_undo = np.linalg.inv(M_comp)
                except Exception:
                    M_undo = None
            # Undo scaling on radii if we have a valid scale product
            if np.isfinite(scale_prod) and scale_prod != 0.0:
                radius_scale_factor = 1.0 / float(scale_prod)
            else:
                radius_scale_factor = 1.0
        if G.number_of_nodes() == 0:
            # Write a minimal header file
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("# Empty SWC exported by mcf2swc SkeletonGraph.to_swc\n")
                f.write(f"# Date: {datetime.now().isoformat()}\n")
                f.write("# Columns: n T x y z R p\n")
                f.write(f"# Type index (T) used for all samples: {t_code}\n")
            return

        # Work on a copy when breaking cycles
        H = nx.Graph()
        H.add_nodes_from(G.nodes)
        H.add_edges_from(G.edges)

        removed_edges: List[Dict[str, Any]] = []
        duplicated_nodes: List[Dict[str, Any]] = []
        component_bridges: List[Dict[str, Any]] = []
        # Attributes for synthetic duplicate nodes (not present in self.G)
        extra_node_attrs: Dict[Any, Dict[str, Any]] = {}

        # Helper for generating unique integer IDs for duplicates
        max_int_id = -1
        try:
            int_nodes = [int(n) for n in G.nodes if isinstance(n, int)]
            if int_nodes:
                max_int_id = max(int_nodes)
        except Exception:
            pass
        next_new_id = max_int_id + 1

        # Iteratively break cycles per selected mode
        while True:
            try:
                cycles = nx.cycle_basis(H)
            except Exception:
                cycles = []
            if not cycles:
                break

            cyc = cycles[0]
            m = len(cyc)

            if cycle_mode == "duplicate_junction":
                # Find a branching node in the cycle (pref degree>2 in original G)
                b_idx: Optional[int] = None
                for i in range(m):
                    if G.degree[cyc[i]] > 2:
                        b_idx = i
                        break
                if b_idx is None:
                    # Fall back to highest-degree node in the cycle
                    b_idx = max(range(m), key=lambda i: G.degree[cyc[i]])
                b = cyc[b_idx]
                prev_n = cyc[(b_idx - 1) % m]

                # Create a duplicate node with identical attributes
                dup_id: Any = next_new_id
                next_new_id += 1
                # Copy essential attributes from G
                b_attrs = G.nodes[b]
                extra_node_attrs[dup_id] = {
                    "kind": b_attrs.get("kind", "junction"),
                    "z": float(b_attrs.get("z", 0.0)),
                    "center": (
                        np.array(b_attrs.get("center"), dtype=float)
                        if b_attrs.get("center") is not None
                        else None
                    ),
                    "radius": float(b_attrs.get("radius", 0.0)),
                    "area": float(b_attrs.get("area", 0.0)),
                    "slice_index": int(b_attrs.get("slice_index", 0)),
                    "cross_section_index": int(b_attrs.get("cross_section_index", 0)),
                }

                # Rewire one incident cycle edge: (prev_n, b) -> (prev_n, dup_id)
                if H.has_edge(prev_n, b):
                    # Capture edge attributes for annotation from original G
                    eattrs = G.get_edge_data(prev_n, b, default={})
                    if H.has_edge(prev_n, b):
                        H.remove_edge(prev_n, b)
                    H.add_node(dup_id)
                    H.add_edge(prev_n, dup_id)
                    duplicated_nodes.append(
                        {
                            "orig": b,
                            "duplicate": dup_id,
                            "redirected_from": prev_n,
                            **({} if eattrs is None else eattrs),
                        }
                    )
                else:
                    # If the expected edge isn't present (unexpected), fall back to removal
                    u = cyc[0]
                    v = cyc[1 % m]
                    if H.has_edge(u, v):
                        eattrs = G.get_edge_data(u, v, default={})
                        removed_edges.append(
                            {"u": u, "v": v, **({} if eattrs is None else eattrs)}
                        )
                        H.remove_edge(u, v)
            else:
                # Default/remove_edge mode: remove one edge touching a branching node
                chosen: Optional[Tuple[Any, Any]] = None
                for i in range(m):
                    u = cyc[i]
                    v = cyc[(i + 1) % m]
                    if G.degree[u] > 2 or G.degree[v] > 2:
                        chosen = (u, v)
                        break
                if chosen is None:
                    chosen = (cyc[0], cyc[1 % m])

                u, v = chosen
                if H.has_edge(u, v):
                    eattrs = G.get_edge_data(u, v, default={})
                    removed_edges.append(
                        {"u": u, "v": v, **({} if eattrs is None else eattrs)}
                    )
                    H.remove_edge(u, v)

        # Helper to transform a center if undo matrix is requested
        def _maybe_undo_center(center: np.ndarray) -> np.ndarray:
            if M_undo is None:
                return center
            try:
                c = np.asarray(center, dtype=float)
                ch = np.array([c[0], c[1], c[2], 1.0], dtype=float)
                cu = M_undo @ ch
                return np.array([float(cu[0]), float(cu[1]), float(cu[2])], dtype=float)
            except Exception:
                return center

        # Helper to get z-value for ordering and root selection
        def _node_z(n: Any) -> float:
            # Prefer attributes from original graph; fall back to duplicates
            if n in G.nodes:
                attrs = G.nodes[n]
            else:
                attrs = extra_node_attrs.get(n, {})
            z = attrs.get("z")
            if z is None:
                c = attrs.get("center")
                if c is not None and len(c) == 3:
                    c3 = (
                        _maybe_undo_center(np.array(c, dtype=float))
                        if M_undo is not None
                        else np.array(c, dtype=float)
                    )
                    return float(c3[2])
                return 0.0
            return float(z)

        # Build a BFS forest over each connected component to define parents
        components = list(nx.connected_components(H))
        # Sort components by their lowest z to make output stable
        components.sort(key=lambda comp: min(_node_z(n) for n in comp))
        n_components_initial = len(components)

        def _center3(n: Any) -> Optional[np.ndarray]:
            attrs = G.nodes[n] if n in G.nodes else extra_node_attrs.get(n, {})
            c = attrs.get("center")
            if c is None:
                return None
            try:
                cc = np.asarray(c, dtype=float)
                return _maybe_undo_center(cc) if M_undo is not None else cc
            except Exception:
                return None

        if force_single_tree and n_components_initial > 1:
            # Use the first (lowest-z) component as base; connect others to it
            base_nodes = set(components[0])
            for comp in components[1:]:
                best_pair: Optional[Tuple[Any, Any]] = None
                best_d = float("inf")
                for u in base_nodes:
                    cu = _center3(u)
                    if cu is None:
                        continue
                    for v in comp:
                        cv = _center3(v)
                        if cv is None:
                            continue
                        d = float(np.linalg.norm(cu - cv))
                        if d < best_d:
                            best_d = d
                            best_pair = (u, v)
                if best_pair is not None:
                    u, v = best_pair
                    if not H.has_edge(u, v):
                        H.add_edge(u, v)
                    component_bridges.append({"u": u, "v": v, "distance": best_d})
                    base_nodes |= set(comp)
            # Recompute components after bridging
            components = list(nx.connected_components(H))
            components.sort(key=lambda comp: min(_node_z(n) for n in comp))

        swc_index: Dict[Any, int] = {}
        rows: List[Tuple[int, int, float, float, float, float, int]] = []
        idx_counter = 1

        for comp in components:
            if not comp:
                continue
            root = min(comp, key=_node_z)
            # Directed BFS tree rooted at root
            T = nx.bfs_tree(H, root)
            predecessors = {
                child: parent for child, parent in nx.bfs_predecessors(H, root)
            }

            # Iterate nodes in a parent-before-children order
            order = list(nx.topological_sort(T))
            for n in order:
                # Prefer attributes from original graph; fall back to duplicate attrs
                attrs = G.nodes[n] if n in G.nodes else extra_node_attrs.get(n, {})
                c = attrs.get("center")
                r = attrs.get("radius", 0.0)
                if c is None or len(c) != 3:
                    raise ValueError(
                        f"Node {n!r} is missing a 3D center; cannot export to SWC"
                    )

                # Apply optional undo to center and radius
                c_use = (
                    _maybe_undo_center(np.array(c, dtype=float))
                    if M_undo is not None
                    else np.array(c, dtype=float)
                )
                x, y, z = float(c_use[0]), float(c_use[1]), float(c_use[2])
                r_use = float(r) * radius_scale_factor

                if n == root:
                    parent_idx = -1
                else:
                    p = predecessors.get(n)
                    # If predecessor is missing for some reason, fall back to root
                    if p is None:
                        p = root
                    parent_idx = swc_index.get(p, -1)

                swc_index[n] = idx_counter
                rows.append(
                    (idx_counter, t_code, x, y, z, float(r_use), int(parent_idx))
                )
                idx_counter += 1

        # Write the SWC file with headers and annotations
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("# SWC exported by mcf2swc SkeletonGraph.to_swc\n")
            f.write(f"# Date: {datetime.now().isoformat()}\n")
            f.write("# Columns: n T x y z R p\n")
            f.write(f"# Type index (T) used for all samples: {t_code}\n")
            if not force_single_tree and len(components) > 1:
                f.write(
                    f"# Note: {len(components)} disconnected components were serialized; each begins with a root (parent -1).\n"
                )
            if force_single_tree and n_components_initial > 1:
                f.write(
                    f"# Bridged {n_components_initial - 1} disconnected component(s) into a single SWC tree.\n"
                )
            if annotate_cycles:
                if removed_edges:
                    f.write(
                        f"# Removed {len(removed_edges)} edge(s) to break cycles for SWC format.\n"
                    )
                    for info in removed_edges:
                        u = info.get("u")
                        v = info.get("v")
                        su = swc_index.get(u)
                        sv = swc_index.get(v)
                        seg_id = info.get("segment_id")
                        slice_idx = info.get("slice_index")
                        z_low = info.get("z_lower")
                        z_up = info.get("z_upper")
                        f.write(
                            "# cycle_edge_removed "
                            f"u_jid={u} v_jid={v} u_swc={su} v_swc={sv} "
                            f"segment_id={seg_id} slice_index={slice_idx} z_lower={z_low} z_upper={z_up}\n"
                        )
                if duplicated_nodes:
                    f.write(
                        f"# Duplicated {len(duplicated_nodes)} junction node(s) to break cycles for SWC format.\n"
                    )
                    for info in duplicated_nodes:
                        orig = info.get("orig")
                        dup = info.get("duplicate")
                        redirected_from = info.get("redirected_from")
                        s_orig = swc_index.get(orig)
                        s_dup = swc_index.get(dup)
                        s_from = swc_index.get(redirected_from)
                        seg_id = info.get("segment_id")
                        slice_idx = info.get("slice_index")
                        z_low = info.get("z_lower")
                        z_up = info.get("z_upper")
                        f.write(
                            "# cycle_node_duplicated "
                            f"orig_jid={orig} dup_id={dup} from_jid={redirected_from} "
                            f"orig_swc={s_orig} dup_swc={s_dup} from_swc={s_from} "
                            f"segment_id={seg_id} slice_index={slice_idx} z_lower={z_low} z_upper={z_up}\n"
                        )
                if force_single_tree and component_bridges:
                    f.write(
                        f"# Added {len(component_bridges)} component bridge(s) to ensure a single SWC tree.\n"
                    )
                    for info in component_bridges:
                        u = info.get("u")
                        v = info.get("v")
                        su = swc_index.get(u)
                        sv = swc_index.get(v)
                        dist = info.get("distance")
                        f.write(
                            "# component_bridge "
                            f"u_jid={u} v_jid={v} u_swc={su} v_swc={sv} distance={dist}\n"
                        )

            for n, T_code, x, y, z, R, parent in rows:
                f.write(f"{n} {T_code} {x:.6f} {y:.6f} {z:.6f} {R:.6f} {parent}\n")


# ============================================================================
# Core algorithm
# ============================================================================


def skeletonize(
    mesh_or_manager: Union[trimesh.Trimesh, MeshManager],
    n_slices: int,
    *,
    radius_mode: str = "equivalent_area",
    validate_volume: bool = True,
    volume_tol: float = 0.05,
    verbose: bool = False,
    verbosity: Optional[int] = None,
    enforce_connected: bool = True,
    connect_isolated_terminals: bool = True,
    polylines: Optional[PolylinesSkeleton] = None,
) -> SkeletonGraph:
    """
    Build a `SkeletonGraph` by slicing the mesh along z into `n_slices` bands.

    Steps:
      1) Validate input mesh (watertight, single component)
      2) Create cut planes (n_cuts = n_slices-1) uniformly in z
      3) For each cut, compute cross-section polygons and fit junctions
      4) For each band between adjacent cuts, compute connected components
         and connect lower/upper junctions according to component membership
      5) Optionally validate volume conservation across all bands

    Returns:
        SkeletonGraph
    """

    # Backward-compatibility mapping for verbosity levels
    if verbosity is None:
        eff_verbosity = 2 if verbose else 0
    else:
        try:
            eff_verbosity = int(verbosity)
        except Exception:
            eff_verbosity = 0
        # If both are provided, verbosity takes precedence

    def v_info(msg: str, *args: Any) -> None:
        if eff_verbosity >= 1:
            logger.info(msg, *args)

    def v_debug(msg: str, *args: Any) -> None:
        if eff_verbosity >= 2:
            logger.debug(msg, *args)

    mm = (
        mesh_or_manager
        if isinstance(mesh_or_manager, MeshManager)
        else MeshManager(mesh_or_manager)
    )
    mesh = mm.to_trimesh()

    # Snapshot transforms applied so far on the MeshManager into the SkeletonGraph later
    # by storing them into a temporary list until skel is created.
    _applied_transforms: List[Dict[str, Any]] = []
    try:
        if isinstance(mesh_or_manager, MeshManager) and hasattr(mm, "transform_stack"):
            for t in getattr(mm, "transform_stack", []):
                try:
                    _applied_transforms.append(
                        {
                            "name": getattr(t, "name", None),
                            "M": np.array(getattr(t, "M", np.eye(4)), dtype=float),
                            "is_uniform_scale": bool(
                                getattr(t, "is_uniform_scale", False)
                            ),
                            "uniform_scale": (
                                float(getattr(t, "uniform_scale", 1.0))
                                if getattr(t, "uniform_scale", None) is not None
                                else None
                            ),
                        }
                    )
                except Exception:
                    continue
    except Exception:
        _applied_transforms = []

    if mesh is None:
        raise ValueError("No mesh provided")

    # ------------------------------ Validation ------------------------------
    if not mesh.is_watertight:
        raise ValueError("Input mesh must be watertight")

    try:
        comps = mesh.split(only_watertight=False)
        if len(comps) != 1:
            raise ValueError(
                f"Input mesh must be a single connected component, got {len(comps)}"
            )
    except Exception:
        # If split fails, proceed but warn in verbose mode
        if eff_verbosity >= 1:
            logger.warning("Failed to verify single component via mesh.split()")

    zmin, zmax = mm.get_z_range()
    if not np.isfinite(zmin) or not np.isfinite(zmax) or zmax <= zmin:
        raise ValueError("Invalid z-range for mesh")

    if n_slices < 1:
        raise ValueError("n_slices must be >= 1")

    v_info(
        "Starting skeletonization: n_slices=%d, radius_mode=%s, validate_volume=%s, volume_tol=%.3f",
        n_slices,
        radius_mode,
        validate_volume,
        volume_tol,
    )
    v_debug(
        "Mesh z-range: zmin=%.6g, zmax=%.6g (dz=%.6g)",
        zmin,
        zmax,
        float(zmax - zmin),
    )

    # Warn about known issue when n_slices is a power of 2
    try:
        n_int = int(n_slices)
        if n_int > 0 and (n_int & (n_int - 1)) == 0:
            msg = "mcf2swc skeletonize: n_slices is a power of 2. The current connectivity algorithm is known to misbehave for some meshes in this case."
            warnings.warn(msg, RuntimeWarning)
            logger.warning(msg)
    except Exception:
        pass

    # Uniform partition including bounding planes
    z_planes = np.linspace(zmin, zmax, n_slices + 1)
    cut_zs = z_planes[1:-1]  # internal cuts

    # Numerical tolerance
    dz = float(zmax - zmin)
    eps = 1e-6 * (dz if dz > 0 else 1.0)
    # Use a terminal probe offset that scales with slice height to avoid
    # degenerate/unstable sections right at the bounding planes.
    slice_height = dz / float(n_slices)
    terminal_probe = max(10.0 * eps, 0.05 * slice_height)

    skel = SkeletonGraph()
    # Store transform snapshot into the skeleton graph
    try:
        skel.transforms_applied = _applied_transforms
    except Exception:
        skel.transforms_applied = []

    # ------------------------ Terminal junctions (ends) ----------------------
    # Fit junctions at bounding planes using near-plane sections
    terminal_bottom_ids = _create_junctions_at_cut(
        skel,
        mesh,
        z_plane=float(zmin),
        slice_index=0,
        probe_offset=+terminal_probe,
        radius_mode=radius_mode,
        verbosity=eff_verbosity,
    )
    terminal_top_ids = _create_junctions_at_cut(
        skel,
        mesh,
        z_plane=float(zmax),
        slice_index=n_slices - 1,
        probe_offset=-terminal_probe,
        radius_mode=radius_mode,
        verbosity=eff_verbosity,
    )

    # --------------------- Internal cuts: cross-sections ---------------------
    for si, zc in enumerate(cut_zs, start=1):
        _create_junctions_at_cut(
            skel,
            mesh,
            z_plane=float(zc),
            slice_index=si,
            probe_offset=0.0,
            radius_mode=radius_mode,
            verbosity=eff_verbosity,
        )

    v_info(
        "Constructed cross-sections: %d cuts, %d total junctions",
        len(skel.cross_sections),
        len(skel.junctions),
    )

    # ---------------------- Polyline guidance (hints) -----------------------
    # Attach per-junction polyline hints at corresponding slice planes.
    if polylines is not None and guidance is not None and guidance.use_guidance:
        # Optional snapping: use a copy to avoid mutating caller's data
        pls_use = polylines
        if guidance.snap_polylines_to_mesh:
            try:
                pls_use = polylines.copy()
                _moved, _mean = pls_use.snap_to_mesh_surface(
                    mesh,
                    project_outside_only=True,
                    max_distance=guidance.max_snap_distance,
                )
            except Exception:
                pls_use = polylines

        # Build per-slice hints
        for cs in skel.cross_sections:
            try:
                hints = _polyline_intersections_at_z(
                    pls_use, float(cs.z), float(guidance.slice_z_tolerance)
                )
            except Exception:
                hints = []

            if not hints or not cs.junction_ids:
                continue

            # Attach hints near each junction center within match radius
            for j_id in cs.junction_ids:
                j = skel.junctions.get(j_id)
                if j is None or j.center is None:
                    continue
                cx, cy = float(j.center[0]), float(j.center[1])
                accepted: List[Dict[str, Any]] = []
                for h in hints:
                    p = h.get("point")
                    if p is None:
                        continue
                    # z proximity already ensured by construction; check xy radius
                    dx = float(p[0]) - cx
                    dy = float(p[1]) - cy
                    if dx * dx + dy * dy <= guidance.junction_match_radius**2:
                        # Store minimal fields to keep memory light
                        accepted.append(
                            {
                                "polyline_id": int(h.get("polyline_id", -1)),
                                "point": np.array(p, dtype=float),
                                "tangent": np.array(
                                    h.get("tangent", [0, 0, 1]), dtype=float
                                ),
                            }
                        )
                if accepted:
                    # Merge with any existing hints
                    prev = skel.G.nodes[j_id].get("polyline_hints", [])
                    skel.G.nodes[j_id]["polyline_hints"] = list(prev) + accepted

    # ---------------- Bands: connect junctions across adjacent cuts ----------
    junctions_by_slice: Dict[int, List[int]] = {}
    for j_id, j in skel.junctions.items():
        junctions_by_slice.setdefault(j.slice_index, []).append(j_id)

    # For each band (slice interval), build components and map to junctions
    segment_id = 0
    total_band_volume = 0.0
    for band_index in range(n_slices):
        z_low = float(z_planes[band_index])
        z_high = float(z_planes[band_index + 1])

        band_mesh = _extract_band_mesh(mesh, z_low, z_high)
        if band_mesh is None:
            continue

        # Split into connected components
        try:
            components = band_mesh.split(only_watertight=False)
        except Exception:
            components = [band_mesh]
        # Track number of components in this band for potential fallbacks
        n_comp_in_band = len(components)
        v_debug(
            "Band %d [%.6g, %.6g]: %d component(s)",
            band_index,
            z_low,
            z_high,
            n_comp_in_band,
        )

        for comp in components:
            # Volumes may be negative if orientation is inverted; take absolute
            c_vol = float(abs(getattr(comp, "volume", 0.0)))
            c_area = float(getattr(comp, "area", 0.0))
            total_band_volume += c_vol

            # Determine associated junctions at z_low (lower) and z_high (upper)
            # Probe just inside the band near the lower/upper planes. If nothing is
            # detected at the immediate near-plane, incrementally step further
            # inside to robustly capture tiny/numerically fragile cross-sections.
            lower_locals = _section_polygon_centroids(comp, z_low + eps)
            if not lower_locals:
                bt = float(max(z_high - z_low, 0.0))
                probe_offsets = [
                    5.0 * eps,
                    20.0 * eps,
                    0.001 * bt,
                    0.002 * bt,
                    0.005 * bt,
                    0.01 * bt,
                ]
                # Near the bottom boundary, probe a little deeper to avoid
                # degenerate sections exactly at the cap. Include terminal_probe
                # and larger fractions of the band thickness.
                if band_index == 0:
                    probe_offsets.extend(
                        [
                            min(0.02 * bt, terminal_probe * 0.5),
                            min(0.05 * bt, terminal_probe),
                            min(0.10 * bt, terminal_probe * 2.0),
                        ]
                    )
                for off in probe_offsets:
                    if off >= bt or off <= 0.0:
                        continue
                    cand = _section_polygon_centroids(comp, z_low + off)
                    if cand:
                        lower_locals = cand
                        break

            upper_locals = _section_polygon_centroids(comp, z_high - eps)
            if not upper_locals:
                bt = float(max(z_high - z_low, 0.0))
                probe_offsets = [
                    5.0 * eps,
                    20.0 * eps,
                    0.001 * bt,
                    0.002 * bt,
                    0.005 * bt,
                    0.01 * bt,
                ]
                # Near the top boundary, probe a little deeper into the band.
                if band_index == n_slices - 1:
                    probe_offsets.extend(
                        [
                            min(0.02 * bt, terminal_probe * 0.5),
                            min(0.05 * bt, terminal_probe),
                            min(0.10 * bt, terminal_probe * 2.0),
                        ]
                    )
                for off in probe_offsets:
                    if off >= bt or off <= 0.0:
                        continue
                    cand = _section_polygon_centroids(comp, z_high - off)
                    if cand:
                        upper_locals = cand
                        break

            lower_ids = _match_centroids_to_junctions(
                centroids=lower_locals,
                z_plane=z_low,
                skel=skel,
                slice_index=band_index,
            )
            # Targeted fallback for the first band: if no lower matches were found
            # but we do have local centroids and existing junctions at slice 0,
            # assign the nearest junction(s) by polygon centroid proximity.
            if band_index == 0 and not lower_ids and lower_locals:
                # Find the cross-section record for slice 0 at z_low
                cs_candidates = [
                    cs
                    for cs in skel.cross_sections
                    if cs.slice_index == 0 and np.isclose(cs.z, z_low)
                ]
                if cs_candidates:
                    cs0 = cs_candidates[-1]
                    # Precompute polygon centroids
                    poly_centroids = []
                    for poly in cs0.polygons:
                        try:
                            poly_centroids.append(
                                np.array(
                                    [float(poly.centroid.x), float(poly.centroid.y)],
                                    dtype=float,
                                )
                            )
                        except Exception:
                            poly_centroids.append(None)
                    chosen: List[int] = []
                    for c in lower_locals:
                        best_j: Optional[int] = None
                        best_d: float = float("inf")
                        for pc, j_id in zip(poly_centroids, cs0.junction_ids):
                            if pc is None:
                                continue
                            d = float(np.linalg.norm(pc - np.asarray(c, dtype=float)))
                            if d < best_d:
                                best_d = d
                                best_j = j_id
                        if best_j is not None:
                            chosen.append(best_j)
                    # Deduplicate while preserving order
                    seen = set()
                    lower_ids = [j for j in chosen if not (j in seen or seen.add(j))]
                if eff_verbosity >= 2:
                    logger.debug(
                        "[diag] band 0 lower fallback -> lower_ids=%s (had lower_locals=%d)",
                        lower_ids,
                        len(lower_locals),
                    )

            # Additional minimal fallback for the very first band: if matching
            # completely failed (no lower_ids), try selecting a single best
            # junction from slice 0 to avoid leaving the terminal isolated.
            if band_index == 0 and not lower_ids:
                cand0 = list(junctions_by_slice.get(0, []))
                if len(cand0) == 1:
                    lower_ids = cand0
                elif len(cand0) > 1 and upper_locals:
                    # Choose the junction whose center is closest to the mean of
                    # the detected upper centroids (xy) as a conservative guess.
                    u_mean = np.mean(np.asarray(upper_locals, dtype=float), axis=0)
                    best_j = None
                    best_d = float("inf")
                    for j_id in cand0:
                        c = skel.junctions[j_id].center
                        d = float(np.linalg.norm(np.asarray([c[0], c[1]]) - u_mean))
                        if d < best_d:
                            best_d = d
                            best_j = j_id
                    if best_j is not None:
                        lower_ids = [best_j]
                if eff_verbosity >= 2:
                    logger.debug(
                        "[diag] band 0 lower minimal fallback -> lower_ids=%s from slice0 candidates=%s",
                        lower_ids,
                        cand0,
                    )

            upper_slice_index = min(band_index + 1, n_slices - 1)
            upper_ids = _match_centroids_to_junctions(
                centroids=upper_locals,
                z_plane=z_high,
                skel=skel,
                slice_index=upper_slice_index,
            )
            v_debug(
                "Band %d component: matched lower=%s, upper=%s",
                band_index,
                lower_ids,
                upper_ids,
            )
            # Symmetric minimal fallback for the first band when no upper_ids matched
            if band_index == 0 and not upper_ids:
                cand1 = list(junctions_by_slice.get(upper_slice_index, []))
                if len(cand1) == 1:
                    upper_ids = cand1
                elif len(cand1) > 1 and lower_locals:
                    l_mean = np.mean(np.asarray(lower_locals, dtype=float), axis=0)
                    best_j = None
                    best_d = float("inf")
                    for j_id in cand1:
                        c = skel.junctions[j_id].center
                        d = float(np.linalg.norm(np.asarray([c[0], c[1]]) - l_mean))
                        if d < best_d:
                            best_d = d
                            best_j = j_id
                    if best_j is not None:
                        upper_ids = [best_j]
                if eff_verbosity >= 2:
                    logger.debug(
                        "[diag] band 0 upper minimal fallback -> upper_ids=%s from slice1 candidates=%s",
                        upper_ids,
                        cand1,
                    )
            # Minimal fallback for the last band when no upper_ids matched
            if band_index == n_slices - 1 and not upper_ids:
                candU = list(junctions_by_slice.get(upper_slice_index, []))
                if len(candU) == 1:
                    upper_ids = candU
                elif len(candU) > 1 and lower_locals:
                    l_mean = np.mean(np.asarray(lower_locals, dtype=float), axis=0)
                    best_j = None
                    best_d = float("inf")
                    for j_id in candU:
                        c = skel.junctions[j_id].center
                        d = float(np.linalg.norm(np.asarray([c[0], c[1]]) - l_mean))
                        if d < best_d:
                            best_d = d
                            best_j = j_id
                    if best_j is not None:
                        upper_ids = [best_j]

            # Build a Segment metadata object
            seg = Segment(
                id=segment_id,
                slice_index=band_index,
                z_lower=z_low,
                z_upper=z_high,
                volume=c_vol,
                surface_area=c_area,
                lower_junction_ids=lower_ids,
                upper_junction_ids=upper_ids,
            )
            skel.segments.append(seg)
            # Guided connectivity selection using polyline hints if available
            used_guided = False
            if (
                polylines is not None
                and guidance is not None
                and guidance.use_guidance
                and lower_ids
                and upper_ids
            ):
                # Precompute node -> set(polyline_id) for quick intersection
                def _poly_ids_for(nid: int) -> List[int]:
                    try:
                        hints = skel.G.nodes[nid].get("polyline_hints", [])
                        return [
                            int(h.get("polyline_id", -1))
                            for h in hints
                            if h.get("polyline_id") is not None
                        ]
                    except Exception:
                        return []

                guided_pairs: List[Tuple[int, int, List[int]]] = []
                for jl in lower_ids:
                    ids_l = set(_poly_ids_for(jl))
                    if not ids_l:
                        continue
                    for ju in upper_ids:
                        if jl == ju:
                            continue
                        ids_u = set(_poly_ids_for(ju))
                        if not ids_u:
                            continue
                        shared = sorted(list(ids_l.intersection(ids_u)))
                        if shared:
                            guided_pairs.append((jl, ju, shared))

                # If we found any guided pairs, add only those; else fall back
                if guided_pairs:
                    used_guided = True
                    for jl, ju, support in guided_pairs:
                        if skel.G.has_edge(jl, ju):
                            continue
                        skel.G.add_edge(
                            jl,
                            ju,
                            kind="segment",
                            segment_id=int(seg.id),
                            slice_index=int(seg.slice_index),
                            z_lower=float(seg.z_lower),
                            z_upper=float(seg.z_upper),
                            volume=float(seg.volume),
                            surface_area=float(seg.surface_area),
                            polyline_support=support,
                            chosen_by="guided",
                        )

            # Fallback (and default) connectivity: fully bipartite within band
            if not used_guided and lower_ids and upper_ids:
                skel.add_segment_edges(seg)
            segment_id += 1

    # -------------------- Attach boundary polylines to nodes -------------------
    # After connectivity is established, store each cross-section polygon's
    # 2D boundary as a node attribute `boundary_2d` for plotting functions.
    for cs in skel.cross_sections:
        if not cs.polygons or not cs.junction_ids:
            continue
        for poly, j_id in zip(cs.polygons, cs.junction_ids):
            try:
                coords = np.asarray(poly.exterior.coords, dtype=float)
                if coords is not None and len(coords) >= 2 and coords.shape[1] >= 2:
                    skel.G.nodes[j_id]["boundary_2d"] = coords[:, :2]
            except Exception:
                # If shapely polygon lacks exterior or any other issue, skip
                continue

    # -------------------------- Volume conservation --------------------------
    if validate_volume:
        try:
            mesh_vol = float(abs(getattr(mesh, "volume", 0.0)))
            rel_err = abs(total_band_volume - mesh_vol) / (mesh_vol + 1e-12)
            if rel_err > volume_tol:
                raise ValueError(
                    f"Volume conservation failed: bands={total_band_volume:.6g}, "
                    f"mesh={mesh_vol:.6g}, rel_err={rel_err:.3%} (> {volume_tol:.1%})"
                )
        except Exception as e:
            if eff_verbosity >= 1:
                logger.warning("Volume validation warning: %s", e)
        else:
            v_info(
                "Volume check OK: bands=%.6g, mesh=%.6g, rel_err=%.3f%%",
                total_band_volume,
                mesh_vol,
                100.0 * (abs(total_band_volume - mesh_vol) / (mesh_vol + 1e-12)),
            )

    # (Removed) Genus-based topological closure: no non-adjacent closure edges are added.

    # ---------------- Safety net: connect isolated terminal nodes -------------
    if connect_isolated_terminals:
        # Bottom slice 0 isolated nodes connect to nearest in slice 1
        s0 = list(junctions_by_slice.get(0, []))
        s1 = list(junctions_by_slice.get(1, [])) if n_slices >= 2 else []
        for j in s0:
            if skel.G.degree(j) == 0 and s1:
                cj = skel.junctions[j].center
                best = None
                best_d = float("inf")
                for k in s1:
                    ck = skel.junctions[k].center
                    d = float(np.linalg.norm(np.asarray(cj[:2]) - np.asarray(ck[:2])))
                    if d < best_d:
                        best_d = d
                        best = k
                if best is not None and not skel.G.has_edge(j, best):
                    if j == best:
                        # Avoid self-loop under any circumstance
                        continue
                    if eff_verbosity >= 2:
                        logger.debug(
                            "[diag] safety-net: connecting isolated bottom node %s -> %s",
                            j,
                            best,
                        )
                    skel.G.add_edge(
                        j,
                        best,
                        kind="segment",
                        segment_id=-2,
                        slice_index=0,
                        z_lower=float(skel.G.nodes[j].get("z", zmin)),
                        z_upper=float(skel.G.nodes[best].get("z", zmin)),
                        volume=0.0,
                        surface_area=0.0,
                    )
        # Top slice isolated nodes connect to nearest in previous slice
        top_slice = n_slices - 1
        sT = list(junctions_by_slice.get(top_slice, []))
        sPrev = (
            list(junctions_by_slice.get(top_slice - 1, []))
            if top_slice - 1 >= 0
            else []
        )
        for j in sT:
            if skel.G.degree(j) == 0 and sPrev:
                cj = skel.junctions[j].center
                best = None
                best_d = float("inf")
                for k in sPrev:
                    ck = skel.junctions[k].center
                    d = float(np.linalg.norm(np.asarray(cj[:2]) - np.asarray(ck[:2])))
                    if d < best_d:
                        best_d = d
                        best = k
                if best is not None and not skel.G.has_edge(j, best):
                    if j == best:
                        # Avoid self-loop under any circumstance
                        continue
                    if eff_verbosity >= 2:
                        logger.debug(
                            "[diag] safety-net: connecting isolated top node %s -> %s",
                            j,
                            best,
                        )
                    skel.G.add_edge(
                        best,
                        j,
                        kind="segment",
                        segment_id=-2,
                        slice_index=top_slice - 1,
                        z_lower=float(skel.G.nodes[best].get("z", zmax)),
                        z_upper=float(skel.G.nodes[j].get("z", zmax)),
                        volume=0.0,
                        surface_area=0.0,
                    )

    # -------------------------- Connectivity check ---------------------------
    # By default, require a single connected component for the SkeletonGraph.
    # This can be bypassed by setting enforce_connected=False.
    if enforce_connected:
        try:
            ncomp = nx.number_connected_components(skel.G)
        except Exception:
            ncomp = 1 if skel.G.number_of_nodes() <= 1 else 2
        if ncomp > 1:
            raise ValueError(
                f"SkeletonGraph has {ncomp} connected components, but a single connected component is required. "
                f"Re-check mesh integrity or call skeletonize(..., enforce_connected=False) to bypass."
            )
        v_info(
            "Final connectivity: %d node(s), %d edge(s), %d component(s)",
            skel.G.number_of_nodes(),
            skel.G.number_of_edges(),
            ncomp,
        )

    return skel


# ============================================================================
# Helpers
# ============================================================================


def _polyline_intersections_at_z(
    pls: "PolylinesSkeleton", z: float, tol: float = 1e-3
) -> List[Dict[str, Any]]:
    """Return intersection hints of polylines with the plane z=const.

    For each polyline segment, if the segment crosses the target z (or an
    endpoint lies within ``tol`` of z), record an intersection point and an
    approximate tangent based on the local segment direction.

    Returns a list of dicts: {"polyline_id", "point": np.ndarray(3), "tangent": np.ndarray(3)}.
    """

    def _norm(v: np.ndarray) -> np.ndarray:
        vv = np.asarray(v, dtype=float)
        n = float(np.linalg.norm(vv))
        if not np.isfinite(n) or n <= 1e-12:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        return vv / n

    results: List[Dict[str, Any]] = []
    if pls is None or not getattr(pls, "polylines", None):
        return results

    for pid, pl in enumerate(pls.polylines):
        P = np.asarray(pl, dtype=float)
        if P.ndim != 2 or P.shape[1] != 3 or P.shape[0] == 0:
            continue

        # Track near-duplicate suppression to avoid exploding hints
        seen_points: List[np.ndarray] = []

        n = P.shape[0]
        for i in range(n - 1):
            p0 = P[i]
            p1 = P[i + 1]
            z0 = float(p0[2])
            z1 = float(p1[2])

            # Helper to add a point with deduplication
            def _try_add(pt: np.ndarray, tvec: np.ndarray) -> None:
                nonlocal seen_points
                # Deduplicate by proximity (3D)
                for q in seen_points:
                    if float(np.linalg.norm(pt - q)) <= max(1e-9, tol * 0.1):
                        return
                seen_points.append(pt)
                results.append(
                    {
                        "polyline_id": int(pid),
                        "point": np.array(pt, dtype=float),
                        "tangent": _norm(tvec),
                    }
                )

            seg_dir = p1 - p0

            # Endpoints on the plane (within tol)
            if abs(z0 - z) <= tol:
                # Tangent preference: forward segment if available, else backward
                tvec = seg_dir if i + 1 < n else (p0 - P[i - 1] if i > 0 else seg_dir)
                _try_add(p0, tvec)
            if abs(z1 - z) <= tol:
                tvec = seg_dir if i + 1 < n else (p1 - p0)
                _try_add(p1, tvec)

            # Proper crossing strictly between endpoints
            if (z - z0) * (z - z1) < 0.0:
                # Linear interpolation fraction along the segment
                t = (z - z0) / (z1 - z0)
                pt = p0 + t * seg_dir
                _try_add(pt, seg_dir)

    return results


def _create_junctions_at_cut(
    skel: SkeletonGraph,
    mesh: trimesh.Trimesh,
    *,
    z_plane: float,
    slice_index: int,
    probe_offset: float,
    radius_mode: str,
    verbosity: int,
) -> List[int]:
    """
    Create junctions for all closed areas at a cut plane.

    probe_offset allows probing slightly off the plane (useful for bounding planes).
    """
    z_probe = float(z_plane + probe_offset)
    polygons = _cross_section_polygons(mesh, z_probe)
    if not polygons:
        # No intersection at this plane – nothing to add
        skel.cross_sections.append(
            CrossSection(
                z=z_plane, slice_index=slice_index, polygons=[], junction_ids=[]
            )
        )
        return []

    # Validate non-overlap within same cut (provide z-level for diagnostics)
    _assert_no_overlap(polygons, z_level=float(z_plane))

    added_ids: List[int] = []
    junction_ids: List[int] = []
    for idx, poly in enumerate(polygons):
        area = float(poly.area)
        if area <= 0:
            continue
        centroid = np.asarray([poly.centroid.x, poly.centroid.y], dtype=float)
        if radius_mode == "equivalent_area":
            radius = float(np.sqrt(area / np.pi))
        else:
            raise ValueError(f"Unknown radius_mode: {radius_mode}")

        j_id = _next_junction_id(skel)
        j = Junction(
            id=j_id,
            z=float(z_plane),
            center=np.array([centroid[0], centroid[1], float(z_plane)], dtype=float),
            radius=radius,
            area=area,
            slice_index=int(slice_index),
            cross_section_index=int(idx),
        )
        skel.add_junction(j)
        added_ids.append(j_id)
        junction_ids.append(j_id)

    skel.cross_sections.append(
        CrossSection(
            z=z_plane,
            slice_index=slice_index,
            polygons=polygons,
            junction_ids=junction_ids,
        )
    )
    # Detailed per-cut diagnostics at verbosity>=2
    if verbosity >= 2:
        logger.debug(
            "Cut z=%.6g (slice %d, probe_offset=%.3g): %d polygon(s) -> junction_ids=%s",
            z_plane,
            slice_index,
            probe_offset,
            len(polygons),
            junction_ids,
        )
    return added_ids


def _cross_section_polygons(mesh: trimesh.Trimesh, z: float) -> List[sgeom.Polygon]:
    """Return shapely polygons for mesh ∩ plane z=constant."""
    try:
        path = mesh.section(
            plane_origin=[0.0, 0.0, float(z)], plane_normal=[0.0, 0.0, 1.0]
        )
    except Exception:
        path = None
    if path is None or not hasattr(path, "entities") or len(path.entities) == 0:
        return []

    polys: List[sgeom.Polygon] = []
    # Collect polygons from all entities; ensure closed loops
    try:
        for entity in path.entities:
            if not hasattr(entity, "points"):
                continue
            points_2d = path.vertices[entity.points]
            if points_2d is None or len(points_2d) < 3:
                continue
            # Ensure closure
            if not np.allclose(points_2d[0], points_2d[-1]):
                pts = np.vstack([points_2d, points_2d[0]])
            else:
                pts = points_2d
            poly = sgeom.Polygon(pts[:, :2])
            if poly.is_valid and poly.area > 0:
                polys.append(poly)
    except Exception:
        # Fallback: no polygons
        return []

    # Compose nested polygons into polygons-with-holes using containment parity
    try:
        composed = _compose_polygons_with_holes(polys)
    except Exception:
        composed = polys
    return composed


def _assert_no_overlap(
    polygons: List[sgeom.Polygon], tol: float = 1e-3, z_level: Optional[float] = None
) -> None:
    """Ensure polygons in the same cross-section do not overlap."""
    n = len(polygons)
    for i in range(n):
        for j in range(i + 1, n):
            inter = polygons[i].intersection(polygons[j])
            if not inter.is_empty and float(inter.area) > tol:
                # Diagnostic plot to visualize the supposed overlap
                try:
                    fig, ax = plt.subplots(figsize=(6, 6))
                    # Plot all polygons with light colors
                    for k, poly in enumerate(polygons):
                        try:
                            x, y = poly.exterior.xy
                            ax.fill(
                                x,
                                y,
                                alpha=0.2,
                                facecolor=(0.2, 0.4, 0.8, 0.2),
                                edgecolor=(0.2, 0.4, 0.8, 0.8),
                                linewidth=1.0,
                            )
                            # Plot any interior rings (holes)
                            for ring in poly.interiors:
                                rx, ry = ring.xy
                                ax.plot(
                                    rx, ry, color=(0.2, 0.4, 0.8, 0.8), linestyle="--"
                                )
                            # Annotate polygon index at centroid
                            c = poly.centroid
                            ax.text(
                                c.x,
                                c.y,
                                f"{k}",
                                fontsize=10,
                                ha="center",
                                va="center",
                                color="black",
                                bbox=dict(
                                    boxstyle="round,pad=0.2",
                                    fc="white",
                                    ec="none",
                                    alpha=0.7,
                                ),
                            )
                        except Exception:
                            continue

                    # Highlight the intersection area in red
                    try:
                        geoms = []
                        if getattr(inter, "geom_type", "") == "Polygon":
                            geoms = [inter]
                        elif getattr(inter, "geom_type", "") in (
                            "MultiPolygon",
                            "GeometryCollection",
                        ):
                            geoms = [
                                g
                                for g in getattr(inter, "geoms", [])
                                if hasattr(g, "exterior")
                            ]
                        for g in geoms:
                            x, y = g.exterior.xy
                            ax.fill(x, y, color="red", alpha=0.4, linewidth=0)
                    except Exception:
                        pass

                    title = (
                        f"Overlap detected between polygons {i} and {j} "
                        f"(area={float(inter.area):.6g})"
                    )
                    if z_level is not None:
                        title += f" at z={z_level:.6g}"
                    ax.set_title(title)
                    ax.set_aspect("equal", adjustable="box")
                    ax.grid(True, linestyle=":", alpha=0.3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    plt.tight_layout()
                    plt.show()
                    plt.close(fig)
                    # Also log a concise diagnostic line
                    if z_level is not None:
                        logger.debug(
                            "[diag] Cross-section overlap at z=%.6g: polygons %d and %d, area=%.6g (> tol=%.2g)",
                            z_level,
                            i,
                            j,
                            float(inter.area),
                            tol,
                        )
                    else:
                        logger.debug(
                            "[diag] Cross-section overlap: polygons %d and %d, area=%.6g (> tol=%.2g)",
                            i,
                            j,
                            float(inter.area),
                            tol,
                        )
                except Exception as e:
                    logger.warning(
                        "Failed to plot cross-section overlap diagnostics: %s", e
                    )

                raise ValueError(
                    "Overlapping cross-section polygons detected in a single cut"
                )


def _compose_polygons_with_holes(polys: List[sgeom.Polygon]) -> List[sgeom.Polygon]:
    """
    Given a list of simple polygons from a planar section, compose them into a set of
    polygons where interior rings represent holes. We use containment parity:
    - Even depth polygons (0, 2, ...) are exteriors.
    - Odd depth polygons (1, 3, ...) are holes relative to their immediate parent.

    Returns a list of shapely Polygons where holes are assigned to their nearest
    containing exterior. Disconnected components remain separate polygons.
    """
    if not polys:
        return []

    # Sort by descending area so potential parents come before children
    indices = list(range(len(polys)))
    indices.sort(key=lambda i: float(polys[i].area), reverse=True)

    # Precompute contains relationships and immediate parent
    parents: Dict[int, Optional[int]] = {i: None for i in indices}
    depths: Dict[int, int] = {i: 0 for i in indices}

    for i_pos, i in enumerate(indices):
        p_i = polys[i]
        # Find the smallest-area parent that strictly contains p_i
        best_parent = None
        best_area = float("inf")
        for j_pos, j in enumerate(indices):
            if j == i:
                continue
            p_j = polys[j]
            # Strict containment: j contains i and not just touching boundary
            try:
                if p_j.contains(p_i):
                    a = float(p_j.area)
                    if a < best_area:
                        best_area = a
                        best_parent = j
            except Exception:
                continue
        parents[i] = best_parent

    # Compute depth by walking up parent chain
    for i in indices:
        d = 0
        cur = parents[i]
        while cur is not None:
            d += 1
            cur = parents[cur]
            # Safety to avoid pathological loops
            if d > len(indices):
                break
        depths[i] = d

    # For each polygon, collect its immediate children
    children: Dict[int, List[int]] = {i: [] for i in indices}
    for i in indices:
        p = parents[i]
        if p is not None:
            children[p].append(i)

    # Build composed polygons: for each even-depth polygon, assign holes as its
    # immediate odd-depth children (depth + 1)
    result: List[sgeom.Polygon] = []
    for i in indices:
        if depths[i] % 2 != 0:
            continue  # not an exterior at this parity level
        exterior_poly = polys[i]
        holes_coords: List[List[Tuple[float, float]]] = []
        for c in children[i]:
            if depths[c] == depths[i] + 1:
                ring = list(polys[c].exterior.coords)
                holes_coords.append([(float(x), float(y)) for x, y in ring])
        try:
            composed = sgeom.Polygon(exterior_poly.exterior.coords, holes=holes_coords)
            if composed.is_valid and composed.area > 0:
                result.append(composed)
        except Exception:
            # Fallback: keep the unmodified exterior
            result.append(exterior_poly)

    return result


def _extract_band_mesh(
    mesh: trimesh.Trimesh, z_low: float, z_high: float
) -> Optional[trimesh.Trimesh]:
    """
    Extract the band sub-mesh with z in [z_low, z_high] by two-plane slicing
    using existing MeshManager.slice_mesh_by_z.
    """
    if z_high <= z_low:
        return None

    mm_top = MeshManager(mesh)
    below_high, _ = mm_top.slice_mesh_by_z(z_high, cap=True, validate=True)
    if below_high is None:
        return None

    mm_band = MeshManager(below_high)
    _, above_low = mm_band.slice_mesh_by_z(z_low, cap=True, validate=True)
    return above_low


def _section_polygon_centroids(
    mesh: trimesh.Trimesh, z_probe: float
) -> List[np.ndarray]:
    """Centroids of intersection polygons at z=z_probe for the given mesh."""
    polys = _cross_section_polygons(mesh, z_probe)
    cents: List[np.ndarray] = []
    for p in polys:
        if p.area > 0:
            cents.append(np.array([p.centroid.x, p.centroid.y], dtype=float))
    return cents


def _match_centroids_to_junctions(
    *,
    centroids: List[np.ndarray],
    z_plane: float,
    skel: SkeletonGraph,
    slice_index: int,
) -> List[int]:
    """
    Map 2D centroids at a cut to the IDs of the corresponding global junctions
    at the same cut plane.
    """
    # Find the cross-section for this slice_index and exact z_plane
    # CrossSections are appended in creation order; retrieve the matching one
    cs_candidates = [
        cs
        for cs in skel.cross_sections
        if cs.slice_index == slice_index and np.isclose(cs.z, z_plane)
    ]
    if not cs_candidates:
        return []
    cs = cs_candidates[-1]

    # Derive a local length scale from polygons to build a robust tolerance.
    radii = []
    for p in cs.polygons:
        try:
            a = float(p.area)
            if a > 0:
                radii.append(math.sqrt(a / math.pi))
        except Exception:
            continue
    local_scale = float(np.median(radii)) if radii else 1.0
    # Base tolerance ~5% of local radius, with a small absolute floor.
    tol = max(1e-6, 0.05 * local_scale)

    result: List[int] = []
    for c in centroids:
        point = sgeom.Point(float(c[0]), float(c[1]))

        # First pass: exact containment or within tolerance distance
        best_id: Optional[int] = None
        best_d = float("inf")
        for poly, j_id in zip(cs.polygons, cs.junction_ids):
            try:
                if poly.contains(point) or poly.touches(point):
                    best_id = j_id
                    best_d = 0.0
                    break
                # Distance is zero if inside; else edge distance in 2D
                d = float(poly.distance(point))
                if d < best_d:
                    best_d = d
                    best_id = j_id
            except Exception:
                continue

        # Accept if within tolerance; otherwise, allow a slightly larger fallback
        if best_id is not None and (best_d <= tol or best_d <= 3.0 * tol):
            result.append(best_id)

    return result


def _next_junction_id(skel: SkeletonGraph) -> int:
    return 0 if not skel.junctions else max(skel.junctions.keys()) + 1


def _vertex_adjacency_graph_with_labels(
    mesh: trimesh.Trimesh,
) -> Tuple[nx.Graph, Dict[int, int]]:
    """
    Build a vertex-adjacency graph for a mesh component where nodes are vertex
    indices and edges connect vertices that share a triangle edge. Edge weights
    are the Euclidean distances between vertices.

    Returns the graph along with a mapping from vertex index to its connected
    component label (0..K-1) in this graph.
    """
    V = np.asarray(mesh.vertices, dtype=float)
    F = np.asarray(mesh.faces, dtype=int)

    Gv = nx.Graph()
    nV = int(V.shape[0]) if V is not None else 0
    if nV <= 0 or F is None or len(F) == 0:
        return Gv, {}

    Gv.add_nodes_from(range(nV))

    def _add_edge(u: int, v: int):
        if u == v:
            return
        duv = float(np.linalg.norm(V[u] - V[v]))
        # Avoid zero-weight degenerate edges
        w = duv if duv > 0.0 else 1e-12
        if Gv.has_edge(u, v):
            # Keep the smaller weight if multiple faces share the same edge
            if w < Gv[u][v].get("weight", w):
                Gv[u][v]["weight"] = w
        else:
            Gv.add_edge(u, v, weight=w)

    # For each triangular face, connect its 3 edges
    try:
        for tri in F:
            if len(tri) < 3:
                continue
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            _add_edge(a, b)
            _add_edge(b, c)
            _add_edge(c, a)
    except Exception:
        # If faces are malformed, return whatever we've built so far
        pass

    # Label connected components in the vertex graph
    labels: Dict[int, int] = {}
    try:
        for idx, comp in enumerate(nx.connected_components(Gv)):
            for v in comp:
                labels[int(v)] = int(idx)
    except Exception:
        # If labeling fails, leave labels empty
        labels = {}

    return Gv, labels


def _nearest_vertex_index(mesh: trimesh.Trimesh, point: np.ndarray) -> Optional[int]:
    """
    Return the index of the mesh vertex nearest to the given 3D point. If the
    mesh has no vertices, return None.
    """
    V = np.asarray(mesh.vertices, dtype=float)
    if V is None or len(V) == 0:
        return None
    p = np.asarray(point, dtype=float)
    d2 = np.sum((V - p[None, :]) ** 2, axis=1)
    idx = int(np.argmin(d2))
    return idx


def _seed_vertices_for_junctions(
    mesh: trimesh.Trimesh,
    skel: SkeletonGraph,
    *,
    slice_index: int,
    z_plane: float,
    junction_ids: List[int],
    z_bias: float = 0.0,
) -> Dict[int, int]:
    """
    For each junction id at a specified plane/slice, pick a representative mesh
    surface vertex to serve as a seed for path-following. We take the junction's
    2D (x,y) center and place it at z=z_plane+z_bias to avoid degeneracy right
    on the slicing plane, then choose the nearest mesh vertex in 3D.

    Returns a dict mapping junction_id -> vertex_index (in `mesh`).
    """
    result: Dict[int, int] = {}
    for jid in junction_ids:
        j = skel.junctions.get(jid)
        if j is None:
            continue
        cx, cy = float(j.center[0]), float(j.center[1])
        seed_p = np.array([cx, cy, float(z_plane + z_bias)], dtype=float)
        vid = _nearest_vertex_index(mesh, seed_p)
        if vid is not None:
            result[jid] = int(vid)
    return result

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class Junction:
    """
    Container for a traced skeleton node.

    Fields mirror what the tracing pipeline in `trace.py` constructs for each
    sample along a polyline. The essential geometry is in `xyz` (XYZ) and
    `radius`; other fields are retained for diagnostics/bookkeeping.
    """

    id: int
    xyz: np.ndarray
    radius: float


class SkeletonGraph(nx.Graph):
    """
    Minimal skeleton graph built during tracing.

    This class subclasses `networkx.Graph`. Nodes should be keyed directly by
    their junction `id` and store at least the attributes `xyz` (3-vector) and
    `radius` (float).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Populated by trace.build_traced_skeleton_graph for optional SWC export adjustments
        
    def add_junction(self, j: Junction) -> None:
        """Add a SWC-like junction as a node with attributes.

        Node key = `j.id`.
        Stored attributes include `xyz` and `radius`.
        """
        self.add_node(
            int(j.id),
            xyz=j.xyz,
            radius=float(j.radius),
        )

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def plot(
        self,
        backend: str = "matplotlib",
        *,
        nodes: bool = True,
        edges: bool = True,
        with_radius: bool = True,
        node_scale: float = 1.0,
        node_color: str = "blue",
        edge_color: str = "black",
        node_alpha: float = 0.9,
        edge_alpha: float = 0.6,
        title: str | None = None,
        ax=None,
        show: bool = True,
    ):
        """
        Plot the skeleton in 3D.

        Parameters:
            backend: "matplotlib" (default) or "plotly".
            nodes: Whether to render nodes as scatter points.
            edges: Whether to render edges as line segments.
            with_radius: If True, scale node marker size by stored `radius`.
            node_scale: Scalar scale factor for node sizes. For matplotlib,
                final marker size is ``s = (radius * node_scale)**2`` if
                ``with_radius`` else ``node_scale**2``.
            node_color: Color for node markers.
            edge_color: Color for edge segments.
            node_alpha: Alpha for node markers.
            edge_alpha: Alpha for edge segments.
            title: Optional title for the plot.
            ax: Existing matplotlib 3D axes to draw on. If None, a new figure
                and axes are created.
            show: If True, call ``plt.show()`` (matplotlib) or display the
                figure (plotly) before returning.

        Returns:
            - Matplotlib: the ``Axes3D`` instance.
            - Plotly: the ``go.Figure`` instance.
        """

        if self.number_of_nodes() == 0:
            if backend == "matplotlib":
                if ax is None:
                    fig = plt.figure(figsize=(6, 5))
                    ax = fig.add_subplot(111, projection="3d")
                ax.set_title(title or "Empty SkeletonGraph")
                if show:
                    plt.show()
                return ax
            elif backend == "plotly":
                try:
                    import plotly.graph_objects as go
                except Exception as exc:  # pragma: no cover - import guard
                    raise ImportError("plotly is required for backend='plotly'") from exc
                fig = go.Figure()
                fig.update_layout(title=title or "Empty SkeletonGraph",
                                  scene_aspectmode="data")
                if show:
                    fig.show()
                return fig
            else:
                raise ValueError("backend must be 'matplotlib' or 'plotly'")

        # Collect node coordinates and radii
        nids = list(self.nodes())
        xyzs = np.array([np.asarray(self.nodes[n]["xyz"]).reshape(3) for n in nids])
        radii = np.array([
            float(self.nodes[n].get("radius", 1.0)) if with_radius else 1.0 for n in nids
        ], dtype=float)

        # Build edges as pairs of coordinates
        edge_pairs = list(self.edges()) if edges else []
        edge_segments = None
        if edge_pairs:
            edge_segments = [
                (
                    np.asarray(self.nodes[u]["xyz"]).reshape(3),
                    np.asarray(self.nodes[v]["xyz"]).reshape(3),
                )
                for u, v in edge_pairs
            ]

        if backend == "matplotlib":
            if ax is None:
                fig = plt.figure(figsize=(6, 5))
                ax = fig.add_subplot(111, projection="3d")

            # Plot edges
            if edge_segments:
                for p0, p1 in edge_segments:
                    ax.plot(
                        [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                        color=edge_color, alpha=edge_alpha, linewidth=1.0,
                    )

            # Plot nodes
            if nodes:
                sizes = (np.clip(radii, 1e-6, np.inf) * node_scale) ** 2
                ax.scatter(
                    xyzs[:, 0], xyzs[:, 1], xyzs[:, 2],
                    s=sizes, c=node_color, alpha=node_alpha, depthshade=True,
                )

            # Equal aspect and labels
            self._set_axes_equal_3d(ax, xyzs)
            if title:
                ax.set_title(title)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            if show:
                plt.show()
            return ax

        elif backend == "plotly":
            try:
                import plotly.graph_objects as go
            except Exception as exc:  # pragma: no cover - import guard
                raise ImportError("plotly is required for backend='plotly'") from exc

            traces = []
            if edges and edge_segments:
                # Build a single lines trace with NaN breaks
                xe, ye, ze = [], [], []
                for p0, p1 in edge_segments:
                    xe += [p0[0], p1[0], None]
                    ye += [p0[1], p1[1], None]
                    ze += [p0[2], p1[2], None]
                traces.append(
                    go.Scatter3d(
                        x=xe, y=ye, z=ze,
                        mode="lines",
                        line=dict(color=edge_color, width=3),
                        opacity=edge_alpha,
                        name="edges",
                        showlegend=False,
                    )
                )

            if nodes:
                size_pts = (np.clip(radii, 1e-6, np.inf) * node_scale)
                traces.append(
                    go.Scatter3d(
                        x=xyzs[:, 0], y=xyzs[:, 1], z=xyzs[:, 2],
                        mode="markers",
                        marker=dict(size=size_pts, color=node_color, opacity=node_alpha),
                        name="nodes",
                        showlegend=False,
                    )
                )

            fig = go.Figure(data=traces)
            fig.update_layout(
                title=title,
                scene_aspectmode="data",
                margin=dict(l=0, r=0, b=0, t=40 if title else 10),
            )
            if show:
                fig.show()
            return fig

        else:
            raise ValueError("backend must be 'matplotlib' or 'plotly'")

    @staticmethod
    def _set_axes_equal_3d(ax, points: np.ndarray) -> None:
        """Set 3D plot axes to equal scale based on provided points.

        This makes spheres appear as spheres and cubes as cubes, regardless of
        data ranges.
        """
        if points.size == 0:
            return
        x_limits = [np.min(points[:, 0]), np.max(points[:, 0])]
        y_limits = [np.min(points[:, 1]), np.max(points[:, 1])]
        z_limits = [np.min(points[:, 2]), np.max(points[:, 2])]
        x_range = x_limits[1] - x_limits[0]
        y_range = y_limits[1] - y_limits[0]
        z_range = z_limits[1] - z_limits[0]
        max_range = max(x_range, y_range, z_range)
        x_mid = np.mean(x_limits)
        y_mid = np.mean(y_limits)
        z_mid = np.mean(z_limits)
        half = max_range / 2.0
        ax.set_xlim(x_mid - half, x_mid + half)
        ax.set_ylim(y_mid - half, y_mid + half)
        ax.set_zlim(z_mid - half, z_mid + half)


    # ------------------------------------------------------------------
    # SWC Export
    # ------------------------------------------------------------------
    def to_swc(
        self,
        path: str | None = None,
        *,
        type_index: int = 3,
        annotate_breaks: bool = True,
    ) -> str:
        """Export the skeleton to SWC format, breaking cycles by duplicating nodes.

        The SWC format is a line-based format with columns:
            n T x y z R parent
        where `n` is the node id (integer), `T` is the SWC type index (integer),
        `x,y,z` are coordinates, `R` is radius, and `parent` is the parent's id
        (or -1 for the root).

        This method constructs a spanning forest over the undirected skeleton
        graph for parent-child relations. For every non-tree edge that would
        introduce a cycle, it duplicates one endpoint node (copying xyz and
        radius) and attaches the duplicate as a child of the other endpoint.

        Args:
            path: If provided, write the SWC text to this file. If None, return
                the SWC text as a string.
            type_index: Integer put in the T column for all nodes.
            annotate_breaks: If True, include header comments indicating how to
                reconnect duplicates to recreate each broken cycle.

        Returns:
            The SWC text if `path` is None; otherwise returns the written text.
        """
        # Collect original node ids and attributes
        if self.number_of_nodes() == 0:
            swc_text = "# Empty skeleton\n"
            if path is None:
                return swc_text
            with open(path, "w", encoding="utf-8") as f:
                f.write(swc_text)
            return swc_text

        # Ensure required attributes exist
        for n in self.nodes:
            if "xyz" not in self.nodes[n] or "radius" not in self.nodes[n]:
                raise KeyError(f"Node {n} missing required attributes 'xyz' and 'radius'")

        # Build a spanning forest per connected component with DFS, choosing
        # a terminal (degree==1) node as root when available. Create a DFS
        # visitation order to determine SWC indices sequentially starting at 1.
        parents_orig: dict[int, int] = {}
        tree_edges: set[frozenset[int]] = set()
        comp_roots: list[int] = []
        comp_orders: list[list[int]] = []

        # Sort components deterministically by min node id
        components = [sorted(int(x) for x in comp) for comp in nx.connected_components(self)]
        components.sort(key=lambda nodes: (len(nodes) == 0, nodes[0] if nodes else -1))

        for comp_nodes in components:
            if not comp_nodes:
                continue
            # Prefer a terminal as root
            terminals = [n for n in comp_nodes if int(self.degree[n]) == 1]
            root = min(terminals) if terminals else min(comp_nodes)
            comp_roots.append(root)
            parents_orig[int(root)] = -1
            # DFS tree
            for u, v in nx.dfs_edges(self, source=root):
                u = int(u)
                v = int(v)
                parents_orig[v] = u
                tree_edges.add(frozenset({u, v}))
            # DFS visitation order (preorder)
            order = [int(n) for n in nx.dfs_preorder_nodes(self, source=root)]
            comp_orders.append(order)
            # Ensure isolated nodes get parent -1
            for n in comp_nodes:
                if n not in parents_orig:
                    parents_orig[int(n)] = -1

        # Determine non-tree edges (these would create cycles)
        extra_edges: list[tuple[int, int]] = []
        for (u, v) in self.edges():
            e = frozenset({int(u), int(v)})
            if e not in tree_edges:
                extra_edges.append((int(u), int(v)))

        # Assign new SWC indices according to concatenated DFS orders
        new_id: dict[int, int] = {}
        order_all: list[int] = []
        for od in comp_orders:
            order_all.extend(od)
        next_index = 1
        for n in order_all:
            if n not in new_id:
                new_id[n] = next_index
                next_index += 1

        # Prepare SWC entries for original nodes using DFS-based indices
        entries: list[tuple[int, int, float, float, float, float, int]] = []
        for n in order_all:
            nid = new_id[n]
            xyz = np.asarray(self.nodes[n]["xyz"], dtype=float).reshape(3)
            r = float(self.nodes[n].get("radius", 0.0))
            parent_orig = parents_orig.get(n, -1)
            parent_id = new_id[parent_orig] if parent_orig in new_id else -1
            entries.append((int(nid), int(type_index), xyz[0], xyz[1], xyz[2], r, int(parent_id)))

        # Process extra edges by duplicating one endpoint and attaching to the other
        cycle_annotations: list[tuple[int, int]] = []  # (duplicate_swc_id, original_swc_id)
        for (u, v) in extra_edges:
            # Choose which node to duplicate: prefer higher degree (branching)
            deg_u = self.degree[u]
            deg_v = self.degree[v]
            dup_orig = v if deg_v >= deg_u else u
            other_orig = u if dup_orig == v else v

            xyz = np.asarray(self.nodes[dup_orig]["xyz"], dtype=float).reshape(3)
            r = float(self.nodes[dup_orig].get("radius", 0.0))
            dup_swc = int(next_index)
            parent_swc = int(new_id.get(other_orig, -1))
            entries.append((dup_swc, int(type_index), xyz[0], xyz[1], xyz[2], r, parent_swc))
            # Record annotation using SWC indices
            cycle_annotations.append((dup_swc, int(new_id.get(dup_orig, dup_swc))))
            next_index += 1

        # Compose SWC text
        lines: list[str] = []
        lines.append("# generated by mcf2swc SkeletonGraph.to_swc")
        lines.append(
            f"# dfs_roots={' '.join(str(new_id.get(r, r)) for r in comp_roots)}"
        )
        lines.append(
            f"# nodes={self.number_of_nodes()} extra_edges={len(extra_edges)} duplicates={len(cycle_annotations)}"
        )
        lines.append(f"# type_index={int(type_index)}")
        if annotate_breaks and cycle_annotations:
            for dup_id, orig_id in cycle_annotations:
                lines.append(f"# CYCLE_BREAK reconnect {dup_id} {orig_id}")

        # Sort entries by SWC id for readability
        entries.sort(key=lambda t: int(t[0]))
        for nid, T, x, y, z, R, parent in entries:
            lines.append(f"{nid} {T} {x:.6f} {y:.6f} {z:.6f} {R:.6f} {parent}")

        swc_text = "\n".join(lines) + "\n"
        if path is None:
            return swc_text
        with open(path, "w", encoding="utf-8") as f:
            f.write(swc_text)
        return swc_text

__all__ = ["SkeletonGraph", "Junction"]

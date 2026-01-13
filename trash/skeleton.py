from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from swctools import SWCModel, FrustaSet, plot_model


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
    # SWC Export
    # ------------------------------------------------------------------
    def to_swc_file(
        self,
        path: str | None = None,
        *,
        tag: int = 3,
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
            tag: Integer put in the T column for all nodes.
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
                raise KeyError(
                    f"Node {n} missing required attributes 'xyz' and 'radius'"
                )

        # Build a spanning forest per connected component with DFS, choosing
        # a terminal (degree==1) node as root when available. Create a DFS
        # visitation order to determine SWC indices sequentially starting at 1.
        parents_orig: dict[int, int] = {}
        tree_edges: set[frozenset[int]] = set()
        comp_roots: list[int] = []
        comp_orders: list[list[int]] = []

        # Sort components deterministically by min node id
        components = [
            sorted(int(x) for x in comp) for comp in nx.connected_components(self)
        ]
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
        for u, v in self.edges():
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
            entries.append(
                (int(nid), int(tag), xyz[0], xyz[1], xyz[2], r, int(parent_id))
            )

        # Process extra edges by duplicating one endpoint and attaching to the other
        cycle_annotations: list[tuple[int, int]] = (
            []
        )  # (duplicate_swc_id, original_swc_id)
        for u, v in extra_edges:
            # Choose which node to duplicate: prefer higher degree (branching)
            deg_u = self.degree[u]
            deg_v = self.degree[v]
            dup_orig = v if deg_v >= deg_u else u
            other_orig = u if dup_orig == v else v

            xyz = np.asarray(self.nodes[dup_orig]["xyz"], dtype=float).reshape(3)
            r = float(self.nodes[dup_orig].get("radius", 0.0))
            dup_swc = int(next_index)
            parent_swc = int(new_id.get(other_orig, -1))
            entries.append((dup_swc, int(tag), xyz[0], xyz[1], xyz[2], r, parent_swc))
            # Record annotation using SWC indices
            cycle_annotations.append((dup_swc, int(new_id.get(dup_orig, dup_swc))))
            next_index += 1

        # Compose SWC text
        lines: list[str] = []
        lines.append("# generated by mcf2swc SkeletonGraph.to_swc_file")
        lines.append(
            f"# dfs_roots={' '.join(str(new_id.get(r, r)) for r in comp_roots)}"
        )
        lines.append(
            f"# nodes={self.number_of_nodes()} extra_edges={len(extra_edges)} duplicates={len(cycle_annotations)}"
        )
        lines.append(f"# tag={int(tag)}")
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

"""
Graph-based skeleton handler.

Provides a `SkeletonGraph` class that inherits from networkx.Graph to:
- Represent skeleton as a graph with xyz coordinates on each node
- Load from polylines array or polylines text format: `N x1 y1 z1 x2 y2 z2 ...`
- Identify terminal nodes (degree 1) and branch nodes (degree 3+)
- Every point from input polylines becomes a node in the graph

Note: This class represents skeleton topology as a graph where:
- Nodes have 'pos' attribute with (x, y, z) coordinates
- Edges connect consecutive nodes along polylines
- Terminal nodes have degree 1
- Branch nodes have degree 3+
- Continuation nodes have degree 2
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Set

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class SkeletonGraph(nx.Graph):
    """
    Graph-based skeleton representation with xyz coordinates on nodes.

    Inherits from networkx.Graph. Each node has a 'pos' attribute storing
    (x, y, z) coordinates. Edges represent connections between consecutive
    points along polylines.

    Every point from input polylines becomes a node.
    Endpoints within tolerance are merged into single nodes.

    Terminal nodes (degree 1) are isolated endpoints.
    Branch nodes (degree 3+) are where multiple branches meet.
    Continuation nodes (degree 2) are intermediate points along branches.
    """

    def __init__(self, tolerance: float = 1e-6, **attr):
        """
        Initialize a SkeletonGraph.

        Args:
            tolerance: Distance threshold for merging nearby endpoints
            **attr: Additional graph attributes
        """
        super().__init__(**attr)
        self.graph["tolerance"] = tolerance
        self._next_node_id = 0

    @classmethod
    def from_polylines(
        cls,
        polylines: Sequence[np.ndarray],
        tolerance: float = 1e-6,
    ) -> "SkeletonGraph":
        """
        Create a SkeletonGraph from a list of polyline arrays.

        Every point in the polylines becomes a node in the graph.
        Consecutive points are connected by edges.
        Endpoints within tolerance are merged into a single node.

        Args:
            polylines: List of (N_i, 3) arrays representing polylines
            tolerance: Distance threshold for merging nearby endpoints

        Returns:
            SkeletonGraph instance
        """
        graph = cls(tolerance=tolerance)

        if not polylines:
            return graph

        # Step 1: Create nodes for all points in all polylines
        point_to_node = {}  # Maps (poly_idx, point_idx) -> node_id
        endpoints = []  # List of (node_id, poly_idx, point_idx, coord)

        for poly_idx, pl in enumerate(polylines):
            if len(pl) == 0:
                continue

            for point_idx, coord in enumerate(pl):
                node_id = graph._get_next_node_id()
                graph.add_node(node_id, pos=np.array(coord, dtype=float))
                point_to_node[(poly_idx, point_idx)] = node_id

                # Track endpoints (first and last points of each polyline)
                if point_idx == 0 or point_idx == len(pl) - 1:
                    endpoints.append((node_id, poly_idx, point_idx, coord))

        # Step 2: Merge endpoints that are within tolerance
        endpoint_groups = []
        used = set()

        for i, (node_i, poly_i, pt_i, coord_i) in enumerate(endpoints):
            if i in used:
                continue

            # Find all endpoints close to this one
            group = [(node_i, poly_i, pt_i, coord_i)]
            for j, (node_j, poly_j, pt_j, coord_j) in enumerate(endpoints):
                if i != j and j not in used:
                    dist = np.linalg.norm(coord_i - coord_j)
                    if dist < tolerance:
                        group.append((node_j, poly_j, pt_j, coord_j))

            # Mark all as used
            for idx in range(len(endpoints)):
                if any(endpoints[idx][0] == node_id for node_id, _, _, _ in group):
                    used.add(idx)

            endpoint_groups.append(group)

        # Merge nodes in each group
        for group in endpoint_groups:
            if len(group) == 1:
                continue

            # Use the first node as the merged node
            merged_node = group[0][0]

            # Compute centroid position for the merged node
            coords = np.array([item[3] for item in group])
            merged_pos = coords.mean(axis=0)
            graph.nodes[merged_node]["pos"] = merged_pos

            # Map all other nodes to the merged node
            for node_id, poly_idx, point_idx, _ in group[1:]:
                point_to_node[(poly_idx, point_idx)] = merged_node
                # Remove the redundant node
                if node_id in graph:
                    graph.remove_node(node_id)

        # Step 3: Add edges connecting consecutive points in each polyline
        for poly_idx, pl in enumerate(polylines):
            if len(pl) < 2:
                continue

            for point_idx in range(len(pl) - 1):
                node_u = point_to_node.get((poly_idx, point_idx))
                node_v = point_to_node.get((poly_idx, point_idx + 1))

                if node_u is not None and node_v is not None and node_u != node_v:
                    # Compute edge length
                    pos_u = graph.nodes[node_u]["pos"]
                    pos_v = graph.nodes[node_v]["pos"]
                    length = float(np.linalg.norm(pos_v - pos_u))

                    graph.add_edge(
                        node_u,
                        node_v,
                        polyline_idx=poly_idx,
                        segment_idx=point_idx,
                        length=length,
                    )

        return graph

    @classmethod
    def from_txt(cls, path: str, tolerance: float = 1e-6) -> "SkeletonGraph":
        """
        Load a SkeletonGraph from a file.

        Supports two formats:
        1. GraphML format (.graphml or .xml extension) - native graph format
        2. Legacy polylines format (.polylines.txt) - for backward compatibility

        Args:
            path: Path to the skeleton file
            tolerance: Distance threshold for merging nearby endpoints (polylines format only)

        Returns:
            SkeletonGraph instance
        """
        import networkx as nx

        # Check file extension to determine format
        if path.endswith(".graphml") or path.endswith(".xml"):
            # Load from GraphML format
            G = nx.read_graphml(path)

            # Create SkeletonGraph and copy data
            graph = cls(tolerance=tolerance)

            # Add nodes with positions
            for node_id in G.nodes():
                node_data = G.nodes[node_id]
                # Convert node_id back to int
                node_int = int(node_id)

                # Parse position from string format "x,y,z"
                pos_str = node_data.get("pos", "0,0,0")
                pos = np.array([float(x) for x in pos_str.split(",")], dtype=float)

                graph.add_node(node_int, pos=pos)

            # Add edges with metadata
            for u, v, data in G.edges(data=True):
                u_int = int(u)
                v_int = int(v)

                edge_data = {}
                if "length" in data:
                    edge_data["length"] = float(data["length"])
                if "polyline_idx" in data:
                    edge_data["polyline_idx"] = int(data["polyline_idx"])
                if "segment_idx" in data:
                    edge_data["segment_idx"] = int(data["segment_idx"])

                graph.add_edge(u_int, v_int, **edge_data)

            # Set graph-level tolerance
            graph.graph["tolerance"] = tolerance

            return graph
        else:
            # Legacy polylines format
            polylines = []

            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) < 4:  # Need at least N and one (x,y,z) coordinate
                        continue

                    try:
                        n = int(parts[0])
                        coords = [float(x) for x in parts[1:]]

                        if len(coords) != n * 3:
                            continue

                        # Reshape into (N, 3) array
                        pl = np.array(coords).reshape(n, 3)
                        polylines.append(pl)

                    except (ValueError, IndexError):
                        continue

            return cls.from_polylines(polylines, tolerance=tolerance)

    def _get_next_node_id(self) -> int:
        """Get the next available node ID."""
        node_id = self._next_node_id
        self._next_node_id += 1
        return node_id

    # ---------------------------------------------------------------------
    # Node classification
    # ---------------------------------------------------------------------
    def get_terminal_nodes(self) -> Set[int]:
        """
        Get all terminal nodes (degree 1 - isolated endpoints).

        Returns:
            Set of node IDs that are terminal nodes
        """
        return {node for node in self.nodes() if self.degree(node) == 1}

    def get_branch_nodes(self) -> Set[int]:
        """
        Get all branch nodes (degree 3+ - where multiple branches meet).

        Returns:
            Set of node IDs that are branch nodes
        """
        return {node for node in self.nodes() if self.degree(node) >= 3}

    def get_continuation_nodes(self) -> Set[int]:
        """
        Get all continuation nodes (degree 2 - intermediate points).

        Returns:
            Set of node IDs that are continuation nodes
        """
        return {node for node in self.nodes() if self.degree(node) == 2}

    def detect_branch_points(self, tolerance: float = 1e-6) -> dict:
        """
        Detect branch points and endpoints in the skeleton.

        Note:
            `SkeletonGraph` already merges endpoints within tolerance on import.
            The `tolerance` argument is accepted for API compatibility.

        Args:
            tolerance: Unused; kept for compatibility.

        Returns:
            Dictionary containing:
                - 'branch_points': List of node IDs for branch nodes
                - 'endpoints': List of node IDs for terminal nodes
                - 'branch_locations': List of 3D coordinates of branch nodes
                - 'endpoint_locations': List of 3D coordinates of terminal nodes
        """
        _ = tolerance
        branch_nodes = sorted(self.get_branch_nodes())
        terminal_nodes = sorted(self.get_terminal_nodes())
        return {
            "branch_points": branch_nodes,
            "endpoints": terminal_nodes,
            "branch_locations": [self.get_node_position(n) for n in branch_nodes],
            "endpoint_locations": [self.get_node_position(n) for n in terminal_nodes],
        }

    def get_branch_point_indices(self, tolerance: float = 1e-6) -> Set[int]:
        """Return the set of branch node IDs.

        Args:
            tolerance: Unused; kept for compatibility.

        Returns:
            Set of node IDs.
        """
        _ = tolerance
        return set(self.get_branch_nodes())

    def get_true_endpoint_indices(self, tolerance: float = 1e-6) -> Set[int]:
        """Return the set of terminal node IDs.

        Args:
            tolerance: Unused; kept for compatibility.

        Returns:
            Set of node IDs.
        """
        _ = tolerance
        return set(self.get_terminal_nodes())

    def build_graph(self, tolerance: float = 1e-6) -> nx.Graph:
        """
        Build a networkx graph representation with node type annotations.

        This is primarily a compatibility helper for tests. The returned graph is a
        plain `nx.Graph` (not a `SkeletonGraph`).

        Args:
            tolerance: Unused; kept for compatibility.

        Returns:
            A `nx.Graph` with nodes annotated with:
                - 'pos': (x, y, z)
                - 'type': 'endpoint' | 'branch' | 'continuation'
        """
        _ = tolerance
        G = nx.Graph()
        for n in self.nodes():
            if self.is_branch_node(n):
                node_type = "branch"
            elif self.is_terminal_node(n):
                node_type = "endpoint"
            else:
                node_type = "continuation"
            pos = self.get_node_position(n)
            G.add_node(n, pos=pos, type=node_type)
        for u, v, data in self.edges(data=True):
            G.add_edge(u, v, **(data or {}))
        return G

    def is_terminal_node(self, node: int) -> bool:
        """Check if a node is a terminal node (degree 1)."""
        return self.degree(node) == 1

    def is_branch_node(self, node: int) -> bool:
        """Check if a node is a branch node (degree 3+)."""
        return self.degree(node) >= 3

    def is_continuation_node(self, node: int) -> bool:
        """Check if a node is a continuation node (degree 2)."""
        return self.degree(node) == 2

    # ---------------------------------------------------------------------
    # Coordinate access and manipulation
    # ---------------------------------------------------------------------
    def get_node_position(self, node: int) -> np.ndarray:
        """
        Get the (x, y, z) position of a node.

        Args:
            node: Node ID

        Returns:
            (3,) array with xyz coordinates
        """
        return np.array(self.nodes[node]["pos"])

    def set_node_position(self, node: int, pos: np.ndarray) -> None:
        """
        Set the (x, y, z) position of a node.

        Args:
            node: Node ID
            pos: (3,) array with xyz coordinates
        """
        self.nodes[node]["pos"] = np.array(pos, dtype=float)

    def get_all_positions(self) -> np.ndarray:
        """
        Get positions of all nodes as an array.

        Returns:
            (N, 3) array where N is the number of nodes
        """
        if self.number_of_nodes() == 0:
            return np.zeros((0, 3), dtype=float)

        positions = []
        for node in sorted(self.nodes()):
            positions.append(self.get_node_position(node))

        return np.array(positions)

    def set_all_positions(self, positions: np.ndarray) -> None:
        """
        Set positions of all nodes from an array.

        Args:
            positions: (N, 3) array where N is the number of nodes
        """
        if positions.shape[0] != self.number_of_nodes():
            raise ValueError(
                f"Position array has {positions.shape[0]} rows but graph has "
                f"{self.number_of_nodes()} nodes"
            )

        for i, node in enumerate(sorted(self.nodes())):
            self.set_node_position(node, positions[i])

    def _edge_length(self, u: int, v: int) -> float:
        data = self.get_edge_data(u, v) or {}
        length = data.get("length")
        if length is not None:
            return float(length)
        pu = self.get_node_position(u)
        pv = self.get_node_position(v)
        return float(np.linalg.norm(pv - pu))

    def _trace_from_terminal(self, start: int) -> tuple[int, list[int], float]:
        """Trace from a terminal node until the next non-continuation node.

        Returns (end_node, path_nodes, length).
        """
        if start not in self:
            return start, [start], 0.0
        if self.degree(start) != 1:
            return start, [start], 0.0

        path = [start]
        prev = None
        current = start
        length = 0.0

        while True:
            nbrs = list(self.neighbors(current))
            if prev is not None:
                nbrs = [n for n in nbrs if n != prev]
            if len(nbrs) == 0:
                break
            nxt = nbrs[0]
            length += self._edge_length(current, nxt)
            prev, current = current, nxt
            path.append(current)
            if self.degree(current) != 2:
                break
        return current, path, length

    def prune_short_branches(
        self,
        min_length: Optional[float] = None,
        min_length_percentile: Optional[float] = None,
        tolerance: float = 1e-6,
        iterative: bool = True,
        verbose: bool = False,
    ) -> "SkeletonGraph":
        """Remove short terminal branches.

        A terminal branch is a path from a terminal node (degree 1) to the next
        non-continuation node (degree != 2). If the path ends at a branch node
        (degree 3+) and its geometric length is below the threshold, it is removed.
        Isolated components that connect terminal-to-terminal with no branch nodes
        are removed regardless of length.

        Length is computed from edge lengths (or node coordinates if missing), so
        "short" refers to *geometric* length, not node count.
        """
        _ = tolerance
        if self.number_of_nodes() == 0:
            return self.copy_skeleton()

        if min_length is None and min_length_percentile is None:
            raise ValueError("Must specify either min_length or min_length_percentile")

        # Determine threshold from the original graph
        original = self.copy_skeleton()
        original_branch_lengths = list(original.compute_branch_lengths().values())
        if len(original_branch_lengths) == 0:
            # Nothing to prune
            return self.copy_skeleton()

        if min_length is not None:
            threshold = float(min_length)
        else:
            threshold = float(
                np.percentile(original_branch_lengths, min_length_percentile)
            )

        if verbose:
            logger.info("Pruning branches with length < %.4f", threshold)

        current = self.copy_skeleton()
        while True:
            terminal_nodes = sorted(
                [n for n in current.nodes() if current.degree(n) == 1]
            )
            nodes_to_remove: Set[int] = set()
            visited_terminals: Set[int] = set()

            for t in terminal_nodes:
                if t in visited_terminals or t not in current:
                    continue

                end, path, length = current._trace_from_terminal(t)
                if len(path) <= 1:
                    visited_terminals.add(t)
                    continue

                visited_terminals.add(t)
                if end != t and current.degree(end) == 1:
                    visited_terminals.add(end)

                is_isolated = end != t and current.degree(end) == 1
                ends_at_branch = end != t and current.degree(end) >= 3

                should_remove = False
                if is_isolated:
                    should_remove = True
                elif ends_at_branch and length < threshold:
                    should_remove = True

                if not should_remove:
                    continue

                # Remove everything except the branch node when applicable
                if ends_at_branch:
                    nodes_to_remove.update(path[:-1])
                else:
                    nodes_to_remove.update(path)

            if len(nodes_to_remove) == 0:
                break

            current.remove_nodes_from([n for n in nodes_to_remove if n in current])

            if not iterative:
                break

        return current

    def prune_short_branches_inplace(
        self,
        min_length: Optional[float] = None,
        min_length_percentile: Optional[float] = None,
        tolerance: float = 1e-6,
        verbose: bool = False,
    ) -> int:
        """In-place version of `prune_short_branches`.

        Returns:
            Number of nodes removed.
        """
        before = self.number_of_nodes()
        pruned = self.prune_short_branches(
            min_length=min_length,
            min_length_percentile=min_length_percentile,
            tolerance=tolerance,
            iterative=True,
            verbose=verbose,
        )
        self.clear()
        self.add_nodes_from(pruned.nodes(data=True))
        self.add_edges_from(pruned.edges(data=True))
        self.graph.update(pruned.graph)
        self._next_node_id = getattr(pruned, "_next_node_id", self._next_node_id)
        return before - self.number_of_nodes()

    # ---------------------------------------------------------------------
    # Basic properties
    # ---------------------------------------------------------------------
    def total_points(self) -> int:
        """
        Get total number of points in the skeleton.

        Since every point is a node, this is just the number of nodes.

        Returns:
            Total number of points in the skeleton
        """
        return self.number_of_nodes()

    def bounds(self) -> Optional[dict]:
        """
        Get bounding box of all node positions.

        Returns:
            Dictionary with 'x', 'y', 'z' keys, each containing (min, max) tuple,
            or None if graph is empty
        """
        if self.number_of_nodes() == 0:
            return None

        positions = self.get_all_positions()
        lo = positions.min(axis=0)
        hi = positions.max(axis=0)

        return {
            "x": (float(lo[0]), float(hi[0])),
            "y": (float(lo[1]), float(hi[1])),
            "z": (float(lo[2]), float(hi[2])),
        }

    def centroid(self) -> Optional[np.ndarray]:
        """
        Get centroid of all node positions.

        Returns:
            (3,) array with centroid coordinates, or None if graph is empty
        """
        if self.number_of_nodes() == 0:
            return None

        positions = self.get_all_positions()
        return positions.mean(axis=0)

    # ---------------------------------------------------------------------
    # Conversion
    # ---------------------------------------------------------------------
    def to_polylines(self) -> List[np.ndarray]:
        """
        Convert the graph back to a list of polyline arrays.

        Reconstructs polylines by grouping edges with the same polyline_idx
        and ordering them by segment_idx, properly handling branch points.

        Returns:
            List of (N_i, 3) arrays representing polylines
        """
        if self.number_of_edges() == 0:
            return []

        # Group edges by polyline_idx
        polyline_edges = {}  # Maps polyline_idx -> list of (segment_idx, u, v)

        for u, v, data in self.edges(data=True):
            poly_idx = data.get("polyline_idx")
            seg_idx = data.get("segment_idx")

            if poly_idx is not None and seg_idx is not None:
                if poly_idx not in polyline_edges:
                    polyline_edges[poly_idx] = []
                polyline_edges[poly_idx].append((seg_idx, u, v))

        # Reconstruct each polyline
        polylines = []

        for poly_idx in sorted(polyline_edges.keys()):
            edges = sorted(polyline_edges[poly_idx], key=lambda x: x[0])

            if not edges:
                continue

            # Build polyline from ordered edges by following the chain
            points = []

            # Start with the first edge
            _, u_first, v_first = edges[0]
            current_node = u_first
            points.append(self.get_node_position(current_node))

            # Follow the chain of edges
            for seg_idx, u, v in edges:
                # Determine which node is next in the chain
                if u == current_node:
                    next_node = v
                elif v == current_node:
                    next_node = u
                else:
                    # Edge doesn't connect to current node - this shouldn't happen
                    # but if it does, just add both nodes
                    points.append(self.get_node_position(u))
                    next_node = v

                points.append(self.get_node_position(next_node))
                current_node = next_node

            polylines.append(np.array(points))

        return polylines

    def to_txt(self, path: str) -> None:
        """
        Save the skeleton to a file.

        Saves in GraphML format (.graphml) which preserves all graph structure,
        node positions, and edge metadata. This is the native format for SkeletonGraph.

        For legacy polylines format, use to_polylines() and save manually.

        Args:
            path: Output file path (will use .graphml extension if not provided)
        """
        import networkx as nx

        # Ensure .graphml extension
        if not path.endswith(".graphml"):
            if path.endswith(".txt"):
                path = path.replace(".txt", ".graphml")
            else:
                path = path + ".graphml"

        # Create a copy of the graph for serialization
        G = nx.Graph()

        # Add nodes with position as string (GraphML requires string attributes)
        for node in self.nodes():
            pos = self.get_node_position(node)
            pos_str = f"{pos[0]},{pos[1]},{pos[2]}"
            G.add_node(str(node), pos=pos_str)

        # Add edges with metadata
        for u, v, data in self.edges(data=True):
            edge_attrs = {}
            if "length" in data:
                edge_attrs["length"] = str(data["length"])
            if "polyline_idx" in data:
                edge_attrs["polyline_idx"] = str(data["polyline_idx"])
            if "segment_idx" in data:
                edge_attrs["segment_idx"] = str(data["segment_idx"])

            G.add_edge(str(u), str(v), **edge_attrs)

        # Write to GraphML format
        nx.write_graphml(G, path)

    # ---------------------------------------------------------------------
    # Copy
    # ---------------------------------------------------------------------
    def copy_skeleton(self) -> "SkeletonGraph":
        """
        Create a deep copy of the skeleton graph.

        Returns:
            New SkeletonGraph instance with copied data
        """
        # Create new graph with same tolerance
        new_graph = SkeletonGraph(tolerance=self.graph.get("tolerance", 1e-6))

        # Copy nodes with positions
        for node, data in self.nodes(data=True):
            pos = np.array(data["pos"])
            new_graph.add_node(node, pos=pos.copy())

        # Copy edges with data
        for u, v, data in self.edges(data=True):
            edge_data = dict(data)
            new_graph.add_edge(u, v, **edge_data)

        # Update node ID counter
        new_graph._next_node_id = self._next_node_id

        return new_graph

    # ---------------------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------------------
    def get_statistics(self) -> dict:
        """
        Get statistics about the skeleton graph.

        Returns:
            Dictionary with various statistics
        """
        terminal_nodes = self.get_terminal_nodes()
        branch_nodes = self.get_branch_nodes()
        continuation_nodes = self.get_continuation_nodes()

        edge_lengths = [data.get("length", 0.0) for _, _, data in self.edges(data=True)]

        stats = {
            "num_nodes": self.number_of_nodes(),
            "num_edges": self.number_of_edges(),
            "num_terminal_nodes": len(terminal_nodes),
            "num_branch_nodes": len(branch_nodes),
            "num_continuation_nodes": len(continuation_nodes),
            "total_points": self.total_points(),
        }

        if edge_lengths:
            stats["total_length"] = sum(edge_lengths)
            stats["mean_edge_length"] = np.mean(edge_lengths)
            stats["min_edge_length"] = min(edge_lengths)
            stats["max_edge_length"] = max(edge_lengths)

        return stats

    def __repr__(self) -> str:
        """String representation of the skeleton graph."""
        stats = self.get_statistics()
        return (
            f"SkeletonGraph(nodes={stats['num_nodes']}, "
            f"edges={stats['num_edges']}, "
            f"terminals={stats['num_terminal_nodes']}, "
            f"branches={stats['num_branch_nodes']})"
        )

    # ---------------------------------------------------------------------
    # Resampling
    # ---------------------------------------------------------------------
    def resample(self, spacing: float) -> "SkeletonGraph":
        """
        Resample the skeleton to have approximately uniform spacing between nodes.

        Works directly on graph edges, subdividing long edges and preserving topology.

        Args:
            spacing: Target distance between consecutive nodes

        Returns:
            New SkeletonGraph with resampled nodes
        """
        import networkx as nx

        new_graph = SkeletonGraph(tolerance=self.graph.get("tolerance", 1e-6))

        # Copy all existing nodes first
        node_mapping = {}  # old_node -> new_node
        for node in self.nodes():
            new_node = new_graph._get_next_node_id()
            new_graph.add_node(new_node, pos=self.get_node_position(node).copy())
            node_mapping[node] = new_node

        # Process each edge, subdividing if necessary
        for u, v, data in self.edges(data=True):
            pos_u = self.get_node_position(u)
            pos_v = self.get_node_position(v)

            # Calculate edge length
            edge_length = np.linalg.norm(pos_v - pos_u)

            # Determine number of segments needed
            n_segments = max(1, int(np.ceil(edge_length / spacing)))

            if n_segments == 1:
                # Edge is short enough, just connect directly
                new_u = node_mapping[u]
                new_v = node_mapping[v]
                length = float(edge_length)
                new_graph.add_edge(
                    new_u,
                    new_v,
                    length=length,
                    polyline_idx=data.get("polyline_idx"),
                    segment_idx=data.get("segment_idx"),
                )
            else:
                # Subdivide edge
                prev_node = node_mapping[u]

                for i in range(1, n_segments):
                    # Interpolate position
                    t = i / n_segments
                    new_pos = pos_u + t * (pos_v - pos_u)

                    # Create intermediate node
                    intermediate_node = new_graph._get_next_node_id()
                    new_graph.add_node(intermediate_node, pos=new_pos)

                    # Add edge from previous to intermediate
                    seg_length = float(
                        np.linalg.norm(
                            new_graph.get_node_position(intermediate_node)
                            - new_graph.get_node_position(prev_node)
                        )
                    )
                    new_graph.add_edge(
                        prev_node,
                        intermediate_node,
                        length=seg_length,
                        polyline_idx=data.get("polyline_idx"),
                        segment_idx=data.get("segment_idx"),
                    )

                    prev_node = intermediate_node

                # Add final segment to v
                new_v = node_mapping[v]
                seg_length = float(
                    np.linalg.norm(
                        new_graph.get_node_position(new_v)
                        - new_graph.get_node_position(prev_node)
                    )
                )
                new_graph.add_edge(
                    prev_node,
                    new_v,
                    length=seg_length,
                    polyline_idx=data.get("polyline_idx"),
                    segment_idx=data.get("segment_idx"),
                )

        return new_graph

    # ---------------------------------------------------------------------
    # Mesh surface projection
    # ---------------------------------------------------------------------
    def snap_to_mesh_surface(
        self,
        mesh,
        project_outside_only: bool = True,
        max_distance: Optional[float] = None,
    ) -> tuple:
        """
        Project node positions to the nearest surface point on mesh.

        Args:
            mesh: trimesh.Trimesh object
            project_outside_only: If True, only project points outside the mesh
            max_distance: If provided, only move points beyond this distance from surface

        Returns:
            (n_moved, mean_move_distance) tuple
        """
        if mesh is None or len(getattr(mesh, "vertices", [])) == 0:
            return 0, 0.0

        if self.number_of_nodes() == 0:
            return 0, 0.0

        # Get all node positions
        positions = self.get_all_positions()

        # Determine which points to project
        use_mask = None
        if project_outside_only:
            try:
                from trimesh.proximity import signed_distance

                d = signed_distance(mesh, positions)
                use_mask = d > 0  # outside
            except Exception:
                use_mask = None

        # Find closest points on mesh
        try:
            from trimesh.proximity import closest_point

            closest_positions, distances, _ = closest_point(mesh, positions)
        except Exception:
            # Fallback: KDTree over vertices only
            vertices = np.asarray(mesh.vertices, dtype=float)
            if vertices.size == 0:
                return 0, 0.0
            from scipy.spatial import cKDTree

            tree = cKDTree(vertices)
            distances, idx = tree.query(positions, k=1)
            closest_positions = vertices[idx]

        # Apply masks
        if use_mask is None:
            mask = np.ones(positions.shape[0], dtype=bool)
        else:
            mask = use_mask

        if max_distance is not None:
            mask = mask & (distances >= float(max_distance))

        # Update positions
        moved = 0
        total_move = 0.0

        for i, node in enumerate(sorted(self.nodes())):
            if mask[i]:
                self.set_node_position(node, closest_positions[i])
                moved += 1
                total_move += distances[i]

        mean_move = (total_move / moved) if moved > 0 else 0.0
        return moved, mean_move

    # ---------------------------------------------------------------------
    # Branch length computation
    # ---------------------------------------------------------------------
    def compute_branch_lengths(self) -> dict:
        """
        Compute the length of each branch (path between terminal/branch nodes).

        Returns:
            Dictionary mapping (start_node, end_node) -> length
        """
        branch_lengths = {}

        # Get terminal and branch nodes
        terminal_nodes = self.get_terminal_nodes()
        branch_nodes = self.get_branch_nodes()
        special_nodes = terminal_nodes | branch_nodes

        # For each special node, trace paths to other special nodes
        for start_node in special_nodes:
            # Use BFS to find paths to other special nodes
            visited = {start_node}
            queue = [(start_node, [start_node], 0.0)]  # (node, path, length)

            while queue:
                current, path, length = queue.pop(0)

                for neighbor in self.neighbors(current):
                    if neighbor in visited:
                        continue

                    # Get edge length
                    edge_data = self.get_edge_data(current, neighbor)
                    edge_length = edge_data.get("length", 0.0) if edge_data else 0.0
                    new_length = length + edge_length
                    new_path = path + [neighbor]

                    # If we reached another special node, record the branch
                    if neighbor in special_nodes and neighbor != start_node:
                        key = tuple(sorted([start_node, neighbor]))
                        if key not in branch_lengths:
                            branch_lengths[key] = new_length
                    else:
                        # Continue searching
                        visited.add(neighbor)
                        queue.append((neighbor, new_path, new_length))

        return branch_lengths

    def get_total_length(self) -> float:
        """
        Get the total length of all edges in the skeleton.

        Returns:
            Total length
        """
        return sum(data.get("length", 0.0) for _, _, data in self.edges(data=True))


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def _resample_polyline(pl: np.ndarray, spacing: float) -> np.ndarray:
    """
    Resample a polyline at approximately constant arc-length spacing.

    Includes the first and last vertex; inserts intermediate points every
    multiple of `spacing` along cumulative arclength.

    Args:
        pl: (N, 3) array of points
        spacing: Target spacing between points

    Returns:
        (M, 3) array of resampled points
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

"""
Test branch point detection on TS2 skeleton.
"""

import numpy as np
from mcf2swc import PolylinesSkeleton

# Load TS2 skeleton
print("Loading TS2 skeleton...")
skeleton = PolylinesSkeleton.from_txt(
    "data/mcf_skeletons/TS2_qst0.6_mcst5.polylines.txt"
)

print(f"\nSkeleton properties:")
print(f"  Number of polylines: {len(skeleton.polylines)}")
print(f"  Total points: {skeleton.total_points()}")
for i, pl in enumerate(skeleton.polylines):
    print(f"  Polyline {i}: {len(pl)} points")

# Detect branch points with different tolerances
print("\n" + "=" * 70)
print("Testing branch point detection with tolerance=1e-6")
print("=" * 70)

topology = skeleton.detect_branch_points(tolerance=1e-6)

print(f"\nBranch points found: {len(topology['branch_points'])}")
print(f"True endpoints found: {len(topology['endpoints'])}")

print(f"\nBranch point details:")
for (poly_idx, point_idx), location in zip(
    topology["branch_points"], topology["branch_locations"]
):
    print(f"  Polyline {poly_idx}, point {point_idx}: {location}")

print(f"\nTrue endpoint details:")
for (poly_idx, point_idx), location in zip(
    topology["endpoints"], topology["endpoint_locations"]
):
    print(f"  Polyline {poly_idx}, point {point_idx}: {location}")

# Try with larger tolerance
print("\n" + "=" * 70)
print("Testing with tolerance=0.1 (larger tolerance)")
print("=" * 70)

topology2 = skeleton.detect_branch_points(tolerance=0.1)

print(f"\nBranch points found: {len(topology2['branch_points'])}")
print(f"True endpoints found: {len(topology2['endpoints'])}")

# Build graph
print("\n" + "=" * 70)
print("Building networkx graph representation")
print("=" * 70)

graph = skeleton.build_graph(tolerance=0.1)

print(f"\nGraph properties:")
print(f"  Nodes: {graph.number_of_nodes()}")
print(f"  Edges: {graph.number_of_edges()}")

print(f"\nNode details:")
for node, data in graph.nodes(data=True):
    print(f"  Node {node}: type={data['type']}, pos={data['pos']}")

print(f"\nEdge details:")
for u, v, data in graph.edges(data=True):
    print(
        f"  Edge {u}-{v}: polyline_idx={data['polyline_idx']}, "
        f"length={data['length']:.2f}, intermediate_points={len(data['points'])}"
    )

# Verify topology makes sense
print("\n" + "=" * 70)
print("Topology verification")
print("=" * 70)

branch_nodes = [n for n, d in graph.nodes(data=True) if d["type"] == "branch"]
endpoint_nodes = [n for n, d in graph.nodes(data=True) if d["type"] == "endpoint"]

print(f"\nBranch nodes: {len(branch_nodes)}")
print(f"Endpoint nodes: {len(endpoint_nodes)}")

print(f"\nDegree of each node:")
for node in graph.nodes():
    degree = graph.degree(node)
    node_type = graph.nodes[node]["type"]
    print(f"  Node {node} ({node_type}): degree={degree}")

# Expected: branch nodes should have degree >= 3, endpoint nodes should have degree 1
print(f"\nValidation:")
valid = True
for node in branch_nodes:
    if graph.degree(node) < 3:
        print(f"  WARNING: Branch node {node} has degree {graph.degree(node)} < 3")
        valid = False

for node in endpoint_nodes:
    if graph.degree(node) != 1:
        print(f"  WARNING: Endpoint node {node} has degree {graph.degree(node)} != 1")
        valid = False

if valid:
    print("  ✓ All nodes have expected degrees!")
else:
    print("  ✗ Some nodes have unexpected degrees")

# Test the convenience methods
print("\n" + "=" * 70)
print("Testing convenience methods")
print("=" * 70)

branch_indices = skeleton.get_branch_point_indices(tolerance=0.1)
endpoint_indices = skeleton.get_true_endpoint_indices(tolerance=0.1)

print(f"\nBranch point indices: {len(branch_indices)}")
for idx in sorted(branch_indices):
    print(f"  {idx}")

print(f"\nTrue endpoint indices: {len(endpoint_indices)}")
for idx in sorted(endpoint_indices):
    print(f"  {idx}")

# Count how many polyline endpoints are actually branch points
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)

total_polyline_endpoints = sum(2 if len(pl) > 1 else 1 for pl in skeleton.polylines)
print(f"\nTotal polyline endpoints: {total_polyline_endpoints}")
print(f"  Branch points: {len(branch_indices)}")
print(f"  True endpoints: {len(endpoint_indices)}")
print(
    f"\nThis means {len(branch_indices)}/{total_polyline_endpoints} polyline endpoints are branch points"
)
print(f"and {len(endpoint_indices)}/{total_polyline_endpoints} are true endpoints")

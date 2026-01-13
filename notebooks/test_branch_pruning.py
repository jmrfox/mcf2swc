"""
Test branch pruning on TS2 skeleton.
"""

import numpy as np
from mcf2swc import PolylinesSkeleton

# Load TS2 skeleton
print("Loading TS2 skeleton...")
skeleton = PolylinesSkeleton.from_txt(
    "data/mcf_skeletons/TS2_qst0.6_mcst5.polylines.txt"
)

print(f"\nOriginal skeleton:")
print(f"  Number of polylines: {len(skeleton.polylines)}")
print(f"  Total points: {skeleton.total_points()}")

# Compute and display branch lengths
print(f"\nBranch lengths:")
for i, pl in enumerate(skeleton.polylines):
    length = skeleton._compute_polyline_length(pl)
    print(f"  Polyline {i}: {len(pl)} points, length = {length:.2f}")

# Show topology before pruning
print("\n" + "=" * 70)
print("Topology before pruning:")
print("=" * 70)

topology_before = skeleton.detect_branch_points(tolerance=1e-6)
print(f"Branch points: {len(topology_before['branch_points'])}")
print(f"True endpoints: {len(topology_before['endpoints'])}")

# Test pruning with different thresholds
print("\n" + "=" * 70)
print("Testing pruning with min_length=10.0")
print("=" * 70)

pruned = skeleton.prune_short_branches(min_length=10.0, verbose=True)

print(f"\nAfter pruning:")
print(f"  Number of polylines: {len(pruned.polylines)}")
print(f"  Total points: {pruned.total_points()}")

print(f"\nRemaining branch lengths:")
for i, pl in enumerate(pruned.polylines):
    length = pruned._compute_polyline_length(pl)
    print(f"  Polyline {i}: {len(pl)} points, length = {length:.2f}")

# Show topology after pruning
print("\n" + "=" * 70)
print("Topology after pruning:")
print("=" * 70)

topology_after = pruned.detect_branch_points(tolerance=1e-6)
print(f"Branch points: {len(topology_after['branch_points'])}")
print(f"True endpoints: {len(topology_after['endpoints'])}")

print(f"\nBranch point details:")
for (poly_idx, point_idx), location in zip(
    topology_after["branch_points"], topology_after["branch_locations"]
):
    print(f"  Polyline {poly_idx}, point {point_idx}")

print(f"\nTrue endpoint details:")
for (poly_idx, point_idx), location in zip(
    topology_after["endpoints"], topology_after["endpoint_locations"]
):
    print(f"  Polyline {poly_idx}, point {point_idx}")

# Try percentile-based pruning
print("\n" + "=" * 70)
print("Testing pruning with min_length_percentile=20")
print("=" * 70)

pruned2 = skeleton.prune_short_branches(min_length_percentile=20, verbose=True)

print(f"\nAfter percentile-based pruning:")
print(f"  Number of polylines: {len(pruned2.polylines)}")

# Save pruned skeleton
print("\n" + "=" * 70)
print("Saving pruned skeleton...")
pruned.to_txt("data/mcf_skeletons/TS2_pruned.polylines.txt")
print("Saved to: data/mcf_skeletons/TS2_pruned.polylines.txt")

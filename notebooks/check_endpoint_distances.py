"""
Check distances between polyline endpoints to determine appropriate tolerance.
"""

import numpy as np
from mcf2swc import PolylinesSkeleton

# Load TS2 skeleton
skeleton = PolylinesSkeleton.from_txt(
    "data/mcf_skeletons/TS2_qst0.6_mcst5.polylines.txt"
)

# Collect all endpoints
endpoints = []
for poly_idx, pl in enumerate(skeleton.polylines):
    if len(pl) > 0:
        endpoints.append((poly_idx, 0, pl[0], "start"))
        if len(pl) > 1:
            endpoints.append((poly_idx, len(pl) - 1, pl[-1], "end"))

print(f"Total endpoints: {len(endpoints)}")
print("\nEndpoint coordinates:")
for poly_idx, point_idx, coord, pos in endpoints:
    print(f"  Polyline {poly_idx} {pos}: {coord}")

# Compute pairwise distances
print("\n" + "=" * 70)
print("Pairwise distances between endpoints:")
print("=" * 70)

for i in range(len(endpoints)):
    for j in range(i + 1, len(endpoints)):
        poly_i, pt_i, coord_i, pos_i = endpoints[i]
        poly_j, pt_j, coord_j, pos_j = endpoints[j]

        dist = np.linalg.norm(coord_i - coord_j)

        if dist < 1.0:  # Only show close pairs
            print(f"\nPolyline {poly_i} {pos_i} <-> Polyline {poly_j} {pos_j}")
            print(f"  Distance: {dist:.6f}")
            print(f"  Coords: {coord_i} <-> {coord_j}")

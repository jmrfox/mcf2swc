"""
Debug script to check why radial and tangent strategies give same results.
"""

import numpy as np
from mcf2swc import (
    example_mesh,
    PolylinesSkeleton,
    TraceOptions,
    build_traced_skeleton_graph,
)

# Create a simple test case
mesh = example_mesh("cylinder", radius=1.0, height=10.0)

# Polyline along the axis
polyline = np.array(
    [
        [0.0, 0.0, 2.0],
        [0.0, 0.0, 5.0],
        [0.0, 0.0, 8.0],
    ]
)
polylines = PolylinesSkeleton(polylines=[polyline])

# Test with tangent strategy
print("=" * 70)
print("Testing TANGENT strategy")
print("=" * 70)
options_tangent = TraceOptions(
    spacing=1.5, normal_strategy="tangent", radius_strategy="equivalent_area"
)
skeleton_tangent = build_traced_skeleton_graph(mesh, polylines, options=options_tangent)

print(f"\nNodes: {skeleton_tangent.number_of_nodes()}")
for nid in sorted(skeleton_tangent.nodes()):
    node = skeleton_tangent.nodes[nid]
    print(f"  Node {nid}: xyz={node['xyz']}, radius={node['radius']:.6f}")

# Test with radial strategy
print("\n" + "=" * 70)
print("Testing RADIAL strategy")
print("=" * 70)
options_radial = TraceOptions(
    spacing=1.5, normal_strategy="radial", radius_strategy="equivalent_area"
)
skeleton_radial = build_traced_skeleton_graph(mesh, polylines, options=options_radial)

print(f"\nNodes: {skeleton_radial.number_of_nodes()}")
for nid in sorted(skeleton_radial.nodes()):
    node = skeleton_radial.nodes[nid]
    print(f"  Node {nid}: xyz={node['xyz']}, radius={node['radius']:.6f}")

# Compare
print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)
radii_tangent = [
    skeleton_tangent.nodes[nid]["radius"] for nid in sorted(skeleton_tangent.nodes())
]
radii_radial = [
    skeleton_radial.nodes[nid]["radius"] for nid in sorted(skeleton_radial.nodes())
]

print(f"\nTangent radii: {radii_tangent}")
print(f"Radial radii:  {radii_radial}")
print(f"\nAre they identical? {np.allclose(radii_tangent, radii_radial)}")

if np.allclose(radii_tangent, radii_radial):
    print(
        "\n⚠️  WARNING: Radii are identical! This suggests radial strategy is not working."
    )
else:
    print("\n✓ Radii are different as expected.")

# Now let's manually test the _compute_radial_normal function
print("\n" + "=" * 70)
print("MANUAL TEST of _compute_radial_normal")
print("=" * 70)

from mcf2swc.trace import _compute_radial_normal
from trimesh.proximity import closest_point

# Test point along the axis
P = np.array([0.0, 0.0, 5.0])
tangent = np.array([0.0, 0.0, 1.0])  # Along z-axis

print(f"\nTest point P: {P}")
print(f"Tangent: {tangent}")

# Find nearest surface point
closest_pts, dist, _tri = closest_point(mesh, P.reshape(1, 3))
nearest_pt = closest_pts[0]
print(f"Nearest surface point: {nearest_pt}")
print(f"Distance to surface: {dist[0]:.6f}")

# Compute radial vector
radial = nearest_pt - P
radial_norm = np.linalg.norm(radial)
print(f"Radial vector: {radial}")
print(f"Radial norm: {radial_norm:.6f}")

if radial_norm > 1e-12:
    radial_unit = radial / radial_norm
    print(f"Radial unit: {radial_unit}")

    # Compute cross product
    normal = np.cross(tangent, radial_unit)
    normal_norm = np.linalg.norm(normal)
    print(f"Cross product (tangent × radial): {normal}")
    print(f"Cross product norm: {normal_norm:.6f}")

    if normal_norm > 1e-12:
        normal_unit = normal / normal_norm
        print(f"Normal (unit): {normal_unit}")
        print(
            f"\n✓ Normal is different from tangent: {not np.allclose(normal_unit, tangent)}"
        )
    else:
        print(f"\n⚠️  Cross product is zero! Tangent and radial are parallel.")
else:
    print(f"\n⚠️  Point is on or very close to surface!")

# Call the actual function
computed_normal = _compute_radial_normal(P, tangent, mesh)
print(f"\nComputed normal from function: {computed_normal}")
print(f"Is it same as tangent? {np.allclose(computed_normal, tangent)}")

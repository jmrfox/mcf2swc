"""
Simple debug script with logging enabled.
"""

import logging
import numpy as np
from mcf2swc import (
    example_mesh,
    PolylinesSkeleton,
    TraceOptions,
    build_traced_skeleton_graph,
)

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG, format="%(name)s - %(levelname)s - %(message)s"
)

# Create a simple test case
mesh = example_mesh("cylinder", radius=1.0, height=10.0)

# Polyline along the axis - just 2 points
polyline = np.array(
    [
        [0.0, 0.0, 5.0],
        [0.0, 0.0, 6.0],
    ]
)
polylines = PolylinesSkeleton(polylines=[polyline])

print("=" * 70)
print("Testing RADIAL strategy with debug logging")
print("=" * 70)

options_radial = TraceOptions(
    spacing=0.5, normal_strategy="radial", radius_strategy="equivalent_area"
)

skeleton_radial = build_traced_skeleton_graph(mesh, polylines, options=options_radial)

print(f"\nNodes created: {skeleton_radial.number_of_nodes()}")
for nid in sorted(skeleton_radial.nodes()):
    node = skeleton_radial.nodes[nid]
    print(f"  Node {nid}: xyz={node['xyz']}, radius={node['radius']:.6f}")

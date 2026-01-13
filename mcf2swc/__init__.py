"""
mcf2swc

A lightweight toolkit for converting mesh cross-sections and skeleton polyline guidance
into SWC models.

Terminology:
- "skeleton": The mesh centroid (polylines format, result of MCF calculation) without radii
- "SWC model" or "swc": Skeleton with radii information attached to each node

Public API:
- MeshManager
- PolylinesSkeleton
- SWCModel (from swctools)
- TraceOptions, build_traced_skeleton_graph
"""

from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError, version

# Package version reported from installed metadata (fallback for editable/dev installs)
try:
    __version__ = version("mcf2swc")
except PackageNotFoundError:  # pragma: no cover - best-effort in dev
    __version__ = "0.1.0"

# Avoid "No handler found" warnings for library users; applications can configure logging.
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Re-export primary classes and functions for convenient access at package level
from swctools import SWCModel  # noqa: E402

from .mesh import MeshManager, example_mesh  # noqa: E402
from .polylines import PolylinesSkeleton  # noqa: E402
from .trace import (  # noqa: E402
    TraceOptions,
    build_traced_skeleton_graph,
)
from .radius_optimizer import (  # noqa: E402
    OptimizerOptions,
    RadiusOptimizer,
    optimize_skeleton_radii,
)
from .radius_optimizer_local import (  # noqa: E402
    LocalOptimizerOptions,
    LocalRadiusOptimizer,
)
from .skeleton_optimizer import (  # noqa: E402
    SkeletonOptimizer,
    SkeletonOptimizerOptions,
)

__all__ = [
    "__version__",
    # Mesh and polylines
    "MeshManager",
    "example_mesh",
    "PolylinesSkeleton",
    # SWC model (from swctools)
    "SWCModel",
    # Tracing API
    "TraceOptions",
    "build_traced_skeleton_graph",
    # Radius optimization
    "OptimizerOptions",
    "RadiusOptimizer",
    "optimize_skeleton_radii",
    "LocalOptimizerOptions",
    "LocalRadiusOptimizer",
    # Skeleton optimization
    "SkeletonOptimizer",
    "SkeletonOptimizerOptions",
]

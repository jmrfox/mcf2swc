"""
mcf2swc

A lightweight toolkit for converting mesh cross-sections and polyline guidance
into SWC skeletons.

Public API:
- MeshManager
- PolylinesSkeleton
- SkeletonGraph, Junction
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
from .mesh import MeshManager, example_mesh  # noqa: E402
from .polylines import PolylinesSkeleton  # noqa: E402
from .skeleton import Junction, SkeletonGraph  # noqa: E402
from .trace import (  # noqa: E402
    TraceOptions,
    build_traced_skeleton_graph,
)
from .radius_optimizer import (  # noqa: E402
    OptimizerOptions,
    RadiusOptimizer,
    optimize_skeleton_radii,
)

__all__ = [
    "__version__",
    # Mesh and polylines
    "MeshManager",
    "example_mesh",
    "PolylinesSkeleton",
    # Skeleton types
    "SkeletonGraph",
    "Junction",
    # Tracing API
    "TraceOptions",
    "build_traced_skeleton_graph",
    # Radius optimization
    "OptimizerOptions",
    "RadiusOptimizer",
    "optimize_skeleton_radii",
]

"""
mcf2swc

A lightweight toolkit for converting mesh cross-sections and polyline guidance
into SWC skeletons.

Public API:
- MeshManager
- PolylinesSkeleton
- SkeletonGraph, Junction, CrossSection, Segment
- TraceOptions, build_traced_skeleton_graph, trace_polylines_to_swc
"""
from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError, version

# Package version reported from installed metadata (fallback for editable/dev installs)
try:
    __version__ = version("mcf2swc")
except PackageNotFoundError:  # pragma: no cover - best-effort in dev
    __version__ = "0.0.0"

# Avoid "No handler found" warnings for library users; applications can configure logging.
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Re-export primary classes and functions for convenient access at package level
from .mesh import MeshManager  # noqa: E402
from .polylines import PolylinesSkeleton  # noqa: E402
from .skeleton import (  # noqa: E402
    CrossSection,
    Junction,
    Segment,
    SkeletonGraph,
)
from .trace import (  # noqa: E402
    TraceOptions,
    build_traced_skeleton_graph,
    trace_polylines_to_swc,
)
from .object3d import Object3D, Transform  # noqa: E402
from .viz import show_swc  # noqa: E402

__all__ = [
    "__version__",
    # Core base
    "Object3D",
    "Transform",
    # Mesh and polylines
    "MeshManager",
    "PolylinesSkeleton",
    # Skeleton graph and data models
    "SkeletonGraph",
    "Junction",
    "CrossSection",
    "Segment",
    # Tracing API
    "TraceOptions",
    "build_traced_skeleton_graph",
    "trace_polylines_to_swc",
    # Visualization
    "show_swc",
]

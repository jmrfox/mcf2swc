"""
mcf2swc

A lightweight toolkit for converting mesh cross-sections and skeleton graph guidance
into SWC models.

Terminology:
- "skeleton": The mesh centroid (SkeletonGraph, result of MCF calculation) without radii
- "SWC model" or "swc": Skeleton with radii information attached to each node

Public API:
- MeshManager
- SkeletonGraph
- SWCModel (from swctools)
- MorphologyGraph
- FitOptions, fit_morphology
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
from .skeleton import SkeletonGraph  # noqa: E402
from .graph_fitting import (  # noqa: E402
    FitOptions,
    fit_morphology,
)
from .morphology_graph import (  # noqa: E402
    MorphologyGraph,
    Junction,
)
from .skeleton_optimizer import (  # noqa: E402
    SkeletonOptimizer,
    SkeletonOptimizerOptions,
)

__all__ = [
    "__version__",
    # Mesh and skeleton
    "MeshManager",
    "example_mesh",
    "SkeletonGraph",
    # SWC model (from swctools)
    "SWCModel",
    # Tracing API
    "FitOptions",
    "fit_morphology",
    "MorphologyGraph",
    "Junction",
    # Skeleton optimization
    "SkeletonOptimizer",
    "SkeletonOptimizerOptions",
]

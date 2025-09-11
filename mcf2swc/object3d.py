"""
Shared 3D object transform management.

Provides a lightweight base class `Object3D` that manages a transform stack and
composite matrices, while delegating the actual geometry application to
subclasses via `_apply_transform_inplace`.

This allows classes like `MeshManager` and `PolylinesSkeleton` to share
consistent transform recording, while keeping their geometry-specific details
separate.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class Transform:
    name: str
    M: np.ndarray  # 4x4 homogeneous
    params: Optional[Dict[str, Any]]
    timestamp: datetime
    is_uniform_scale: bool = False
    uniform_scale: Optional[float] = None


class Object3D:
    """
    Base class for 3D objects that tracks applied transforms.

    Subclasses must implement `_apply_transform_inplace(self, M: np.ndarray) -> None`
    to actually apply the 4x4 transform to their underlying geometry.

    Optionally, subclasses may override `_post_apply_transform(...)` to run any
    post-processing after a transform is applied (e.g., recompute bounds).
    """

    def __init__(self) -> None:
        # Stack of applied transforms (in order of application)
        self.transform_stack: list[Transform] = []
        # Composite matrices
        self.M_world_from_local: np.ndarray = np.eye(4, dtype=float)
        self.M_local_from_world: np.ndarray = np.eye(4, dtype=float)

    # ---- abstract hooks -------------------------------------------------
    def _apply_transform_inplace(self, M: np.ndarray) -> None:  # pragma: no cover - abstract
        """Apply the 4x4 matrix `M` to the underlying geometry in-place.

        Subclasses must implement this.
        """
        raise NotImplementedError

    def _post_apply_transform(
        self,
        name: str,
        M: np.ndarray,
        *,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Optional hook for subclasses to run after a transform is applied.

        Default implementation does nothing.
        """
        return None

    # ---- core shared logic ---------------------------------------------
    def _apply_and_record_transform(
        self,
        name: str,
        M: np.ndarray,
        *,
        params: Optional[Dict[str, Any]] = None,
        is_uniform_scale: bool = False,
        uniform_scale: Optional[float] = None,
    ) -> None:
        """Apply a 4x4 transform and record it on the stack.

        This updates the composite matrices and stores a `Transform` record.
        """
        M = np.asarray(M, dtype=float)
        if M.shape != (4, 4):
            raise ValueError("Transform matrix must be 4x4")

        # 1) Apply to subclass geometry
        self._apply_transform_inplace(M)

        # 2) Update composite matrices
        self.M_world_from_local = M @ self.M_world_from_local
        try:
            self.M_local_from_world = np.linalg.inv(self.M_world_from_local)
        except Exception:
            # Keep previous inverse if singular; still record transform
            pass

        # 3) Record on stack
        self.transform_stack.append(
            Transform(
                name=name,
                M=M.copy(),
                params=None if params is None else dict(params),
                timestamp=datetime.now(),
                is_uniform_scale=bool(is_uniform_scale),
                uniform_scale=(float(uniform_scale) if uniform_scale is not None else None),
            )
        )

        # 4) Allow subclass to post-process (e.g., recompute bounds)
        try:
            self._post_apply_transform(name, M, params=params)
        except Exception:
            logger.debug("_post_apply_transform hook raised; continuing", exc_info=True)

    def apply_transform(
        self,
        M: np.ndarray,
        *,
        name: str = "custom",
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Public convenience to apply and record an arbitrary 4x4 transform."""
        self._apply_and_record_transform(name, M, params=params)

    def get_composite_matrix(self) -> np.ndarray:
        """Return the cumulative 4x4 matrix mapping local->world (applied order)."""
        return self.M_world_from_local.copy()

    def get_inverse_matrix(self) -> np.ndarray:
        """Return the inverse composite 4x4 matrix (world->local)."""
        return self.M_local_from_world.copy()

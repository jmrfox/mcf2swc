"""
Optimization-based radius estimation for skeleton graphs.

This module provides an alternative to the procedural radius strategies in trace.py.
Instead of determining each node's radius independently using local geometric
measurements, this treats the entire set of radii as parameters in an optimization
problem. The goal is to minimize a loss function that measures how well the SWC
model (represented as a collection of frusta) approximates the original mesh.

Key features:
- Loss functions: surface area error, volume error, or custom combinations
- Multiple optimization backends: scipy.optimize, gradient descent, etc.
- Flexible initialization from existing radius estimates
- Constraint handling (e.g., minimum/maximum radius bounds)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import trimesh

from .skeleton import SkeletonGraph

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class OptimizerOptions:
    """
    Configuration for radius optimization.

    Attributes:
        loss_function: Name of the loss function to minimize. Options:
            - "surface_area": Minimize difference between SWC model surface area
              and mesh surface area
            - "volume": Minimize difference between SWC model volume and mesh volume
            - "combined": Weighted combination of surface area and volume
        loss_weights: Dictionary of weights for combined loss functions.
            E.g., {"surface_area": 1.0, "volume": 0.5}
        constraint_mode: How to constrain the optimization problem. Options:
            - "unconstrained": No additional constraints (may be under-constrained)
            - "regularization": Add penalty for deviating from initial radii
            - "scale_only": Optimize only a global scale factor, preserving relative
              proportions of initial radii
        regularization_weight: Weight for regularization penalty (when constraint_mode
            is "regularization"). Higher values keep radii closer to initial estimates.
            Default: 0.1
        optimizer: Optimization algorithm to use. Options:
            - "scipy_lbfgsb": L-BFGS-B from scipy.optimize (default)
            - "scipy_slsqp": SLSQP from scipy.optimize
            - "scipy_minimize": Generic scipy.optimize.minimize
        min_radius: Minimum allowed radius (constraint)
        max_radius: Maximum allowed radius (constraint)
        max_iterations: Maximum number of optimization iterations
        tolerance: Convergence tolerance for the optimizer
        verbose: If True, print optimization progress
    """

    loss_function: str = "surface_area"
    loss_weights: Optional[Dict[str, float]] = None
    constraint_mode: str = "regularization"
    regularization_weight: float = 0.1
    optimizer: str = "scipy_lbfgsb"
    min_radius: float = 0.01
    max_radius: Optional[float] = None
    max_iterations: int = 1000
    tolerance: float = 1e-6
    verbose: bool = False


class RadiusOptimizer:
    """
    Optimizer for skeleton graph radii based on mesh approximation quality.

    This class takes a SkeletonGraph with initial radius estimates and a target
    mesh, then optimizes all radii jointly to minimize a loss function measuring
    the discrepancy between the SWC model and the mesh.

    Example:
        >>> optimizer = RadiusOptimizer(skeleton_graph, mesh)
        >>> optimized_graph = optimizer.optimize()
    """

    def __init__(
        self,
        skeleton: SkeletonGraph,
        mesh: trimesh.Trimesh,
        *,
        options: Optional[OptimizerOptions] = None,
    ):
        """
        Initialize the optimizer.

        Args:
            skeleton: SkeletonGraph with initial radius estimates at each node
            mesh: Target mesh to approximate
            options: Optimization configuration
        """
        self.skeleton = skeleton
        self.mesh = mesh
        self.options = options if options is not None else OptimizerOptions()

        # Extract node ordering for parameter vector
        self.node_ids = sorted(skeleton.nodes())
        self.n_nodes = len(self.node_ids)

        # Build index mapping
        self.node_to_idx = {nid: i for i, nid in enumerate(self.node_ids)}

        # Store initial radii for regularization
        self.initial_radii = self.get_initial_radii()

        # Cache mesh properties
        self._mesh_surface_area = None
        self._mesh_volume = None

        # Optimization history
        self.history: List[Dict[str, Any]] = []

        logger.info(
            "RadiusOptimizer initialized: nodes=%d, loss=%s, constraint=%s, optimizer=%s",
            self.n_nodes,
            self.options.loss_function,
            self.options.constraint_mode,
            self.options.optimizer,
        )

    def get_initial_radii(self) -> np.ndarray:
        """
        Extract current radii from the skeleton graph as a parameter vector.

        Returns:
            Array of shape (n_nodes,) with current radius values
        """
        radii = np.zeros(self.n_nodes, dtype=float)
        for i, nid in enumerate(self.node_ids):
            radii[i] = float(self.skeleton.nodes[nid].get("radius", 1.0))
        return radii

    def set_radii(self, radii: np.ndarray) -> None:
        """
        Update skeleton graph with new radius values.

        Args:
            radii: Array of shape (n_nodes,) with new radius values
        """
        if radii.shape[0] != self.n_nodes:
            raise ValueError(f"Expected {self.n_nodes} radii, got {radii.shape[0]}")
        for i, nid in enumerate(self.node_ids):
            self.skeleton.nodes[nid]["radius"] = float(radii[i])

    def compute_swc_surface_area(self, radii: np.ndarray) -> float:
        """
        Compute total surface area of the SWC model (frusta).

        Each edge in the skeleton represents a frustum (truncated cone) connecting
        two nodes with potentially different radii. The surface area includes the
        lateral surface of each frustum.

        Args:
            radii: Array of radii for each node

        Returns:
            Total surface area of all frusta
        """
        total_area = 0.0

        for u, v in self.skeleton.edges():
            # Get node indices
            i_u = self.node_to_idx[u]
            i_v = self.node_to_idx[v]

            # Get positions and radii
            xyz_u = self.skeleton.nodes[u]["xyz"]
            xyz_v = self.skeleton.nodes[v]["xyz"]
            r_u = radii[i_u]
            r_v = radii[i_v]

            # Compute frustum lateral surface area
            # A = π(r1 + r2) * sqrt(h^2 + (r1 - r2)^2)
            # where h is the height (edge length)
            h = float(np.linalg.norm(xyz_v - xyz_u))
            if h <= 0:
                continue

            slant = math.sqrt(h * h + (r_u - r_v) ** 2)
            area = math.pi * (r_u + r_v) * slant
            total_area += area

        return total_area

    def compute_swc_volume(self, radii: np.ndarray) -> float:
        """
        Compute total volume of the SWC model (frusta).

        Args:
            radii: Array of radii for each node

        Returns:
            Total volume of all frusta
        """
        total_volume = 0.0

        for u, v in self.skeleton.edges():
            # Get node indices
            i_u = self.node_to_idx[u]
            i_v = self.node_to_idx[v]

            # Get positions and radii
            xyz_u = self.skeleton.nodes[u]["xyz"]
            xyz_v = self.skeleton.nodes[v]["xyz"]
            r_u = radii[i_u]
            r_v = radii[i_v]

            # Compute frustum volume
            # V = (π * h / 3) * (r1^2 + r1*r2 + r2^2)
            h = float(np.linalg.norm(xyz_v - xyz_u))
            if h <= 0:
                continue

            volume = (math.pi * h / 3.0) * (r_u * r_u + r_u * r_v + r_v * r_v)
            total_volume += volume

        return total_volume

    def get_mesh_surface_area(self) -> float:
        """Get cached mesh surface area."""
        if self._mesh_surface_area is None:
            self._mesh_surface_area = float(self.mesh.area)
        return self._mesh_surface_area

    def get_mesh_volume(self) -> float:
        """Get cached mesh volume."""
        if self._mesh_volume is None:
            self._mesh_volume = float(self.mesh.volume)
        return self._mesh_volume

    def compute_data_loss(self, radii: np.ndarray) -> float:
        """
        Compute the data loss (mesh approximation error) for a given set of radii.

        Args:
            radii: Array of radii for each node

        Returns:
            Data loss value (lower is better)
        """
        loss_type = self.options.loss_function

        if loss_type == "surface_area":
            swc_area = self.compute_swc_surface_area(radii)
            mesh_area = self.get_mesh_surface_area()
            # Relative error
            loss = abs(swc_area - mesh_area) / (mesh_area + 1e-12)

        elif loss_type == "volume":
            swc_vol = self.compute_swc_volume(radii)
            mesh_vol = self.get_mesh_volume()
            # Relative error
            loss = abs(swc_vol - mesh_vol) / (abs(mesh_vol) + 1e-12)

        elif loss_type == "combined":
            weights = self.options.loss_weights or {
                "surface_area": 1.0,
                "volume": 1.0,
            }

            swc_area = self.compute_swc_surface_area(radii)
            mesh_area = self.get_mesh_surface_area()
            area_loss = abs(swc_area - mesh_area) / (mesh_area + 1e-12)

            swc_vol = self.compute_swc_volume(radii)
            mesh_vol = self.get_mesh_volume()
            vol_loss = abs(swc_vol - mesh_vol) / (abs(mesh_vol) + 1e-12)

            loss = (
                weights.get("surface_area", 1.0) * area_loss
                + weights.get("volume", 1.0) * vol_loss
            )

        else:
            raise ValueError(f"Unknown loss function: {loss_type}")

        return loss

    def compute_regularization_loss(self, radii: np.ndarray) -> float:
        """
        Compute regularization penalty for deviating from initial radii.

        Args:
            radii: Array of radii for each node

        Returns:
            Regularization loss (L2 penalty)
        """
        # Normalized L2 distance from initial radii
        diff = radii - self.initial_radii
        norm_initial = np.linalg.norm(self.initial_radii) + 1e-12
        return float(np.linalg.norm(diff) / norm_initial)

    def compute_loss(self, radii: np.ndarray) -> float:
        """
        Compute the total loss function for a given set of radii.

        Includes data loss and optional regularization based on constraint_mode.

        Args:
            radii: Array of radii for each node

        Returns:
            Total loss value (lower is better)
        """
        data_loss = self.compute_data_loss(radii)

        # Add regularization if requested
        if self.options.constraint_mode == "regularization":
            reg_loss = self.compute_regularization_loss(radii)
            total_loss = data_loss + self.options.regularization_weight * reg_loss
            return total_loss
        else:
            return data_loss

    def compute_loss_and_gradient(self, radii: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute loss and its gradient with respect to radii.

        Uses finite differences for gradient approximation.

        Args:
            radii: Array of radii for each node

        Returns:
            Tuple of (loss, gradient)
        """
        loss = self.compute_loss(radii)

        # Finite difference gradient
        eps = 1e-6
        grad = np.zeros_like(radii)

        for i in range(len(radii)):
            radii_plus = radii.copy()
            radii_plus[i] += eps
            loss_plus = self.compute_loss(radii_plus)
            grad[i] = (loss_plus - loss) / eps

        return loss, grad

    def _optimize_scale_only(self) -> SkeletonGraph:
        """
        Optimize only a global scale factor, preserving relative radii proportions.

        This reduces the optimization to a 1D problem: find the best scale factor s
        such that radii_optimized = s * initial_radii minimizes the loss function.

        Returns:
            A new SkeletonGraph with scaled radii
        """
        logger.info("Starting scale-only optimization")

        # Define loss function for scale factor
        def scale_loss(s: float) -> float:
            scaled_radii = s * self.initial_radii
            return self.compute_data_loss(scaled_radii)

        # Set up bounds for scale factor based on radius constraints
        min_scale = self.options.min_radius / (np.min(self.initial_radii) + 1e-12)
        if self.options.max_radius is not None:
            max_scale = self.options.max_radius / (np.max(self.initial_radii) + 1e-12)
        else:
            max_scale = 100.0  # Reasonable upper bound

        # Ensure bounds are valid
        min_scale = max(min_scale, 0.01)
        max_scale = max(max_scale, min_scale * 1.1)

        # Callback for tracking progress
        iteration = [0]

        def callback(xk):
            iteration[0] += 1
            if self.options.verbose and iteration[0] % 10 == 0:
                s = float(xk) if np.isscalar(xk) else float(xk[0])
                loss = scale_loss(s)
                logger.info(
                    f"Iteration {iteration[0]}: scale = {s:.6f}, loss = {loss:.6e}"
                )
                self.history.append(
                    {"iteration": iteration[0], "scale": s, "loss": loss}
                )

        # Optimize scale factor using scipy
        from scipy.optimize import minimize_scalar

        result = minimize_scalar(
            scale_loss,
            bounds=(min_scale, max_scale),
            method="bounded",
            options={"maxiter": self.options.max_iterations},
        )

        optimal_scale = float(result.x)
        optimized_radii = optimal_scale * self.initial_radii

        # Log results
        initial_loss = scale_loss(1.0)
        final_loss = scale_loss(optimal_scale)
        logger.info(
            "Scale-only optimization complete: optimal_scale=%.6f, "
            "initial_loss=%.6e, final_loss=%.6e, improvement=%.2f%%",
            optimal_scale,
            initial_loss,
            final_loss,
            100 * (initial_loss - final_loss) / (initial_loss + 1e-12),
        )

        # Create new skeleton with scaled radii
        optimized_skeleton = self.skeleton.copy()
        for i, nid in enumerate(self.node_ids):
            optimized_skeleton.nodes[nid]["radius"] = float(optimized_radii[i])

        return optimized_skeleton

    def optimize(self) -> SkeletonGraph:
        """
        Optimize radii to minimize the loss function.

        Returns:
            A new SkeletonGraph with optimized radii
        """
        # Handle scale-only constraint mode separately
        if self.options.constraint_mode == "scale_only":
            return self._optimize_scale_only()

        # Get initial radii
        r0 = self.get_initial_radii()

        # Set up bounds
        bounds = []
        for i in range(self.n_nodes):
            min_r = self.options.min_radius
            max_r = self.options.max_radius if self.options.max_radius else np.inf
            bounds.append((min_r, max_r))

        # Callback for tracking progress
        iteration = [0]

        def callback(xk):
            iteration[0] += 1
            if self.options.verbose and iteration[0] % 10 == 0:
                loss = self.compute_loss(xk)
                logger.info(f"Iteration {iteration[0]}: loss = {loss:.6e}")
                self.history.append({"iteration": iteration[0], "loss": loss})

        # Run optimization
        logger.info("Starting optimization with %s", self.options.optimizer)

        if self.options.optimizer == "scipy_lbfgsb":
            from scipy.optimize import minimize

            result = minimize(
                fun=self.compute_loss,
                x0=r0,
                method="L-BFGS-B",
                jac=lambda x: self.compute_loss_and_gradient(x)[1],
                bounds=bounds,
                options={
                    "maxiter": self.options.max_iterations,
                    "ftol": self.options.tolerance,
                },
                callback=callback,
            )
            optimized_radii = result.x

        elif self.options.optimizer == "scipy_slsqp":
            from scipy.optimize import minimize

            result = minimize(
                fun=self.compute_loss,
                x0=r0,
                method="SLSQP",
                bounds=bounds,
                options={
                    "maxiter": self.options.max_iterations,
                    "ftol": self.options.tolerance,
                },
                callback=callback,
            )
            optimized_radii = result.x

        elif self.options.optimizer == "scipy_minimize":
            from scipy.optimize import minimize

            result = minimize(
                fun=self.compute_loss,
                x0=r0,
                bounds=bounds,
                options={
                    "maxiter": self.options.max_iterations,
                },
                callback=callback,
            )
            optimized_radii = result.x

        else:
            raise ValueError(f"Unknown optimizer: {self.options.optimizer}")

        # Log results
        initial_loss = self.compute_loss(r0)
        final_loss = self.compute_loss(optimized_radii)
        logger.info(
            "Optimization complete: initial_loss=%.6e, final_loss=%.6e, "
            "improvement=%.2f%%",
            initial_loss,
            final_loss,
            100 * (initial_loss - final_loss) / (initial_loss + 1e-12),
        )

        # Create new skeleton with optimized radii
        optimized_skeleton = self.skeleton.copy()
        for i, nid in enumerate(self.node_ids):
            optimized_skeleton.nodes[nid]["radius"] = float(optimized_radii[i])

        return optimized_skeleton


def optimize_skeleton_radii(
    skeleton: SkeletonGraph,
    mesh: trimesh.Trimesh,
    *,
    options: Optional[OptimizerOptions] = None,
) -> SkeletonGraph:
    """
    Convenience function to optimize skeleton radii.

    Args:
        skeleton: SkeletonGraph with initial radius estimates
        mesh: Target mesh to approximate
        options: Optimization configuration

    Returns:
        New SkeletonGraph with optimized radii

    Example:
        >>> from mcf2swc import SkeletonGraph, MeshManager
        >>> from mcf2swc.radius_optimizer import optimize_skeleton_radii, OptimizerOptions
        >>>
        >>> # Build initial skeleton with trace.py
        >>> skeleton = build_traced_skeleton_graph(mesh, polylines)
        >>>
        >>> # Optimize radii
        >>> opts = OptimizerOptions(loss_function="surface_area", verbose=True)
        >>> optimized = optimize_skeleton_radii(skeleton, mesh, options=opts)
    """
    optimizer = RadiusOptimizer(skeleton, mesh, options=options)
    return optimizer.optimize()

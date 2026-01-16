"""
Radius optimization.

This module implements a segment-by-segment optimization approach where each
frustum's radii are optimized to minimize the distance from surface points
to the mesh surface.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import trimesh
from scipy.optimize import minimize
from swctools import SWCModel

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _get_node_xyz(graph: SWCModel, node_id: int) -> np.ndarray:
    """Extract xyz coordinates from SWCModel."""
    node = graph.nodes[node_id]
    if "xyz" in node:
        return np.asarray(node["xyz"], dtype=float)
    elif "x" in node and "y" in node and "z" in node:
        return np.array([node["x"], node["y"], node["z"]], dtype=float)
    else:
        raise ValueError(f"Node {node_id} missing xyz coordinates")


def _get_node_radius(graph: SWCModel, node_id: int) -> float:
    """Extract radius from SWCModel."""
    node = graph.nodes[node_id]
    if "radius" in node:
        return float(node["radius"])
    elif "r" in node:
        return float(node["r"])
    else:
        raise ValueError(f"Node {node_id} missing radius")


@dataclass
class RadiusOptimizerOptions:
    """
    Configuration for radius optimization.

    Attributes:
        n_longitudinal: Number of sample points along the frustum axis
        n_radial: Number of sample points around the circumference
        max_iterations: Maximum number of passes over all segments
        convergence_threshold: Stop when max radius change is below this value
        min_radius: Minimum allowed radius
        max_radius: Maximum allowed radius
        step_size_factor: Factor for computing optimization step size from distances
        verbose: If True, print optimization progress
    """

    n_longitudinal: int = 3
    n_radial: int = 8
    max_iterations: int = 10
    convergence_threshold: float = 1e-4
    min_radius: float = 0.01
    max_radius: Optional[float] = None
    step_size_factor: float = 0.1
    verbose: bool = False


def _sample_frustum_surface_points(
    xyz_a: np.ndarray,
    xyz_b: np.ndarray,
    r_a: float,
    r_b: float,
    n_long: int,
    n_rad: int,
) -> np.ndarray:
    """
    Sample points uniformly on the lateral surface of a frustum.

    Args:
        xyz_a: Position of endpoint A (3,)
        xyz_b: Position of endpoint B (3,)
        r_a: Radius at endpoint A
        r_b: Radius at endpoint B
        n_long: Number of samples along the axis
        n_rad: Number of samples around the circumference

    Returns:
        Array of surface points, shape (n_long * n_rad, 3)
    """
    # Compute axis direction
    axis = xyz_b - xyz_a
    axis_length = np.linalg.norm(axis)

    if axis_length < 1e-12:
        # Degenerate segment - return points on a sphere
        if r_a > 0:
            points = []
            for i in range(n_rad):
                theta = 2.0 * np.pi * (i / n_rad)
                x = xyz_a[0] + r_a * np.cos(theta)
                y = xyz_a[1] + r_a * np.sin(theta)
                z = xyz_a[2]
                points.append([x, y, z])
            return np.array(points)
        else:
            return np.array([xyz_a])

    axis_unit = axis / axis_length

    # Create orthonormal frame (U, V, W) with W along axis
    # Choose U perpendicular to axis
    if abs(axis_unit[0]) < 0.9:
        temp = np.array([1.0, 0.0, 0.0])
    else:
        temp = np.array([0.0, 1.0, 0.0])

    U = np.cross(temp, axis_unit)
    U = U / (np.linalg.norm(U) + 1e-12)
    V = np.cross(axis_unit, U)
    V = V / (np.linalg.norm(V) + 1e-12)

    # Sample points
    points = []
    for i in range(n_long):
        # Parameter along axis [0, 1]
        t = i / max(1, n_long - 1) if n_long > 1 else 0.5

        # Position and radius at this t
        center = xyz_a + t * axis
        radius = r_a + t * (r_b - r_a)

        # Sample points around the circumference
        for j in range(n_rad):
            theta = 2.0 * np.pi * (j / n_rad)
            offset = radius * (np.cos(theta) * U + np.sin(theta) * V)
            point = center + offset
            points.append(point)

    return np.array(points)


def _compute_distances_to_mesh(
    points: np.ndarray,
    mesh: trimesh.Trimesh,
) -> np.ndarray:
    """
    Compute unsigned distances from points to nearest mesh surface.

    Args:
        points: Array of points, shape (N, 3)
        mesh: Target mesh

    Returns:
        Array of distances, shape (N,)
    """
    try:
        from trimesh.proximity import closest_point

        _, distances, _ = closest_point(mesh, points)
        return distances
    except Exception:
        # Fallback: compute distances to vertices
        vertices = np.asarray(mesh.vertices, dtype=float)
        distances = []
        for p in points:
            dists = np.linalg.norm(vertices - p, axis=1)
            distances.append(np.min(dists))
        return np.array(distances)


def _optimize_segment_radii(
    xyz_a: np.ndarray,
    xyz_b: np.ndarray,
    r_a_init: float,
    r_b_init: float,
    mesh: trimesh.Trimesh,
    options: LocalOptimizerOptions,
) -> Tuple[float, float, float]:
    """
    Optimize radii for a single segment to minimize MSE distance to mesh.

    Args:
        xyz_a: Position of endpoint A
        xyz_b: Position of endpoint B
        r_a_init: Initial radius at A
        r_b_init: Initial radius at B
        mesh: Target mesh
        options: Optimization options

    Returns:
        Tuple of (optimized r_a, optimized r_b, final MSE)
    """

    def objective(radii):
        r_a, r_b = radii

        # Sample surface points with current radii
        points = _sample_frustum_surface_points(
            xyz_a, xyz_b, r_a, r_b, options.n_longitudinal, options.n_radial
        )

        # Compute distances to mesh
        distances = _compute_distances_to_mesh(points, mesh)

        # Return MSE
        return np.mean(distances**2)

    # Initial guess
    x0 = np.array([r_a_init, r_b_init])

    # Bounds
    bounds = [
        (options.min_radius, options.max_radius if options.max_radius else np.inf),
        (options.min_radius, options.max_radius if options.max_radius else np.inf),
    ]

    # Optimize
    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 50, "ftol": 1e-6},
    )

    r_a_opt, r_b_opt = result.x
    mse = result.fun

    return float(r_a_opt), float(r_b_opt), float(mse)


class RadiusOptimizer:
    """
    Segment-by-segment radius optimizer.

    This optimizer iterates over all segments (edges) in the skeleton graph,
    optimizing each segment's endpoint radii to minimize the distance from
    frustum surface points to the mesh surface.
    """

    def __init__(
        self,
        morphology: SWCModel,
        mesh: trimesh.Trimesh,
        *,
        options: Optional[RadiusOptimizerOptions] = None,
    ):
        """
        Initialize the radius optimizer.

        Args:
            skeleton: SWC model with initial radius estimates
            mesh: Target mesh to fit
            options: Optimization options
        """
        self.skeleton = morphology
        self.mesh = mesh
        self.options = options if options is not None else RadiusOptimizerOptions()

        # Build node index mapping
        self.node_to_idx = {nid: i for i, nid in enumerate(sorted(morphology.nodes()))}
        self.idx_to_node = {i: nid for nid, i in self.node_to_idx.items()}

        # Extract initial radii using helper function
        self.initial_radii = np.array(
            [
                _get_node_radius(morphology, self.idx_to_node[i])
                for i in range(len(self.node_to_idx))
            ]
        )

        # Current radii (will be updated during optimization)
        self.current_radii = self.initial_radii.copy()

        logger.info(
            "RadiusOptimizer initialized: nodes=%d, edges=%d, n_long=%d, n_rad=%d",
            self.skeleton.number_of_nodes(),
            self.skeleton.number_of_edges(),
            self.options.n_longitudinal,
            self.options.n_radial,
        )

    def optimize(self) -> SWCModel:
        """
        Run the iterative segment-by-segment optimization.

        Returns:
            New SWC model with optimized radii
        """
        edges = list(self.skeleton.edges())
        n_edges = len(edges)

        if self.options.verbose:
            print(
                f"Starting local optimization: {n_edges} segments, max {self.options.max_iterations} iterations"
            )

        for iteration in range(self.options.max_iterations):
            max_change = 0.0
            total_mse = 0.0

            # Process each segment
            for edge_idx, (u, v) in enumerate(edges):
                # Get node indices
                i_u = self.node_to_idx[u]
                i_v = self.node_to_idx[v]

                # Get positions using helper function
                xyz_u = _get_node_xyz(self.skeleton, u)
                xyz_v = _get_node_xyz(self.skeleton, v)

                # Get current radii
                r_u = self.current_radii[i_u]
                r_v = self.current_radii[i_v]

                # Optimize this segment
                r_u_new, r_v_new, mse = _optimize_segment_radii(
                    xyz_u, xyz_v, r_u, r_v, self.mesh, self.options
                )

                # Update radii
                self.current_radii[i_u] = r_u_new
                self.current_radii[i_v] = r_v_new

                # Track changes
                change_u = abs(r_u_new - r_u)
                change_v = abs(r_v_new - r_v)
                max_change = max(max_change, change_u, change_v)
                total_mse += mse

            avg_mse = total_mse / n_edges

            if self.options.verbose:
                print(
                    f"  Iteration {iteration + 1}: max_change={max_change:.6f}, avg_mse={avg_mse:.6f}"
                )

            logger.info(
                "Iteration %d: max_change=%.6f, avg_mse=%.6f",
                iteration + 1,
                max_change,
                avg_mse,
            )

            # Check convergence
            if max_change < self.options.convergence_threshold:
                if self.options.verbose:
                    print(f"Converged after {iteration + 1} iterations")
                logger.info("Converged after %d iterations", iteration + 1)
                break

        # Compute final global metrics for monitoring
        final_sa = self._compute_skeleton_surface_area(self.current_radii)
        final_vol = self._compute_skeleton_volume(self.current_radii)
        mesh_sa = float(self.mesh.area)
        mesh_vol = float(self.mesh.volume)

        sa_error = abs(final_sa - mesh_sa) / mesh_sa if mesh_sa > 0 else 0.0
        vol_error = abs(final_vol - mesh_vol) / mesh_vol if mesh_vol > 0 else 0.0

        if self.options.verbose:
            print(f"\nFinal metrics:")
            print(
                f"  Surface area: skeleton={final_sa:.2f}, mesh={mesh_sa:.2f}, error={sa_error:.2%}"
            )
            print(
                f"  Volume: skeleton={final_vol:.2f}, mesh={mesh_vol:.2f}, error={vol_error:.2%}"
            )

        logger.info(
            "Optimization complete: SA_error=%.2f%%, Vol_error=%.2f%%",
            sa_error * 100,
            vol_error * 100,
        )

        # Create new skeleton with optimized radii
        return self._create_optimized_skeleton()

    def _compute_skeleton_surface_area(self, radii: np.ndarray) -> float:
        """Compute total surface area of frustum segments."""
        total_area = 0.0

        for u, v in self.skeleton.edges():
            i_u = self.node_to_idx[u]
            i_v = self.node_to_idx[v]

            xyz_u = _get_node_xyz(self.skeleton, u)
            xyz_v = _get_node_xyz(self.skeleton, v)
            r_u = radii[i_u]
            r_v = radii[i_v]

            # Frustum lateral surface area: π * (r1 + r2) * s
            # where s = sqrt(h^2 + (r2 - r1)^2) is slant height
            h = float(np.linalg.norm(xyz_v - xyz_u))
            if h <= 0:
                continue

            s = math.sqrt(h * h + (r_v - r_u) * (r_v - r_u))
            area = math.pi * (r_u + r_v) * s
            total_area += area

        return total_area

    def _compute_skeleton_volume(self, radii: np.ndarray) -> float:
        """Compute total volume of frustum segments."""
        total_volume = 0.0

        for u, v in self.skeleton.edges():
            i_u = self.node_to_idx[u]
            i_v = self.node_to_idx[v]

            xyz_u = _get_node_xyz(self.skeleton, u)
            xyz_v = _get_node_xyz(self.skeleton, v)
            r_u = radii[i_u]
            r_v = radii[i_v]

            # Frustum volume: V = (π * h / 3) * (r1^2 + r1*r2 + r2^2)
            h = float(np.linalg.norm(xyz_v - xyz_u))
            if h <= 0:
                continue

            volume = (math.pi * h / 3.0) * (r_u * r_u + r_u * r_v + r_v * r_v)
            total_volume += volume

        return total_volume

    def _create_optimized_skeleton(self) -> SWCModel:
        """Create a new SWC model with optimized radii."""
        new_skeleton = SWCModel()
        if hasattr(self.skeleton, "_parents"):
            new_skeleton._parents = dict(self.skeleton._parents)

        # Copy nodes with updated radii
        for nid in self.skeleton.nodes():
            idx = self.node_to_idx[nid]
            node_data = dict(self.skeleton.nodes[nid])

            # Update radius (handle both 'radius' and 'r' attributes)
            if "r" in node_data:
                node_data["r"] = float(self.current_radii[idx])
            else:
                node_data["radius"] = float(self.current_radii[idx])

            new_skeleton.add_node(nid, **node_data)

        # Copy edges
        for u, v in self.skeleton.edges():
            edge_data = dict(self.skeleton.edges[u, v])
            new_skeleton.add_edge(u, v, **edge_data)

        return new_skeleton

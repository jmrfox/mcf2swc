"""
Skeleton optimization for MCF-generated polylines.

This module provides optimization for skeleton polylines produced by mean-curvature
flow (MCF) calculations. MCF skeletons reproduce general topology well but can deviate
from the true medial axis and sometimes clip through the mesh surface, especially in
regions with holes or high curvature.

The SkeletonOptimizer gently pushes skeleton points toward the center of the mesh
volume while preserving the overall topology and structure.

Key features:
- Detection of skeleton points that cross the mesh surface
- Optimization to push points toward the medial axis
- Preservation of skeleton topology and connectivity
- Configurable optimization parameters
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import trimesh

from .polylines import PolylinesSkeleton

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class SkeletonOptimizerOptions:
    """
    Configuration for skeleton optimization.

    Attributes:
        check_surface_crossing: If True, check whether skeleton points cross
            the mesh surface before optimization. Default: True
        max_iterations: Maximum number of optimization iterations. Default: 100
        step_size: Step size for moving points toward the center. Smaller values
            are more conservative. Default: 0.1
        convergence_threshold: Stop optimization when average point movement
            is below this threshold. Default: 1e-4
        preserve_endpoints: If True, do not move the first and last points of
            each polyline. Default: True
        preserve_branch_points: If True, do not move branch points (where 3+
            polylines meet). Requires topology detection. Default: False
        branch_point_tolerance: Distance threshold for detecting branch points.
            Default: 1e-6
        centering_method: Method for computing centering direction:
            'closest_point' - move toward closest point on surface (simple)
            'medial_axis' - equalize distances in perpendicular directions (better)
            Default: 'medial_axis'
        probe_distance: Fallback distance for medial axis centering when ray
            tracing fails or finds no intersection. Also used as the default
            return value for rays that don't hit the mesh. Default: 10.0
        num_probe_directions: Number of directions to probe around the skeleton
            tangent for medial axis centering. More directions = more robust but
            slower. Must be >= 2. Default: 4 (samples at 0°, 90°, 180°, 270°)
        smoothing_weight: Weight for smoothing regularization to maintain
            polyline smoothness (0 = no smoothing, 1 = strong smoothing).
            Default: 0.5
        verbose: If True, print optimization progress. Default: False
    """

    check_surface_crossing: bool = True
    max_iterations: int = 100
    step_size: float = 0.1
    convergence_threshold: float = 1e-4
    preserve_endpoints: bool = True
    preserve_branch_points: bool = False
    branch_point_tolerance: float = 1e-6
    centering_method: str = "medial_axis"
    probe_distance: float = 10.0
    num_probe_directions: int = 4
    smoothing_weight: float = 0.5
    verbose: bool = False


class SkeletonOptimizer:
    """
    Optimizer for skeleton polylines to push points toward the mesh medial axis.

    This class takes a skeleton (polylines format) produced by MCF skeletonization
    and a mesh, then optimizes the skeleton points to better approximate the
    medial axis of the mesh volume.

    Example:
        >>> from mcf2swc import PolylinesSkeleton, MeshManager, SkeletonOptimizer
        >>> skeleton = PolylinesSkeleton.from_txt("skeleton.polylines.txt")
        >>> mesh_mgr = MeshManager(mesh_path="mesh.obj")
        >>> optimizer = SkeletonOptimizer(skeleton, mesh_mgr.mesh)
        >>> optimized_skeleton = optimizer.optimize()
    """

    def __init__(
        self,
        skeleton: PolylinesSkeleton,
        mesh: trimesh.Trimesh,
        options: Optional[SkeletonOptimizerOptions] = None,
    ):
        """
        Initialize the skeleton optimizer.

        Args:
            skeleton: Input skeleton polylines from MCF calculation
            mesh: Target mesh that the skeleton should approximate
            options: Optimization configuration options
        """
        self.skeleton = skeleton.copy()
        self.mesh = mesh
        self.options = options or SkeletonOptimizerOptions()

        self._surface_crossing_detected = False
        self._optimization_history = []

    def check_surface_crossing(self) -> Tuple[bool, int, float]:
        """
        Check if any skeleton points are outside the mesh surface.

        Returns:
            Tuple of (has_crossing, num_outside_points, max_distance)
            - has_crossing: True if any points are outside the mesh
            - num_outside_points: Number of points outside the mesh
            - max_distance: Maximum distance to surface for outside points
        """
        if not self.skeleton.polylines:
            return False, 0, 0.0

        all_pts = np.vstack(self.skeleton.polylines)

        try:
            inside_mask = self.mesh.contains(all_pts)
            outside_mask = ~inside_mask
            num_outside = int(np.sum(outside_mask))

            max_dist = 0.0
            if num_outside > 0:
                from trimesh.proximity import closest_point

                outside_pts = all_pts[outside_mask]
                cp, distances, _ = closest_point(self.mesh, outside_pts)
                max_dist = float(np.max(distances))

            has_crossing = num_outside > 0
            self._surface_crossing_detected = has_crossing

            if self.options.verbose:
                if has_crossing:
                    logger.info(
                        "Surface crossing detected: %d/%d points outside mesh (max distance: %.4f)",
                        num_outside,
                        len(all_pts),
                        max_dist,
                    )
                else:
                    logger.info("No surface crossing detected - all points inside mesh")

            return has_crossing, num_outside, max_dist

        except Exception as e:
            logger.warning("Failed to check surface crossing: %s", e)
            return False, 0, 0.0

    def optimize(self) -> PolylinesSkeleton:
        """
        Optimize the skeleton by pushing points toward the mesh medial axis.

        Returns:
            Optimized skeleton polylines
        """
        if self.options.check_surface_crossing:
            self.check_surface_crossing()

        if self.options.verbose:
            logger.info("Starting skeleton optimization...")
            logger.info("  Max iterations: %d", self.options.max_iterations)
            logger.info("  Step size: %.4f", self.options.step_size)
            logger.info("  Smoothing weight: %.4f", self.options.smoothing_weight)

        optimized_polylines = []
        for poly_idx, polyline in enumerate(self.skeleton.polylines):
            if len(polyline) == 0:
                optimized_polylines.append(polyline.copy())
                continue

            optimized = self._optimize_polyline(polyline, poly_idx)
            optimized_polylines.append(optimized)

        result = PolylinesSkeleton(optimized_polylines)

        if self.options.verbose:
            logger.info("Optimization complete")

        return result

    def _optimize_polyline(self, polyline: np.ndarray, poly_idx: int) -> np.ndarray:
        """
        Optimize a single polyline.

        Args:
            polyline: (N, 3) array of points
            poly_idx: Index of this polyline for logging

        Returns:
            Optimized (N, 3) array of points
        """
        points = polyline.copy()
        n_points = len(points)

        if n_points <= 1:
            return points

        for iteration in range(self.options.max_iterations):
            points_old = points.copy()

            for i in range(n_points):
                if self.options.preserve_endpoints and (i == 0 or i == n_points - 1):
                    continue

                # Compute tangent for medial axis centering
                tangent = None
                if self.options.centering_method == "medial_axis":
                    tangent = self._compute_tangent(points, i)

                direction = self._compute_centering_direction(points[i], tangent)

                smoothing_direction = np.zeros(3)
                if self.options.smoothing_weight > 0 and n_points > 2:
                    smoothing_direction = self._compute_smoothing_direction(points, i)

                total_direction = (
                    1.0 - self.options.smoothing_weight
                ) * direction + self.options.smoothing_weight * smoothing_direction

                points[i] = points[i] + self.options.step_size * total_direction

            movement = np.linalg.norm(points - points_old, axis=1).mean()

            if self.options.verbose and iteration % 10 == 0:
                logger.info(
                    "  Polyline %d, iteration %d: avg movement = %.6f",
                    poly_idx,
                    iteration,
                    movement,
                )

            if movement < self.options.convergence_threshold:
                if self.options.verbose:
                    logger.info(
                        "  Polyline %d converged at iteration %d", poly_idx, iteration
                    )
                break

        return points

    def _compute_centering_direction(
        self, point: np.ndarray, tangent: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute the direction to move a point toward the mesh center.

        Args:
            point: (3,) array representing a single point
            tangent: Optional (3,) array representing the skeleton tangent at this point.
                Required for 'medial_axis' centering method.

        Returns:
            (3,) array representing the direction to move (unit vector)
        """
        if self.options.centering_method == "medial_axis":
            return self._compute_medial_axis_direction(point, tangent)
        else:
            return self._compute_closest_point_direction(point)

    def _compute_closest_point_direction(self, point: np.ndarray) -> np.ndarray:
        """
        Simple centering: move toward/away from closest surface point.

        For points outside: move toward the closest surface point
        For points inside: move away from the closest surface point (toward center)

        Args:
            point: (3,) array representing a single point

        Returns:
            (3,) array representing the direction to move (unit vector)
        """
        try:
            from trimesh.proximity import closest_point

            is_inside = self.mesh.contains(point.reshape(1, 3))[0]
            cp, _, _ = closest_point(self.mesh, point.reshape(1, 3))
            surface_point = cp[0]

            to_surface = surface_point - point
            dist_to_surface = np.linalg.norm(to_surface)

            if dist_to_surface < 1e-10:
                return np.zeros(3)

            if not is_inside:
                direction = to_surface / dist_to_surface
            else:
                direction = -to_surface / dist_to_surface

            return direction

        except Exception as e:
            logger.warning("Failed to compute centering direction: %s", e)
            return np.zeros(3)

    def _compute_medial_axis_direction(
        self, point: np.ndarray, tangent: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Medial axis centering: equalize distances in perpendicular directions.

        At the true medial axis, distances to the surface should be equal in all
        directions perpendicular to the skeleton tangent. This method:
        1. Computes perpendicular axes to the tangent
        2. Uses ray tracing to find exact distances to surface in multiple directions
        3. Computes a force to equalize these distances

        Args:
            point: (3,) array representing a single point
            tangent: (3,) array representing the skeleton tangent direction

        Returns:
            (3,) array representing the direction to move
        """
        if tangent is None:
            # Fallback to simple method if no tangent provided
            return self._compute_closest_point_direction(point)

        try:
            # Compute two perpendicular axes to the tangent
            perp1, perp2 = self._compute_perpendicular_axes(tangent)

            # Sample at multiple angles around the tangent
            n_dirs = max(2, self.options.num_probe_directions)

            # Sample at evenly spaced angles around the tangent
            # Each direction is a linear combination of perp1 and perp2
            force = np.zeros(3)

            for i in range(n_dirs):
                # Angle in radians
                angle = 2.0 * np.pi * i / n_dirs

                # Direction vector in the plane perpendicular to tangent
                direction = np.cos(angle) * perp1 + np.sin(angle) * perp2

                # Use ray tracing to find distance to surface in positive direction
                d_pos = self._ray_distance_to_surface(point, direction)

                # Use ray tracing to find distance to surface in negative direction
                d_neg = self._ray_distance_to_surface(point, -direction)

                # Compute force along this direction
                # If d_pos > d_neg, move in positive direction (away from closer surface)
                diff = d_pos - d_neg
                force += diff * direction

            # Normalize to unit vector
            force_mag = np.linalg.norm(force)
            if force_mag > 1e-10:
                return force / force_mag
            else:
                return np.zeros(3)

        except Exception as e:
            logger.warning("Failed to compute medial axis direction: %s", e)
            return self._compute_closest_point_direction(point)

    def _compute_tangent(self, points: np.ndarray, index: int) -> np.ndarray:
        """
        Compute the tangent direction at a point along the polyline.

        Args:
            points: (N, 3) array of all points in the polyline
            index: Index of the current point

        Returns:
            (3,) array representing the tangent direction (unit vector)
        """
        n = len(points)
        if n < 2:
            return np.array([1.0, 0.0, 0.0])  # Default direction

        if index == 0:
            # Use forward difference
            tangent = points[1] - points[0]
        elif index == n - 1:
            # Use backward difference
            tangent = points[n - 1] - points[n - 2]
        else:
            # Use central difference
            tangent = points[index + 1] - points[index - 1]

        tangent_mag = np.linalg.norm(tangent)
        if tangent_mag > 1e-10:
            return tangent / tangent_mag
        else:
            return np.array([1.0, 0.0, 0.0])

    def _ray_distance_to_surface(
        self, point: np.ndarray, direction: np.ndarray
    ) -> float:
        """
        Compute distance from a point to the mesh surface along a ray direction.

        Uses ray tracing to find the exact intersection with the mesh surface.

        Args:
            point: (3,) array representing the ray origin
            direction: (3,) array representing the ray direction (should be normalized)

        Returns:
            Distance to the surface along the ray direction. Returns probe_distance
            if no intersection is found.
        """
        try:
            # Cast a ray from the point in the given direction
            ray_origins = point.reshape(1, 3)
            ray_directions = direction.reshape(1, 3)

            # Find intersections with the mesh
            locations, index_ray, index_tri = self.mesh.ray.intersects_location(
                ray_origins=ray_origins, ray_directions=ray_directions
            )

            if len(locations) == 0:
                # No intersection found - use fallback distance
                if self.options.verbose:
                    logger.debug("Ray found no intersection, using fallback distance")
                return self.options.probe_distance

            # Find the closest intersection point
            distances = np.linalg.norm(locations - point, axis=1)
            min_dist = float(np.min(distances))

            if self.options.verbose and len(locations) > 1:
                logger.debug(
                    "Ray found %d intersections, using closest (%.4f)",
                    len(locations),
                    min_dist,
                )

            return min_dist

        except Exception as e:
            logger.warning("Ray tracing failed, falling back to probe method: %s", e)
            # Fallback to probe distance method
            from trimesh.proximity import closest_point

            probe_pos = point + self.options.probe_distance * direction
            cp, _, _ = closest_point(self.mesh, probe_pos.reshape(1, 3))
            return np.linalg.norm(cp[0] - point)

    def _compute_perpendicular_axes(
        self, tangent: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute two orthonormal axes perpendicular to the tangent.

        Args:
            tangent: (3,) array representing the tangent direction (should be normalized)

        Returns:
            Tuple of two (3,) arrays representing perpendicular axes
        """
        # Normalize tangent
        t = tangent / (np.linalg.norm(tangent) + 1e-10)

        # Find a vector not parallel to tangent
        if abs(t[0]) < 0.9:
            v = np.array([1.0, 0.0, 0.0])
        else:
            v = np.array([0.0, 1.0, 0.0])

        # First perpendicular axis using cross product
        perp1 = np.cross(t, v)
        perp1 = perp1 / (np.linalg.norm(perp1) + 1e-10)

        # Second perpendicular axis
        perp2 = np.cross(t, perp1)
        perp2 = perp2 / (np.linalg.norm(perp2) + 1e-10)

        return perp1, perp2

    def _compute_smoothing_direction(
        self, points: np.ndarray, index: int
    ) -> np.ndarray:
        """
        Compute smoothing direction using Laplacian smoothing.

        Args:
            points: (N, 3) array of all points in the polyline
            index: Index of the current point

        Returns:
            (3,) array representing the smoothing direction
        """
        n = len(points)
        if n <= 2:
            return np.zeros(3)

        if index == 0:
            neighbor_avg = points[1]
        elif index == n - 1:
            neighbor_avg = points[n - 2]
        else:
            neighbor_avg = (points[index - 1] + points[index + 1]) / 2.0

        direction = neighbor_avg - points[index]
        norm = np.linalg.norm(direction)
        if norm > 1e-10:
            return direction / norm
        return np.zeros(3)

    def get_optimization_stats(self) -> dict:
        """
        Get statistics about the optimization process.

        Returns:
            Dictionary containing optimization statistics
        """
        stats = {
            "surface_crossing_detected": self._surface_crossing_detected,
            "num_polylines": len(self.skeleton.polylines),
            "total_points": self.skeleton.total_points(),
        }

        if self.options.check_surface_crossing:
            has_crossing, num_outside, max_dist = self.check_surface_crossing()
            stats["points_outside_mesh"] = num_outside
            stats["max_distance_outside"] = max_dist

        return stats

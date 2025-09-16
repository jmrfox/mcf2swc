"""
Main mesh class
"""

import logging
import multiprocessing
import traceback
from typing import Any, Dict, Optional, Tuple

import numpy as np
import trimesh

# Module-level logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def example_mesh(
    kind: str = "cylinder",
    *,
    # Cylinder params
    radius: float = 1,
    height: float = 10,
    sections: int | None = 16,
    # Torus params
    major_radius: float = 4,
    minor_radius: float = 1,
    major_sections: int | None = 32,
    minor_sections: int | None = 16,
    **kwargs,
) -> trimesh.Trimesh:
    """Create a simple demo mesh using trimesh primitives.

    Parameters
    ----------
    kind : {"cylinder", "torus"}
        Type of primitive to generate. Default "cylinder".
    radius : float
        Cylinder radius (when kind="cylinder"). Default 1.
    height : float
        Cylinder height (when kind="cylinder"). Default 10.
    sections : int or None
        Cylinder radial resolution (pie wedges). Default 16.
    major_radius : float
        Torus major radius (center of hole to centerline of tube). Default 4.
    minor_radius : float
        Torus minor radius (tube radius). Default 1.
    major_sections : int or None
        Torus resolution around major circle. Default 32.
    minor_sections : int or None
        Torus resolution around tube section. Default 16.
    **kwargs : dict
        Passed through to Trimesh constructor via trimesh.creation.* helpers
        (e.g., process=False).

    Returns
    -------
    trimesh.Trimesh
        Generated primitive mesh.

    Examples
    --------
    >>> m = example_mesh("cylinder", radius=0.4, height=1.5)
    >>> t = example_mesh("torus", major_radius=1.0, minor_radius=0.25)
    """
    k = (kind or "cylinder").lower()
    if k == "cylinder":
        return trimesh.creation.cylinder(
            radius=float(radius),
            height=float(height),
            sections=None if sections is None else int(sections),
            **kwargs,
        )
    elif k == "torus":
        # trimesh.creation.torus parameters
        return trimesh.creation.torus(
            major_radius=float(major_radius),
            minor_radius=float(minor_radius),
            major_sections=None if major_sections is None else int(major_sections),
            minor_sections=None if minor_sections is None else int(minor_sections),
            **kwargs,
        )
    else:
        raise ValueError("example_mesh kind must be 'cylinder' or 'torus'")


class MeshManager:
    """
    Unified mesh class handling loading, processing, and analysis.
    """

    def __init__(self, mesh: Optional[trimesh.Trimesh] = None, verbose: bool = True):
        # Core mesh attributes
        self.mesh = mesh

        # Attributes
        self.verbose = verbose
        self.stats = {
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "volume_fixed": 0,
            "watertight_fixed": 0,
            "degenerate_removed": 0,
        }
        
    
    # =================================================================
    # MESH LOADING AND BASIC OPERATIONS
    # =================================================================

    def load_mesh(
        self, filepath: str, file_format: Optional[str] = None
    ) -> trimesh.Trimesh:
        """
        Load a mesh from file.

        Args:
            filepath: Path to mesh file
            file_format: Optional format specification (auto-detected if None)

        Returns:
            Loaded trimesh object
        """
        try:
            if file_format:
                mesh = trimesh.load(filepath, file_type=file_format)
            else:
                mesh = trimesh.load(filepath)

            # Ensure we have a single mesh
            if isinstance(mesh, trimesh.Scene):
                # If it's a scene, try to get the first geometry
                geometries = list(mesh.geometry.values())
                if geometries:
                    mesh = geometries[0]
                else:
                    raise ValueError("No geometry found in mesh scene")

            if not isinstance(mesh, trimesh.Trimesh):
                raise ValueError(f"Loaded object is not a mesh: {type(mesh)}")

            self.mesh = mesh

            if self.verbose:
                logger.info(
                    "Loaded mesh: %d vertices, %d faces",
                    len(mesh.vertices),
                    len(mesh.faces),
                )
                logger.debug("Bounds: %s", self.bounds)

            return mesh

        except Exception as e:
            raise ValueError(f"Failed to load mesh from {filepath}: {str(e)}")

    def save(self, filepath, file_format="obj"):
        self.mesh.export(filepath, file_type=file_format)

    def copy(self):
        return MeshManager(self.mesh.copy())

    def to_trimesh(self):
        return self.mesh


    # combining the functions from utils into this class

    def analyze_mesh(self) -> dict:
        """
        Analyze and return mesh properties for diagnostic purposes.
        This function performs pure analysis without modifying the input mesh.

        Returns:
            Dictionary of mesh properties including volume, watertightness, winding consistency,
            face count, vertex count, bounds, and potential issues.
        """

        mesh = self.to_trimesh()
        # Initialize results dictionary
        results = {
            "face_count": len(mesh.faces),
            "vertex_count": len(mesh.vertices),
            "bounds": mesh.bounds.tolist() if hasattr(mesh, "bounds") else None,
            "is_watertight": mesh.is_watertight,
            "is_winding_consistent": mesh.is_winding_consistent,
            "issues": [],
        }

        # Calculate volume (report actual value, even if negative)
        try:
            results["volume"] = mesh.volume
            if mesh.volume < 0:
                results["issues"].append(
                    "Negative volume detected - face normals may be inverted"
                )
        except Exception as e:
            results["volume"] = None
            results["issues"].append(f"Volume calculation failed: {str(e)}")

        # Check for non-manifold edges
        try:
            if hasattr(mesh, "is_manifold"):
                results["is_manifold"] = mesh.is_manifold
                if not mesh.is_manifold:
                    results["issues"].append("Non-manifold edges detected")
        except Exception:
            results["is_manifold"] = None

        # Calculate topological properties using trimesh's built-in methods
        try:
            # Use trimesh's built-in euler_number property for correct topology calculation
            # For a sphere: euler_number = 2
            # For a torus: euler_number = 0
            # For a double torus: euler_number = -2
            # Genus = (2 - euler_number) / 2

            results["euler_characteristic"] = mesh.euler_number

            # Only calculate genus for closed (watertight) meshes
            if mesh.is_watertight:
                # For a closed orientable surface: genus = (2 - euler_number) / 2
                results["genus"] = int((2 - mesh.euler_number) / 2)

                # Sanity check - genus should be non-negative for simple shapes
                if results["genus"] < 0:
                    results["genus"] = (
                        0  # Default to 0 for simple shapes like spheres, cylinders
                    )
                    results["issues"].append(
                        "Calculated negative genus, defaulting to 0"
                    )
            else:
                # For non-watertight meshes, genus is not well-defined
                results["genus"] = None
                results["issues"].append("Genus undefined for non-watertight mesh")
        except Exception as e:
            results["genus"] = None
            results["euler_characteristic"] = None
            results["issues"].append(f"Topology calculation failed: {str(e)}")

        # Analyze face normals
        try:
            if hasattr(mesh, "face_normals") and mesh.face_normals is not None:
                # Get statistics on face normal directions
                results["normal_stats"] = {
                    "mean": mesh.face_normals.mean(axis=0).tolist(),
                    "std": mesh.face_normals.std(axis=0).tolist(),
                    "sum": mesh.face_normals.sum(axis=0).tolist(),
                }

                # Check if normals are predominantly pointing inward (negative volume)
                if results.get("volume", 0) < 0:
                    results["normal_direction"] = "inward"
                else:
                    results["normal_direction"] = "outward"
        except Exception as e:
            results["normal_stats"] = None
            results["issues"].append(f"Normal analysis failed: {str(e)}")

        # Check for duplicate vertices and faces
        try:
            unique_verts = np.unique(mesh.vertices, axis=0)
            results["duplicate_vertices"] = len(mesh.vertices) - len(unique_verts)
            if results["duplicate_vertices"] > 0:
                results["issues"].append(
                    f"Found {results['duplicate_vertices']} duplicate vertices"
                )
        except Exception:
            results["duplicate_vertices"] = None

        # Check for degenerate faces (zero area)
        try:
            if hasattr(mesh, "area_faces"):
                degenerate_count = np.sum(mesh.area_faces < 1e-8)
                results["degenerate_faces"] = int(degenerate_count)
                if degenerate_count > 0:
                    results["issues"].append(
                        f"Found {degenerate_count} degenerate faces"
                    )
        except Exception:
            results["degenerate_faces"] = None

        # Check for connected components
        try:
            components = mesh.split(only_watertight=False)
            results["component_count"] = len(components)
            if len(components) > 1:
                results["issues"].append(
                    f"Mesh has {len(components)} disconnected components"
                )
        except Exception:
            results["component_count"] = None

        return results

    def print_mesh_analysis(self, verbose: bool = False) -> None:
        """
        Analyze a mesh and print a formatted report of its properties.

        Args:
            verbose: Whether to print detailed information
        """
        analysis = self.analyze_mesh()

        logger.info("Mesh Analysis Report")
        logger.info("====================")

        # Basic properties
        logger.info("\nGeometry:")
        logger.info("  * Vertices: %s", analysis["vertex_count"])
        logger.info("  * Faces: %s", analysis["face_count"])
        if analysis.get("component_count") is not None:
            logger.info("  * Components: %s", analysis["component_count"])
        if analysis.get("volume") is not None:
            logger.info("  * Volume: %.2f", analysis["volume"])
        if analysis.get("bounds") is not None:
            min_bound, max_bound = analysis["bounds"]
            logger.info(
                "  * Bounds: [%.1f, %.1f, %.1f] to [%.1f, %.1f, %.1f]",
                min_bound[0],
                min_bound[1],
                min_bound[2],
                max_bound[0],
                max_bound[1],
                max_bound[2],
            )

        # Mesh quality
        logger.info("\nMesh Quality:")
        logger.info("  * Watertight: %s", analysis["is_watertight"])
        logger.info("  * Winding Consistent: %s", analysis["is_winding_consistent"])
        if analysis.get("is_manifold") is not None:
            logger.info("  * Manifold: %s", analysis["is_manifold"])
        if analysis.get("normal_direction") is not None:
            logger.info("  * Normal Direction: %s", analysis["normal_direction"])
        if analysis.get("duplicate_vertices") is not None:
            logger.info("  * Duplicate Vertices: %s", analysis["duplicate_vertices"])
        if analysis.get("degenerate_faces") is not None:
            logger.info("  * Degenerate Faces: %s", analysis["degenerate_faces"])

        # Topology
        if (
            analysis.get("genus") is not None
            or analysis.get("euler_characteristic") is not None
        ):
            logger.info("\nTopology:")
            if analysis.get("genus") is not None:
                logger.info("  * Genus: %s", analysis["genus"])
            if analysis.get("euler_characteristic") is not None:
                logger.info(
                    "  * Euler Characteristic: %s", analysis["euler_characteristic"]
                )

        # Issues
        if analysis["issues"]:
            logger.info("\nIssues Detected (%d):", len(analysis["issues"]))
            for i, issue in enumerate(analysis["issues"]):
                logger.info("  %d. %s", i + 1, issue)
        else:
            logger.info("\nNo issues detected")

        # Detailed stats
        if verbose and analysis.get("normal_stats") is not None:
            logger.info("\nNormal Statistics:")
            mean = analysis["normal_stats"]["mean"]
            sum_val = analysis["normal_stats"]["sum"]
            logger.info("  * Mean: [%.4f, %.4f, %.4f]", mean[0], mean[1], mean[2])
            logger.info(
                "  * Sum: [%.4f, %.4f, %.4f]", sum_val[0], sum_val[1], sum_val[2]
            )

        logger.info("\nRecommendation:")
        if analysis["issues"]:
            logger.info("  Consider using repair_mesh() to fix the detected issues.")
        else:
            logger.info("  Mesh appears to be in good condition.")
        logger.info("====================")

    def repair_mesh(
        self,
        fix_holes: bool = True,
        remove_duplicates: bool = True,
        fix_normals: bool = True,
        remove_degenerate: bool = True,
        fix_negative_volume: bool = True,
        keep_largest_component: bool = False,
        verbose: bool = True,
    ) -> trimesh.Trimesh:
        """
        Attempt to repair common mesh issues to improve watertightness and quality.

        Args:
            mesh_data: Either a Trimesh object or (vertices, faces) tuple
            fix_holes: Whether to attempt filling holes
            remove_duplicates: Whether to remove duplicate faces and vertices
            fix_normals: Whether to fix face normal consistency
            remove_degenerate: Whether to remove degenerate faces
            fix_negative_volume: Whether to invert faces if mesh has negative volume
            keep_largest_component: Whether to keep only the largest connected component
            verbose: Whether to print repair summary

        Returns:
            Repaired mesh (new copy, original is not modified)
        """

        mesh = self.to_trimesh()

        repair_log = []

        # Fix negative volume by inverting faces if needed
        if fix_negative_volume:
            try:
                # Check if the mesh has a negative volume
                if hasattr(mesh, "volume") and mesh.volume < 0:
                    initial_volume = mesh.volume
                    mesh.invert()
                    repair_log.append(
                        f"Inverted faces to fix negative volume: {initial_volume:.2f} → {mesh.volume:.2f}"
                    )
            except Exception as e:
                repair_log.append(f"Failed to fix negative volume: {e}")

        # Remove duplicate and degenerate faces
        if remove_duplicates:
            try:
                initial_faces = len(mesh.faces)
                mesh.remove_duplicate_faces()
                removed_faces = initial_faces - len(mesh.faces)
                if removed_faces > 0:
                    repair_log.append(f"Removed {removed_faces} duplicate faces")
            except Exception as e:
                repair_log.append(f"Failed to remove duplicate faces: {e}")

        if remove_degenerate:
            try:
                initial_faces = len(mesh.faces)
                mesh.remove_degenerate_faces()
                removed_faces = initial_faces - len(mesh.faces)
                if removed_faces > 0:
                    repair_log.append(f"Removed {removed_faces} degenerate faces")
            except Exception as e:
                repair_log.append(f"Failed to remove degenerate faces: {e}")

        # Fix winding consistency
        if fix_normals:
            try:
                if not mesh.is_winding_consistent:
                    mesh.fix_normals()
                    if mesh.is_winding_consistent:
                        repair_log.append("Fixed face normal winding consistency")
                    else:
                        repair_log.append(
                            "Attempted to fix normals but still inconsistent"
                        )
            except Exception as e:
                repair_log.append(f"Failed to fix normals: {e}")

        # Attempt to fill holes
        if fix_holes:
            try:
                if not mesh.is_watertight:
                    initial_watertight = mesh.is_watertight
                    mesh.fill_holes()
                    if mesh.is_watertight and not initial_watertight:
                        repair_log.append(
                            "Successfully filled holes - mesh is now watertight"
                        )
                    elif mesh.is_watertight:
                        repair_log.append("Mesh was already watertight")
                    else:
                        repair_log.append(
                            "Attempted to fill holes but mesh still not watertight"
                        )
            except Exception as e:
                repair_log.append(f"Failed to fill holes: {e}")

        # Keep only the largest component if requested
        if keep_largest_component:
            try:
                components = mesh.split(only_watertight=False)
                if len(components) > 1:
                    # Keep the largest component by volume or face count
                    volumes = [
                        abs(c.volume) if hasattr(c, "volume") else len(c.faces)
                        for c in components
                    ]
                    largest_idx = np.argmax(volumes)
                    mesh = components[largest_idx]
                    repair_log.append(
                        f"Kept largest of {len(components)} components (volume: {volumes[largest_idx]:.2f})"
                    )
            except Exception as e:
                repair_log.append(f"Failed to isolate largest component: {e}")

        # Final processing to ensure consistency
        try:
            mesh.process(validate=True)
            repair_log.append("Applied final mesh processing and validation")
        except Exception as e:
            repair_log.append(f"Final processing failed: {e}")

        # Store repair log as mesh metadata
        if not hasattr(mesh, "metadata"):
            mesh.metadata = {}
        mesh.metadata["repair_log"] = repair_log

        # Log repair summary
        if verbose:
            if repair_log:
                logger.info("Mesh Repair Summary:")
                for log_entry in repair_log:
                    logger.info("  • %s", log_entry)

                # Final mesh status
                logger.info("\nFinal Mesh Status:")
                logger.info(
                    "  • Volume: %s",
                    mesh.volume if hasattr(mesh, "volume") else "N/A",
                )
                logger.info("  • Watertight: %s", mesh.is_watertight)
                logger.info("  • Winding consistent: %s", mesh.is_winding_consistent)
                logger.info("  • Faces: %d", len(mesh.faces))
                logger.info("  • Vertices: %d", len(mesh.vertices))
            else:
                logger.info("No repairs needed - mesh is in good condition")

        self.mesh = mesh
        return mesh


    def visualize_mesh_3d(
        self,
        title: str = "3D Mesh Visualization",
        color: str = "lightblue",
        backend: str = "auto",
        show_axes: bool = True,
        show_wireframe: bool = False,
        width: int = 800,
        height: int = 600,
        *,
        polylines: Optional["PolylinesSkeleton"] = None,
        poly_color: str = "crimson",
        poly_line_width: float = 3.0,
        poly_opacity: float = 0.95,
    ) -> Optional[object]:
        """
        Create a 3D visualization of a mesh.

        Args:
            title: Plot title
            color: Mesh color (named color or RGB tuple)
            backend: Visualization backend ('plotly' or 'matplotlib')
            show_axes: Whether to show coordinate axes
            show_wireframe: Whether to show wireframe overlay
            polylines: Optional PolylinesSkeleton to overlay as 3D lines
            poly_color: Color for polylines overlay
            poly_line_width: Line width for polylines overlay
            poly_opacity: Opacity for polylines overlay (plotly only)

        Returns:
            Figure object (backend-dependent) or None if visualization fails
        """
        if backend == "auto":
            # Try plotly first, then fallback to matplotlib
            try:
                import plotly.graph_objects as go  # noqa: F401
                backend = "plotly"
            except ImportError:
                try:
                    import matplotlib.pyplot as plt  # noqa: F401
                    backend = "matplotlib"
                except ImportError:
                    backend = "plotly"

        if backend == "plotly":
            return self._visualize_mesh_plotly(
                title,
                color,
                show_axes,
                show_wireframe,
                width,
                height,
                polylines=polylines,
                poly_color=poly_color,
                poly_line_width=poly_line_width,
                poly_opacity=poly_opacity,
            )
        elif backend == "matplotlib":
            return self._visualize_mesh_matplotlib(
                title,
                color,
                show_axes,
                show_wireframe,
                polylines=polylines,
                poly_color=poly_color,
                poly_line_width=poly_line_width,
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _visualize_mesh_plotly(
        self,
        title,
        color,
        show_axes,
        show_wireframe,
        width=800,
        height=600,
        *,
        polylines: Optional["PolylinesSkeleton"] = None,
        poly_color: str = "crimson",
        poly_line_width: float = 3.0,
        poly_opacity: float = 0.95,
    ):
        """Plotly-based mesh visualization with optional polylines overlay."""
        try:
            import plotly.graph_objects as go

            vertices = self.mesh.vertices
            faces = self.mesh.faces

            # Create mesh trace
            mesh_trace = go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                opacity=0.8,
                color=color,
                name="Mesh",
            )

            fig = go.Figure(data=[mesh_trace])

            # Add wireframe if requested
            if show_wireframe:
                edge_x = []
                edge_y = []
                edge_z = []
                for face in faces:
                    for i in range(3):
                        v1, v2 = face[i], face[(i + 1) % 3]
                        edge_x += [vertices[v1][0], vertices[v2][0], None]
                        edge_y += [vertices[v1][1], vertices[v2][1], None]
                        edge_z += [vertices[v1][2], vertices[v2][2], None]
                fig.add_trace(
                    go.Scatter3d(
                        x=edge_x,
                        y=edge_y,
                        z=edge_z,
                        mode="lines",
                        line=dict(color="black", width=1),
                        name="Wireframe",
                    )
                )

            # Add polylines overlay if provided
            if polylines is not None and hasattr(polylines, "polylines"):
                for idx, pl in enumerate(polylines.polylines):
                    if pl is None:
                        continue
                    pts = np.asarray(pl, dtype=float)
                    if pts.ndim == 2 and pts.shape[1] >= 3 and pts.shape[0] >= 2:
                        fig.add_trace(
                            go.Scatter3d(
                                x=pts[:, 0],
                                y=pts[:, 1],
                                z=pts[:, 2],
                                mode="lines",
                                line=dict(color=poly_color, width=float(poly_line_width)),
                                opacity=float(poly_opacity),
                                name=f"Polyline {idx}",
                                showlegend=False,
                            )
                        )

            # Configure layout
            fig.update_layout(
                title=title,
                autosize=False,
                width=width,
                height=height,
                scene=dict(
                    aspectmode="data",
                    xaxis=dict(visible=show_axes),
                    yaxis=dict(visible=show_axes),
                    zaxis=dict(visible=show_axes),
                ),
            )

            return fig

        except ImportError:
            print("Plotly not available")
            return None
        except Exception as e:
            print(f"Plotly visualization failed: {e}")
            return None

    def _visualize_mesh_matplotlib(
        self,
        title,
        color,
        show_axes,
        show_wireframe,
        *,
        polylines: Optional["PolylinesSkeleton"] = None,
        poly_color: str = "crimson",
        poly_line_width: float = 3.0,
    ):
        """Matplotlib-based mesh visualization with optional polylines overlay."""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")

            vertices = self.mesh.vertices
            faces = self.mesh.faces

            # Create mesh surface
            poly3d = Poly3DCollection(
                vertices[faces],
                alpha=0.7,
                facecolor=color,
                edgecolor="black" if show_wireframe else None,
            )
            ax.add_collection3d(poly3d)

            # Add polylines overlay if provided
            if polylines is not None and hasattr(polylines, "polylines"):
                for pl in polylines.polylines:
                    if pl is None:
                        continue
                    pts = np.asarray(pl, dtype=float)
                    if pts.ndim == 2 and pts.shape[1] >= 3 and pts.shape[0] >= 2:
                        ax.plot(
                            pts[:, 0],
                            pts[:, 1],
                            pts[:, 2],
                            color=poly_color,
                            linewidth=float(poly_line_width),
                        )

            ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
            ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
            ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())

            ax.set_xlabel("X (µm)")
            ax.set_ylabel("Y (µm)")
            ax.set_zlabel("Z (µm)")
            ax.set_title(title)

            if not show_axes:
                ax.set_axis_off()

            plt.tight_layout()
            return fig

        except ImportError:
            print("Matplotlib not available")
            return None
        except Exception as e:
            print(f"Matplotlib visualization failed: {e}")
            return None

    def visualize_mesh_slice_interactive(
        self,
        title: str = "Interactive Mesh Slice",
        z_range: Optional[Tuple[float, float]] = None,
        num_slices: int = 50,
        slice_color: str = "red",
        mesh_color: str = "lightblue",
        mesh_opacity: float = 0.3,
    ) -> Optional[object]:
        """
        Create an interactive 3D visualization of a mesh with a controllable slice plane.

        This function displays a 3D mesh and calculates the intersection of the mesh
        with an xy-plane at a user-controlled z-value. The intersection is shown as a
        colored line on the mesh. A slider allows the user to interactively change the
        z-value of the intersection plane.

        Args:
            title: Plot title
            z_range: Tuple of (min_z, max_z) for slice range. Auto-detected if None.
            num_slices: Number of positions for the slider
            slice_color: Color for the intersection line
            mesh_color: Color for the 3D mesh
            mesh_opacity: Opacity of the 3D mesh (0-1)

        Returns:
            Plotly figure with interactive slider for controlling the z-value
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("Plotly is required for interactive visualization")
            return None

        mesh = self.mesh

        # Determine z-range if not provided
        if z_range is None:
            z_min, z_max = mesh.vertices[:, 2].min(), mesh.vertices[:, 2].max()
            # Add small padding
            padding = (z_max - z_min) * 0.05
            z_min -= padding
            z_max += padding
        else:
            z_min, z_max = z_range

        # Create the base figure with the mesh
        fig = go.Figure()

        # Add the mesh to the figure
        fig.add_trace(
            go.Mesh3d(
                x=mesh.vertices[:, 0],
                y=mesh.vertices[:, 1],
                z=mesh.vertices[:, 2],
                i=mesh.faces[:, 0],
                j=mesh.faces[:, 1],
                k=mesh.faces[:, 2],
                opacity=mesh_opacity,
                color=mesh_color,
                name="Mesh",
            )
        )

        # Function to create a slice at a given z-value
        def create_slice_trace(z_value):
            # Calculate intersection with plane at z_value
            section = mesh.section(plane_origin=[0, 0, z_value], plane_normal=[0, 0, 1])

            # If no intersection, return None
            if (
                section is None
                or not hasattr(section, "entities")
                or len(section.entities) == 0
            ):
                return None

            # Process all entities in the section to get 3D coordinates
            all_points = []

            for entity in section.entities:
                if hasattr(entity, "points") and len(entity.points) > 0:
                    # Get the actual 2D coordinates
                    points_2d = section.vertices[entity.points]

                    # Convert to 3D by adding z_value
                    points_3d = np.column_stack(
                        [points_2d, np.full(len(points_2d), z_value)]
                    )

                    # Add closing point if needed (to complete the loop)
                    if len(points_2d) > 2 and not np.array_equal(
                        points_2d[0], points_2d[-1]
                    ):
                        closing_point = np.array(
                            [points_2d[0][0], points_2d[0][1], z_value]
                        )
                        points_3d = np.vstack([points_3d, closing_point])

                    # Add to all points list
                    all_points.extend(points_3d.tolist())

                    # Add None to create a break between separate entities
                    all_points.append([None, None, None])

            # If we have points, create a scatter trace
            if all_points:
                x_coords = [p[0] if p is not None else None for p in all_points]
                y_coords = [p[1] if p is not None else None for p in all_points]
                z_coords = [p[2] if p is not None else None for p in all_points]

                return go.Scatter3d(
                    x=x_coords,
                    y=y_coords,
                    z=z_coords,
                    mode="lines",
                    line=dict(color=slice_color, width=5),
                    name=f"Slice at z={z_value:.2f}",
                )

            return None

        # Create initial slice
        initial_z = (z_min + z_max) / 2
        initial_slice = create_slice_trace(initial_z)

        # Add initial slice to figure if it exists
        if initial_slice:
            fig.add_trace(initial_slice)

        # Create frames for animation
        frames = []
        for i, z_val in enumerate(np.linspace(z_min, z_max, num_slices)):
            # Create a slice at this z-value
            slice_trace = create_slice_trace(z_val)

            # If we have a valid slice, add it to frames
            if slice_trace:
                frame_data = [fig.data[0], slice_trace]  # Mesh and slice
            else:
                frame_data = [fig.data[0]]  # Just the mesh

            frames.append(
                go.Frame(
                    data=frame_data,
                    name=f"frame_{i}",
                    traces=[0, 1],  # Update both traces
                )
            )

        # Create slider steps
        steps = []
        for i, z_val in enumerate(np.linspace(z_min, z_max, num_slices)):
            step = dict(
                args=[
                    [f"frame_{i}"],
                    {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"},
                ],
                label=f"{z_val:.2f}",
                method="animate",
            )
            steps.append(step)

        # Configure the slider
        sliders = [
            dict(
                active=num_slices // 2,  # Start in the middle
                currentvalue={
                    "prefix": "Z-value: ",
                    "visible": True,
                    "xanchor": "right",
                },
                pad={"t": 50, "b": 10},
                len=0.9,
                x=0.1,
                y=0,
                steps=steps,
            )
        ]

        # Configure the figure layout
        fig.update_layout(
            title=title,
            scene=dict(aspectmode="data", camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))),
            height=800,  # Taller to make room for slider
            margin=dict(l=50, r=50, b=100, t=100),  # Add margin at bottom for slider
            sliders=sliders,
            # Add animation controls
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=0,
                    x=0,
                    xanchor="left",
                    yanchor="top",
                    pad=dict(t=60, r=10),
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                {
                                    "frame": {"duration": 200, "redraw": True},
                                    "fromcurrent": True,
                                },
                            ],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                },
                            ],
                        ),
                        dict(
                            label="Reset View",
                            method="relayout",
                            args=[{"scene.camera.eye": dict(x=1.5, y=1.5, z=1.5)}],
                        ),
                    ],
                )
            ],
        )

        # Set frames
        fig.frames = frames

        return fig


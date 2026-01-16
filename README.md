# mcf2swc

This is a Python package designed to take a 3D closed mesh surface (obj format, triangle-faced mesh) and the results of a mean curvature flow calculation (MCF) (polylines format) and produce an SWC model.

[Triangle mesh format](https://en.wikipedia.org/wiki/Triangle_mesh)

[SWC format](http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html)

Polylines text format is
```N x1 y1 z1 x2 y2 z2 ... xN yN zN```
for each branch of the model.

## Core Classes

The package provides three main classes for working with neuron morphology:

- **`MeshManager`**: Handles 3D mesh surfaces (OBJ format)
- **`SkeletonGraph`**: Graph-based skeleton representation (loads from polylines format)
- **`MorphologyGraph`**: SWC-like morphology model produced by `fit_morphology`, can be saved to SWC format with cycle annotations

## Algorithm

The main purpose of this package is to take a closed triangle mesh and the output of a mean curvature flow calculation (MCF) and convert it into an SWC model. The algorithm is as follows:

1. Load the triangle mesh using `MeshManager` and the MCF polylines using `SkeletonGraph`.
2. Optionally optimize the skeleton using `SkeletonOptimizer` to improve medial axis alignment.
3. Use `fit_morphology` to determine junction locations along the skeleton and estimate radii.
4. The result is a `MorphologyGraph` containing the skeleton with radii information.
5. Export the graph as an SWC file.

Since SWC format does not support cycles, a graph with cycles is broken by locating a duplicate node and rewiring one incident cycle edge to the duplicate.

## Mean Curvature Flow using CGAL

The mean curvature flow (MCF) is a geometric evolution of a surface in 3D space. It is a process where the surface evolves over time, with the rate of change of the surface at each point determined by the mean curvature at that point. The mean curvature flow is a way to smooth out the surface and make it more regular, while preserving its overall shape and topology.

You can easily compute the MCF data using the [Computational Geometry Algorithms Library (CGAL)](https://doc.cgal.org/latest/Surface_mesh_skeletonization/index.html) package. CGAL provides a standalone GUI program called `CGALLab` that includes the MCF skeletonization operation. You can download the Windows demo version of it here: [CGALLab Windows Demo](https://doc.cgal.org/latest/Manual/packages.html#:~:text=Triangulated%20Surface%20Mesh%20Skeletonization).

Simply open the program, load your mesh, go to `Operations -> Triangulated Surface Mesh Skeletonization -> Mean Curvature Skeleton (Advanced)`, apply the operation, then right-click on the new object labelled "fixed points" and save it to a `polylines.txt` file.

## Skeleton Optimization

MCF skeletonization reproduces general topology very well, but the resulting skeleton can deviate from the true medial axis and sometimes even clip through the mesh surface, especially in regions with holes or high curvature. The `SkeletonOptimizer` module provides optimization to refine MCF-generated skeleton polylines by gently pushing points toward the center of the mesh volume.

### Features

- **Surface crossing detection**: Automatically detects skeleton points that are outside the mesh surface
- **Medial axis optimization**: Pushes skeleton points toward the mesh center while preserving topology
- **Smoothing regularization**: Maintains polyline smoothness during optimization
- **Endpoint preservation**: Optional preservation of polyline endpoints
- **Configurable parameters**: Control optimization behavior with flexible options

### Usage Example

```python
from mcf2swc import MeshManager, SkeletonGraph, SkeletonOptimizer, SkeletonOptimizerOptions

# Load mesh and skeleton
mesh_mgr = MeshManager(mesh_path="mesh.obj")
skeleton = SkeletonGraph.from_txt("skeleton.polylines.txt")

# Configure optimization
opts = SkeletonOptimizerOptions(
    max_iterations=100,
    step_size=0.1,
    convergence_threshold=1e-4,
    preserve_terminal_nodes=True,  # Preserve endpoints
    preserve_branch_nodes=False,   # Allow branch nodes to move
    smoothing_weight=0.5,
    n_rays=6,  # Number of rays for medial axis estimation
    verbose=True
)

# Optimize skeleton
optimizer = SkeletonOptimizer(skeleton, mesh_mgr.mesh, opts)
optimized_skeleton = optimizer.optimize()

# Save optimized skeleton (GraphML format)
optimized_skeleton.to_txt("skeleton_optimized.graphml")
```

### Optimization Strategy

The optimizer uses an iterative approach:

1. **For points outside the mesh**: Move toward the closest surface point
2. **For points inside the mesh**: Cast rays in multiple directions to estimate distance to surface, then move toward the medial axis based on distance imbalance
3. **Apply smoothing**: Use Laplacian smoothing to maintain skeleton continuity
4. **Iterate until convergence**: Repeat until point movement falls below threshold

The optimization balances two objectives:

- **Centering**: Push points toward the medial axis using multi-directional ray sampling
- **Smoothing**: Maintain smooth skeleton structure

The relative weight of these objectives is controlled by the `smoothing_weight` parameter (0 = no smoothing, 1 = strong smoothing). The `n_rays` parameter controls how many directions are sampled for medial axis estimation (default: 6 for axis-aligned rays).

## Complete Workflow Example

```python
from mcf2swc import MeshManager, SkeletonGraph, SkeletonOptimizer, SkeletonOptimizerOptions, fit_morphology, FitOptions

# 1. Load inputs
mesh_mgr = MeshManager(mesh_path="neuron.obj")
skeleton = SkeletonGraph.from_txt("neuron_mcf.polylines.txt")

# 2. (Optional) Optimize skeleton for better medial axis alignment
opt_options = SkeletonOptimizerOptions(
    max_iterations=50,
    step_size=0.1,
    preserve_terminal_nodes=True,
    smoothing_weight=0.5
)
optimizer = SkeletonOptimizer(skeleton, mesh_mgr.mesh, opt_options)
optimized_skeleton = optimizer.optimize()

# 3. Fit morphology (estimate radii and build SWC-like graph)
fit_options = FitOptions(
    spacing=1.0,  # Sample spacing along skeleton
    radius_strategy="equivalent_area"  # How to estimate radii
)
morphology = fit_morphology(mesh_mgr.mesh, optimized_skeleton, fit_options)

# 4. Convert to SWC and save
swc_model = morphology.to_swc()
swc_model.to_swc("neuron.swc")
```

The resulting SWC file will have cycle annotations if any cycles were detected in the skeleton topology.

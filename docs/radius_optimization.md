# Radius Optimization

## Overview

The `radius_optimizer` module provides an optimization-based approach to determining node radii in skeleton graphs. Unlike the procedural strategies in `trace.py` (which compute each radius independently using local geometric measurements), this module treats **all radii as parameters in a joint optimization problem**.

The optimizer minimizes a loss function that measures how well the SWC model (represented as a collection of frusta) approximates the original mesh.

## Key Concepts

### Procedural vs. Optimization-Based Approaches

**Procedural approach** (`trace.py`):
- Each node's radius is computed independently
- Uses local geometric measurements (cross-section area, median distance, etc.)
- Fast and simple
- May not produce globally optimal results

**Optimization-based approach** (`radius_optimizer.py`):
- All radii are optimized jointly
- Minimizes a global loss function (e.g., surface area error)
- Uses powerful optimization algorithms (L-BFGS-B, SLSQP, etc.)
- Can produce better approximations of the original mesh

### Loss Functions

The optimizer supports multiple loss functions:

1. **`surface_area`**: Minimizes the relative error between SWC model surface area and mesh surface area
   - Best for preserving surface properties
   - Formula: `|A_swc - A_mesh| / A_mesh`

2. **`volume`**: Minimizes the relative error between SWC model volume and mesh volume
   - Best for preserving volumetric properties
   - Formula: `|V_swc - V_mesh| / V_mesh`

3. **`combined`**: Weighted combination of surface area and volume errors
   - Balances both surface and volumetric properties
   - Formula: `w_area * area_error + w_volume * volume_error`

### SWC Model Geometry

The SWC model represents each edge as a **frustum** (truncated cone):
- **Surface area**: `A = π(r₁ + r₂) * √(h² + (r₁ - r₂)²)`
- **Volume**: `V = (πh/3) * (r₁² + r₁r₂ + r₂²)`

where `r₁` and `r₂` are the radii at the two endpoints, and `h` is the edge length.

## Usage

### Basic Usage

```python
from mcf2swc import (
    build_traced_skeleton_graph,
    optimize_skeleton_radii,
    OptimizerOptions,
)

# Step 1: Build initial skeleton with trace.py
skeleton = build_traced_skeleton_graph(mesh, polylines)

# Step 2: Optimize radii
optimized_skeleton = optimize_skeleton_radii(skeleton, mesh)

# Step 3: Export to SWC
optimized_skeleton.to_swc_file("output.swc")
```

### Customizing Optimization

```python
from mcf2swc import OptimizerOptions, optimize_skeleton_radii

# Configure optimizer
options = OptimizerOptions(
    loss_function="surface_area",  # or "volume", "combined"
    optimizer="scipy_lbfgsb",      # or "scipy_slsqp", "scipy_minimize"
    min_radius=0.1,                # minimum allowed radius
    max_radius=10.0,               # maximum allowed radius
    max_iterations=1000,           # optimization budget
    tolerance=1e-6,                # convergence tolerance
    verbose=True,                  # print progress
)

optimized = optimize_skeleton_radii(skeleton, mesh, options=options)
```

### Combined Loss Function

```python
options = OptimizerOptions(
    loss_function="combined",
    loss_weights={
        "surface_area": 2.0,  # 2x weight on surface area
        "volume": 1.0,        # 1x weight on volume
    },
)

optimized = optimize_skeleton_radii(skeleton, mesh, options=options)
```

### Advanced Usage with RadiusOptimizer Class

```python
from mcf2swc.radius_optimizer import RadiusOptimizer, OptimizerOptions

# Create optimizer instance
optimizer = RadiusOptimizer(skeleton, mesh, options=options)

# Access initial state
initial_radii = optimizer.get_initial_radii()
initial_loss = optimizer.compute_loss(initial_radii)

# Run optimization
optimized_skeleton = optimizer.optimize()

# Inspect optimization history
for entry in optimizer.history:
    print(f"Iteration {entry['iteration']}: loss = {entry['loss']}")
```

## API Reference

### `OptimizerOptions`

Configuration dataclass for radius optimization.

**Attributes:**
- `loss_function` (str): Loss function to minimize. Options: `"surface_area"`, `"volume"`, `"combined"`. Default: `"surface_area"`
- `loss_weights` (dict, optional): Weights for combined loss. E.g., `{"surface_area": 1.0, "volume": 0.5}`
- `optimizer` (str): Optimization algorithm. Options: `"scipy_lbfgsb"`, `"scipy_slsqp"`, `"scipy_minimize"`. Default: `"scipy_lbfgsb"`
- `min_radius` (float): Minimum allowed radius. Default: `0.01`
- `max_radius` (float, optional): Maximum allowed radius. Default: `None` (unbounded)
- `max_iterations` (int): Maximum optimization iterations. Default: `1000`
- `tolerance` (float): Convergence tolerance. Default: `1e-6`
- `verbose` (bool): Print optimization progress. Default: `False`

### `RadiusOptimizer`

Main optimizer class.

**Methods:**
- `__init__(skeleton, mesh, *, options=None)`: Initialize optimizer
- `get_initial_radii()`: Extract current radii as numpy array
- `set_radii(radii)`: Update skeleton with new radii
- `compute_swc_surface_area(radii)`: Compute total surface area of SWC model
- `compute_swc_volume(radii)`: Compute total volume of SWC model
- `compute_loss(radii)`: Compute loss function value
- `optimize()`: Run optimization and return new SkeletonGraph

**Attributes:**
- `history`: List of optimization checkpoints (when `verbose=True`)

### `optimize_skeleton_radii()`

Convenience function for radius optimization.

**Signature:**
```python
optimize_skeleton_radii(
    skeleton: SkeletonGraph,
    mesh: trimesh.Trimesh,
    *,
    options: Optional[OptimizerOptions] = None,
) -> SkeletonGraph
```

**Returns:** New `SkeletonGraph` with optimized radii

## Workflow Integration

### Typical Pipeline

```python
from mcf2swc import (
    MeshManager,
    PolylinesSkeleton,
    TraceOptions,
    build_traced_skeleton_graph,
    OptimizerOptions,
    optimize_skeleton_radii,
)

# 1. Load mesh and polylines
mesh_mgr = MeshManager(mesh_path="neuron.obj")
polylines = PolylinesSkeleton.from_file("skeleton.polylines.txt")

# 2. Build initial skeleton with procedural radius estimates
trace_opts = TraceOptions(
    spacing=1.0,
    radius_strategy="equivalent_area",  # Initial estimate
)
skeleton = build_traced_skeleton_graph(
    mesh_mgr.mesh,
    polylines,
    options=trace_opts,
)

# 3. Refine radii using optimization
optimizer_opts = OptimizerOptions(
    loss_function="surface_area",
    max_iterations=500,
    verbose=True,
)
optimized_skeleton = optimize_skeleton_radii(
    skeleton,
    mesh_mgr.mesh,
    options=optimizer_opts,
)

# 4. Export final result
optimized_skeleton.to_swc_file("neuron_optimized.swc")
```

### When to Use Optimization

**Use optimization when:**
- You need high-fidelity approximation of the mesh
- Surface area or volume preservation is critical
- You have computational budget for optimization
- Initial procedural estimates are poor

**Stick with procedural methods when:**
- Speed is critical
- Initial estimates are already good
- The mesh is very complex (optimization may be slow)
- You're doing exploratory analysis

## Performance Considerations

### Computational Cost

- **Time complexity**: O(iterations × nodes × edges) for gradient computation
- **Memory**: O(nodes) for parameter vector
- **Typical runtime**: Seconds to minutes depending on skeleton size

### Optimization Tips

1. **Good initialization**: Start with reasonable procedural estimates (e.g., `equivalent_area`)
2. **Appropriate bounds**: Set `min_radius` and `max_radius` based on mesh scale
3. **Loss function choice**: 
   - Use `surface_area` for thin structures
   - Use `volume` for bulky structures
   - Use `combined` for balanced results
4. **Iteration budget**: Start with 100-500 iterations; increase if not converged

## Examples

See `examples/radius_optimization_demo.py` for complete working examples including:
- Basic optimization
- Comparing loss functions
- Using constraints
- Advanced usage with the `RadiusOptimizer` class

## Limitations

1. **Local minima**: Optimization may converge to local minima; initialization matters
2. **Mesh quality**: Poor mesh quality (non-manifold, holes) may affect results
3. **Skeleton topology**: Optimization only adjusts radii, not node positions or connectivity
4. **Computational cost**: Large skeletons (>1000 nodes) may be slow to optimize

## Future Extensions

Potential improvements:
- Analytical gradients (faster than finite differences)
- Additional loss functions (Hausdorff distance, point-to-surface distance)
- Multi-objective optimization (Pareto front exploration)
- Adaptive optimization (different strategies for different skeleton regions)
- GPU acceleration for large-scale problems

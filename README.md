# mcf2swc

This is a Python package designed to take a 3D closed mesh surface (obj format, triangle-faced mesh) and the results of a mean curvature flow calculation (MCF) (polylines format) and produce an SWC model.

[Triangle mesh format](https://en.wikipedia.org/wiki/Triangle_mesh)

[SWC format](http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html)

Polylines text format is 
```N x1 y1 z1 x2 y2 z2 ... xN yN zN```
for each branch of the model.
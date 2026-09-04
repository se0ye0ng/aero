"""N3 - radiometrically consistent 3D multi-view generation.

Scaffolded, not implemented. See ``docs/roadmap.md``.

The premise of the whole study is that some imagery cannot be acquired: ranges, aspect
angles, atmospheric conditions and times of day that are out of reach. Restyling imagery that
already exists does not address that; rendering a scene that carries temperature, emissivity
and environmental irradiance at novel viewpoints does.

Physically consistent thermal 3D reconstruction has been published and is evaluated on
reconstruction quality. The open question is whether such views work as *training data* -
which is exactly what the rest of this repository is built to measure. The differentiation is
therefore the evaluation axis and the small-target regime, not the renderer.
"""

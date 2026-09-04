# N3 implementation plan

1. **Scene representation.** Gaussian primitives carrying temperature, emissivity and
   environmental irradiance in addition to geometry; rendering composes emission, reflection
   and atmospheric transmittance.
2. **Initialisation.** Thermal imagery has low texture and poor radiometric consistency
   across views, so structure-from-motion initialisation is unreliable. A learned initialiser
   over IR (optionally with paired RGB) is the practical route.
3. **Scenario rendering.** Sample the axes the real set cannot cover - range, aspect angle,
   time of day, atmospheric transmittance - and render targets at the pixel scales that
   matter (below 16 px^2).
4. **Evaluation.** Feed the rendered views through the same S3 to S5 path as any other
   generated data. The question is not reconstruction PSNR; it is dAP in the held-out
   scenario slice.

**Risk.** The renderer is the expensive part and is not the contribution. If it proves
disproportionate, the fallback is a physics-based scene simulator for the same coverage axes,
evaluated identically - the evaluation axis is what this track is really for.

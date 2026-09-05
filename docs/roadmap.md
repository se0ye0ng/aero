# Roadmap

## N1 — Radiometric Fidelity Score (core)
Diagnostic vector, distributional distances, and held-out predictive-power evaluation against
FID / LPIPS / SSIM and detection-specific SDQM / CCDM baselines. The central question is whether
sensor-aware features add value across unseen generators and domains. Delivers H1, H2, and the
curation policy for H4 if the evidence supports them.
Modules: `aero_ir.rfs`, `aero_ir.curate`.

## N2 — Sensor-in-the-loop generation
A differentiable IR sensor model placed between the generator and the detector, so generation
is optimised for the detector's decision statistics rather than for perceptual realism.
The sensor stack (NETD, MTF, AGC and quantisation, fixed-pattern noise) is already implemented
as `torch.nn.Module`s in `aero_ir.sensor` for exactly this reason.
Open questions: which sensor parameters are identifiable from a real set, and whether
back-propagating through AGC requires a relaxed surrogate.

## N3 — Radiometrically consistent 3D multi-view generation
The only route that addresses the original constraint — *imagery that cannot be acquired* —
rather than restyling imagery that already exists. A 3D scene representation carrying
temperature, emissivity and environmental irradiance is rendered at novel ranges, aspect
angles and atmospheric conditions.

Physically consistent thermal 3D reconstruction has been published; it is evaluated on
reconstruction quality. **The open question is whether such views are useful as training
data** — which is what this repository is built to measure. Differentiation is therefore
the evaluation axis and the small-target regime, not the renderer.
Module: `aero_ir.scene3d` (scaffolded, not implemented).

## Deployment track
ONNX export, INT8 quantisation, latency-vs-mAP curve on a fixed device profile. Small in
effort, and it is the difference between a research repository and one that reads as
deployable. Module: `aero_ir.deploy`.

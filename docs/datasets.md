# Datasets

Public data only. Each entry lists the access route and the licence that governs it; check
the licence before redistribution. No dataset is vendored into this repository.

| Dataset | Regime | Access |
|---|---|---|
| Teledyne FLIR ADAS Thermal v2 | urban / driving thermal and visible data, COCO labels; only explicitly verified pairs may enter generation | request form at the vendor site; a Kaggle mirror exists |
| LLVIP | low-light aligned visible-infrared pairs, pedestrians | project GitHub |
| M3FD | multi-scenario visible-infrared fusion and detection | public fusion-benchmark collections |
| KAIST Multispectral Pedestrian | day/night RGB-thermal pairs | project page |
| DroneVehicle | aerial RGB-IR vehicles, oriented boxes, large scale | project GitHub |
| Anti-UAV300 | paired RGB/IR UAV tracking data for generator development | project GitHub |
| Anti-UAV410 | IR-only UAV tracking benchmark for external thermal evaluation | project GitHub |
| CST Anti-UAV | tiny UAV thermal tracking | conference workshop release |

`scripts/download_*.sh` prints the access route and verifies the directory layout after manual
placement. Dataset-specific audit targets record archive checksums and content integrity.
Nothing is downloaded automatically from a gated source.

FLIR mirrors/releases in circulation do not all expose the same still-image count. Every run
must therefore record the actual COCO counts and source-archive hash; the dataset name alone is
not a sufficient provenance identifier.

## Coverage-split definition

For `data.coverage_split=held_out_scenario`, a scenario slice (a range band, an aspect-angle
band, or a time-of-day band, depending on dataset metadata) is removed from the real training
set entirely and retained in validation. Generated data is then the only training source for
that slice. This is valid only when the generator has an independent source for the withheld
scenario; it must not see validation imagery or annotations. The slice definition and source
provenance live in `configs/data/*.yaml` and are frozen before any run.

Anti-UAV does not currently satisfy that genuine-coverage design. E5 instead uses Anti-UAV300
paired training sequences for development and Anti-UAV410 as IR-only, sequence-disjoint external
evaluation. A tracking-to-frame-detection adapter must preserve video ids and visibility flags.

### Local Anti-UAV300 audit (2026-09-05)

`make audit-antiuav300` checks the archive CRC/SHA-256, official split manifests, every extracted
annotation, and paired video metadata. The audited release contains 160/67/91 mutually disjoint
train/validation/test sequences. Its legacy `test-dev` directory duplicates 100 training
sequences and must not be reported as an independent test set. Across both modalities, 445
present-frame annotations have zero-area boxes and require recorded exclusion. Visible and IR
videos also have different resolutions and substantial residual box-centre offsets, so direct
RGB-to-IR box transfer is prohibited until a calibrated transform is qualified. The corresponding
train-only RFS pilot uses display-referred 8-bit IR video and is a plumbing diagnostic, not a
radiometric or downstream-AP result.

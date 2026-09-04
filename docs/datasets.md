# Datasets

Public data only. Each entry lists the access route and the licence that governs it; check
the licence before redistribution. No dataset is vendored into this repository.

| Dataset | Regime | Access |
|---|---|---|
| Teledyne FLIR ADAS Thermal v2 | urban / driving, aligned RGB-thermal, COCO labels | request form at the vendor site; a Kaggle mirror exists |
| LLVIP | low-light aligned visible-infrared pairs, pedestrians | project GitHub |
| M3FD | multi-scenario visible-infrared fusion and detection | public fusion-benchmark collections |
| KAIST Multispectral Pedestrian | day/night RGB-thermal pairs | project page |
| DroneVehicle | aerial RGB-IR vehicles, oriented boxes, large scale | project GitHub |
| Anti-UAV410 | thermal-infrared UAV tracking benchmark | project GitHub |
| CST Anti-UAV | tiny UAV thermal tracking | conference workshop release |

`scripts/download_*.sh` prints the access route and verifies checksums after manual
placement. Nothing is downloaded automatically from a gated source.

## Coverage-split definition

For `data.coverage_split=held_out_scenario`, a scenario slice (a range band, an aspect-angle
band, or a time-of-day band, depending on dataset metadata) is removed from the real training
set entirely and retained in validation. Generated data is then the only source of coverage
for that slice. The slice definition per dataset lives in `configs/data/*.yaml` and is fixed
before any run.

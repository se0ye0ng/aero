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

### Local FLIR pair audit (2026-09-06)

`make manifest-flir-pairs` freezes all 3,749 entries in the provider's official
RGB-to-thermal map as a one-to-one, content-addressed manifest spanning eight `video_test`
sequence pairs. The manifest is valid only for post-freeze generator evaluation and
time-synchronisation/registration diagnostics. The release does not provide an official map for
the RGB and thermal train/validation still images, so COCO ids, filenames or frame numbers must
not be used to invent those pairs.

The conservative registration audit compares boxes only when a unique shared
`(track_id, category_id)` key exists in an officially paired frame. It finds 676 comparisons,
all in one sequence pair; the other seven sequence pairs have no shared keys. For the available
comparisons, normalised centre residual p95 is 0.0638 and median normalised box IoU is 0.5852.
This fails the frozen 0.02 centre-residual threshold and complete-sequence-coverage requirement.
Cross-modal track-id semantics are also undocumented. Consequently, direct label transfer,
training-time transform fitting and generator training on these test pairs remain prohibited.
The pair-manifest and audit hashes are `94d1a3b1...57acd69` and `6b6602eb...64bd8e9`.

## Coverage-split definition

For `data.coverage_split=held_out_scenario`, a scenario slice (a range band, an aspect-angle
band, or a time-of-day band, depending on dataset metadata) is removed from the real training
set entirely and retained in validation. Generated data is then the only training source for
that slice. This is valid only when the generator has an independent source for the withheld
scenario; it must not see validation imagery or annotations. The slice definition and source
provenance live in `configs/data/*.yaml` and are frozen before any run.

Anti-UAV does not currently satisfy that genuine-coverage design. E5 instead uses Anti-UAV300
paired training sequences for development and Anti-UAV410 as IR-only, sequence-disjoint external
evaluation. The tracking-to-frame-detection adapter preserves sequence ids and visibility flags.

### Local Anti-UAV300 audit (2026-09-05)

`make audit-antiuav300` checks the archive CRC/SHA-256, official split manifests, every extracted
annotation, and paired video metadata. The audited release contains 160/67/91 mutually disjoint
train/validation/test sequences. Its legacy `test-dev` directory duplicates 100 training
sequences and must not be reported as an independent test set. Across both modalities, 445
present-frame annotations have zero-area boxes and require recorded exclusion. Visible and IR
videos also have different resolutions and substantial residual box-centre offsets, so direct
RGB-to-IR box transfer is prohibited. A sequence-balanced robust transform fit only on 141,816
usable training pairs reaches a joint frame pass rate of 8.82% on train and 14.63% on the 57,982
held-out validation pairs, against the frozen 95% requirement. Validation median IoU is 0.293 and
median centroid shift is 0.312 target-box diagonals. Its content-addressed audit is
`d9575809...5868`; official test metrics are report-only. Target trajectories also provide no
dense background-registration evidence, so generator training remains prohibited. The
corresponding train-only RFS pilot uses display-referred 8-bit IR video and is a plumbing
diagnostic, not a radiometric or downstream-AP result.

The replacement dense-registration audit initializes from the public SuperFusion RoadScene
checkpoint, then fine-tunes for 300 epochs on a frozen sequence-balanced cache containing 16
pairs from each official training sequence. Validation/test are not opened during fitting. The
audit transforms visible boxes by sub-pixel inversion of the predicted target-to-source flow
and separately measures flow coverage and cross-modal edge-NCC gain. A sequence-balanced screen
is explicitly incapable of clearing the gate; only an exhaustive train/held-out-validation
report can do so. The official test split remains report-only. This qualification is pending,
so the generator-training hold is unchanged.

The registration-independent detector smoke uses only native IR frames. An endpoint-inclusive,
label-independent grid selects 8 frames per training sequence and 4 per validation sequence;
test is never accessed. The prepared manifest `c50c5df6...2f86fa` contains 1,279 training images
with 1,266 boxes and 13 negatives, plus 266 validation images with 257 boxes and 9 negatives.
One selected training positive and two selected validation positives have zero extent and are
excluded with explicit counts. Every decoded lossless PNG and both COCO files are hashed. This
qualifies a YOLOX engineering smoke only and supplies no evidence for cross-modal registration,
generator training, or Phase 5 scientific claims.

### Local Anti-UAV410 audit (2026-09-12)

`make audit-antiuav410` binds the extracted release to a CRC-clean 9,361,681,896-byte archive
with SHA-256 `339e0e56...e055`. The mutually disjoint train/validation/test splits contain
200/90/120 sequences and 213,995/94,711/129,691 frames. All 410 sampled sequence headers are
RGB-encoded 640x512 thermal JPEGs. Across the release, 641 present-frame boxes are invalid and
37 absent frames carry nonzero source boxes. The test-only adapter excludes its 64 invalid
positive frames, retains all 2,622 negatives according to the canonical existence flag, and
exports 129,627 images with 127,005 boxes. The manifest and COCO hashes are
`d5e391ff...0bf9d` and `55b3edfc...0ce3`. The frozen area strata contain 8 tiny
(`<16 px^2`), 64,082 small, 62,915 medium and zero large positive frames.

Anti-UAV410 is eligible only for external evaluation. It is IR-only, attribute arrays are absent
from 90 of 120 test sequences, and no frame may enter detector training, generator training or
curation. The incomplete attributes and source-box anomalies are recorded rather than repaired.

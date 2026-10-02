# Independent physical correspondence review

**Distribution paused:** do not begin either the48-pair form or the six-pair practice.
Free-choice landmarks do not support direct inter-reviewer coordinate agreement.
Human annotation is not a prerequisite to run automatic repair experiments; see
[the active automatic diagnostic path](registration_automatic_repair.md).
The historical panel and its artifacts are retained for provenance, not promoted as GT.

This workflow prepares **model-blind human annotation**, not automatic registration
qualification. No model matches, boxes, predicted points or warped images are shown.
The existing frozen v4 geometry criteria are unchanged. A valid review JSON does not
mean the registration is physically accurate or eligible for generator training.

## Prepare the train-only development panel

```bash
.venv/bin/python -B -m scripts.prepare_registration_review \
  --root /lustre/winston1214/dataset/Anti-UAV300 \
  --output-dir experiments/registration_physical_review_train48_02
```

This exports the same 16 sequence IDs as the original detector-free train16 screen,
with three frames at usable-index quantiles 25%, 50%, 75% (floor of `q*(N-1)`).
Only official train videos/annotations are read. Native RGB 1920x1080 and IR 640x512
PNG images are preserved. The manifest includes video, annotation, decoded-image,
PNG and preparation-source hashes. Existing output directories are never overwritten.
The panel is a development/observability resource, not untouched final test evidence.

## Two independent reviewers

Copy the entire panel directory to a local computer, preserving `images/` next to
the HTML files. Reviewer A opens `review_A.html`; reviewer B opens `review_B.html`.
Give each reviewer only their own HTML/template and the common images/manifest,
and do not exchange their outputs before both reviews are complete. Do not inspect
model predictions during annotation. The software records attestations; it cannot
verify human identity or actual independence.

1. Enter a real, nonempty reviewer identifier. A and B must be different people.
2. Inspect each pair. Use the display zoom and scrollbars for tiny UAV features.
3. Give a landmark a descriptive semantic ID, such as an identifiable rotor hub or
   building corner. Identify the **same physical feature**, not simply a similar blob.
4. Select target/background, click the feature in both images, and enter each
   localization uncertainty radius in native pixels. No uncertainty is prefilled.
5. If the feature is not observable in one image, select `unobservable`, provide
   a reason, and do not invent coordinates. Thermal hotspots and silhouette extrema
   are not automatically the same physical surface point in RGB and IR.
6. Add the landmark. The list preserves each reviewer's own observations. Distinct
   semantic IDs are not automatically matched or averaged across reviewers.
7. Record frame-level visibility, occlusion, overlay and synchronization concerns.
   A frame with no landmarks must have an explanation. Mark each frame reviewed.
8. Export JSON frequently; there is no autosave or upload. Resume your own downloaded
   JSON using the file picker. Pair selection, slot and panel hash must match.
9. Sign the independence/model-blind attestation only when accurate and export final
   `review_A.json` or `review_B.json`. Preserve these files without overwriting templates.

Coordinates use pixel centres: the top-left centre is `(0,0)`, last centre is
`(width-1,height-1)`. Display clicks are converted back to native coordinates.
Annotations on text, timestamps, crosshairs or box corners are not scene evidence.
If virtually no UAV landmarks are observable, report that limitation instead of
forcing annotations. Target-only landmarks do not validate full-image alignment.

## Validate the submitted files

```bash
.venv/bin/python -B -m scripts.validate_registration_review \
  --panel experiments/registration_physical_review_train48_02 \
  --review-a /path/to/review_A.json \
  --review-b /path/to/review_B.json \
  --output experiments/registration_physical_review_validation_01.json
```

Exit 0 means **annotation structure/provenance checks passed**, not registration PASS.
Exit 2 means incomplete/invalid submissions. The validator verifies every panel image
hash, reviewer identities/slots, attestations, pair coverage, coordinate bounds,
positive finite uncertainty and explicit unobservability. It preserves both reviewers'
points and disagreements. Same-name landmarks still require semantic adjudication;
different-name landmarks may identify the same point and also require human review.

The generated `physical_gate_draft.json` is deliberately `unapproved`, with no
invented tolerances. This validator accepts no checkpoint and cannot issue physical
PASS. Next steps are independent semantic adjudication, uncertainty/coverage assessment,
and a separately reviewed, hashed, frozen acceptance protocol **before** evaluating
candidate checkpoints. Existing geometry thresholds must not be weakened. A final
claim additionally requires a protocol-locked held-out panel because development
train/validation results have already influenced model choice.

The frame pairs use the dataset's equal frame indices. Equal FPS/frame counts alone
do not establish physical sensor synchronization. Suspected timing or parallax failures
must be investigated separately; reviewers must not silently repair pairings.

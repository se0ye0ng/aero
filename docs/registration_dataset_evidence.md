# Dataset evidence required for registration qualification

Audit date: 2026-09-24. Status: **HOLD**, not a replacement qualification protocol.
No training budget, threshold, split, checkpoint or data loader was changed.

## What the public source actually supports

The Anti-UAV benchmark paper explicitly describes its RGB/thermal video pairs as
unaligned (Fig. 1, Fig. 4 and §III-E). Its annotations describe target boxes,
existence and sequence attributes, not dense cross-modal pixel correspondence.
It also states that training and validation use non-overlapping clips from the
same source videos; the test set is separate. A sequence-disjoint validation
result must not be described as independent-recording generalization without an
additional recording-level audit. The paper is benchmark documentation, not proof
that every local release detail matches it.
[Primary paper](https://arxiv.org/abs/2101.08466).

## Local archive inspection

Read-only inspection of `Anti-UAV300.zip` and the official train list found:

| Inspected evidence | Result |
|---|---|
| Non-directory archive members | 1,676 |
| File extensions | 836 MP4, 839 JSON, 1 Python |
| JSON basenames | `train.json`, `val.json`, `test.json`: one each; `visible.json`, `infrared.json`: 318 each; `RGB_label.json`, `IR_label.json`: 100 each |
| Official train sequences | 160 |
| Train annotation JSONs read | 320, both modalities for every train sequence |
| Keys in every inspected train annotation | Exactly `exist`, `gt_rect` |
| Only non-video/non-JSON member | `framecut.py`, a frame extraction utility, not a registration transform |

Held-out annotation contents and video pixels were not inspected for this audit.
Archive member counts are packaging counts, not a claim of836 distinct scenes.
The extra label JSON contents were not audited; their existence does not
constitute verified calibration. The bounded MP4 inspection below also found no
usable calibration. External calibration may exist and should be assessed if
supplied. Box coordinate interpretation remains
the existing tested loader contract, not inferred from prose in a paper.

Inventory identities, SHA256:

- Sorted `(member_name, uncompressed_size, ZIP_CRC)` tuples encoded using
  `json.dumps(..., separators=(',', ':')).encode()`:
  `16b9dd95276f288ca257e5956b234de0e0a4cc4f50017ecfb4703176b15aab61`.
  This identifies directory metadata, **not a full archive-content SHA256**.
- Ordered `(member_name, SHA256(annotation_bytes))` tuples for sorted official
  train IDs, with `visible` then `infrared`, encoded the same way:
  `d4ff539f99b20509bf58185d4ffbdeecb81cecaefda5373fd777a96e986bec90`.

### Training-video container metadata follow-up

A read-only ISO-BMFF box-header inspection covered all320 official-train MP4
files, seeking over media payloads without decoding frames. Each has one video
track and one metadata handler. In every file the movie/media header creation
values (`mvhd`, `mdhd`) are zero. The only item-list metadata is encoder software
`©too = Lavf58.35.100`; no UUID/XMP boxes were found in the inspected hierarchy.
No usable camera calibration or common acquisition-time reference was identified.

Scope: traversal included `moov/trak/mdia/minf/stbl` and `moov/udta/meta/ilst`.
Codec sample descriptions, compressed payload/SEI messages, edit lists and
per-sample timing tables were not interpreted. This is not proof that all
possible embedded information is absent. Container creation times are not sensor
exposure times; zero values neither establish synchronization nor prove a timing
offset. No held-out video was opened and no timing correction was inferred.

## Two separate obstacles

1. The completed uniform model fails the frozen engineering screen:125/160
   midpoints,133/160 quarter frames, below95%. This is a measured model failure.
2. Even a future engineering pass would not establish physical correspondence.
   Boxes used to fit the model, its own cycle closure, and fitted pseudo-masks
   cannot become independent pixel ground truth. The current v4 gate deliberately
   keeps that decision on HOLD; a scientific authorization path remains absent.

The absence of supplied pixel ground truth is **not proof that automatic
registration is impossible**. It means the present artifacts cannot certify it.
More identical training cannot by itself create independent validation evidence.
Do not manually change gate strings, filter away failures or label a synthetic
warp recovery test as physical RGB/IR accuracy.

## Evidence acquisition decision, not a silent dataset switch

An automated qualification route needs independently justified reference geometry
on the intended domain, including coordinate conventions, coverage, uncertainty
and provenance. Examples to investigate are public camera calibration with
timing and depth/scene assumptions checked, or independently produced reference
correspondences not used for fitting. Calibration alone is insufficient when
parallax, changing zoom or timing errors violate its assumptions. No request for
new human reviewer annotations is imposed by this audit.

LLVIP is a possible **separate control**, not a solution to Anti-UAV qualification.
Its authors align pairs using selected points, a projective transform and cropping;
its pedestrian domain differs from small UAVs. Their alignment claim does not
certify our pixel-error limits. The final aligned release and raw unregistered
release must be distinguished.
[LLVIP paper, §3](https://arxiv.org/abs/2108.10831).
Its terms permit non-commercial use subject to attribution and other conditions;
there is no blanket authorization to redistribute all source or derivative data.
[Official terms](https://github.com/bupt-ai-cz/LLVIP/blob/main/Term%20of%20Use%20and%20License.md).

Adding such a control requires an explicit scope decision and a separate data/
residual-error/license audit before acquisition or training. Its success would
not close the Anti-UAV requirement. No new dataset was downloaded, author contacted,
held-out test opened, GPU job launched or generator approved in this audit.

Follow-up: the user authorized additional public-data validation on2026-09-24.
The [external landmark diagnostic](registration_external_landmarks.md) records
that new work separately; it does not revise the conclusions of this audit.

The user subsequently also authorized a qualified additional generator training
source, without replacing Anti-UAV results. The
[MS² source preflight](registration_ms2_source.md) records that separate acquisition
and coordinate/evidence audit. It is not an Anti-UAV pass or training approval.

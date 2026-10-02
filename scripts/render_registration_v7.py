"""Render stratified train diagnostics from a verified completed v7 arm.

The reference is the shared-velocity INITIALIZATION, not a separately trained
baseline. Dataset boxes are weak annotations, not pixel-correspondence GT.
No model fitting, checkpoint selection, test access or qualification approval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from aero_ir.registration.geometry import warp  # noqa: E402
from aero_ir.registration.protocol_v7 import SharedVelocityMatcher  # noqa: E402
from aero_ir.registration.qualification_v4 import direction_pass, direction_statistics  # noqa: E402
from aero_ir.registration.superfusion import load_superfusion_matcher  # noqa: E402
from aero_ir.utils.manifest import file_sha256  # noqa: E402
from scripts.train_antiuav300_registration_v7 import (  # noqa: E402
    INITIAL_SHA256,
    load_trained,
    read_batch,
    tensors,
)
from scripts.verify_registration_v7 import verify_arm  # noqa: E402


def selected_cases(initial, final, per_group=2):
    if type(per_group) is not int or per_group < 1:
        raise ValueError("per_group must be a positive integer")
    first = {r["sequence_id"]: r for r in initial}
    last = {r["sequence_id"]: r for r in final}
    if len(first) != len(initial) or len(last) != len(final) or first.keys() != last.keys():
        raise ValueError("screens must have identical unique sequence identities")
    groups = {k: [] for k in ("new_pass", "regression", "both_fail", "both_pass")}
    for sid in sorted(first):
        before, after = (
            all(direction_pass(r[d]) for d in ("ir_to_rgb_points", "rgb_to_ir_points"))
            for r in (first[sid], last[sid])
        )
        group = (
            ("both_pass" if before else "new_pass")
            if after
            else ("regression" if before else "both_fail")
        )
        groups[group].append(sid)
    return groups, [(group, sid) for group, ids in groups.items() for sid in ids[:per_group]]


def crop_limits(box, shape):
    height, width = shape[:2]
    cx, cy, bw, bh = np.asarray(box) * [width, height, width, height]
    radius = max(bw, bh, 16) * 1.5
    return (
        max(0, cx - radius),
        min(width, cx + radius),
        min(height, cy + radius),
        max(0, cy - radius),
    )


def render(path, images, vb, ib, title, *, titles=None):
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    if titles is None:
        titles = ("Visible input", "IR observation", "Initial RGB warp", "Trained RGB warp")
    for col, (image, label) in enumerate(zip(images, titles, strict=True)):
        box = vb if col == 0 else ib
        h, w = image.shape[:2]
        cx, cy, bw, bh = np.asarray(box) * [w, h, w, h]
        for row in range(2):
            ax = axes[row, col]
            # Box coordinates use image boundaries [0,W]x[0,H], not imshow's
            # default pixel-centre limits [-0.5,W-0.5]x[-0.5,H-0.5].
            ax.imshow(image, vmin=0, vmax=255, extent=(0, w, h, 0))
            ax.add_patch(
                Rectangle(
                    (cx - bw / 2, cy - bh / 2),
                    bw,
                    bh,
                    fill=False,
                    edgecolor="cyan" if col == 0 else "orange",
                    linewidth=1,
                )
            )
            ax.axis("off")
            if row:
                x0, x1, y1, y0 = crop_limits(box, image.shape)
                ax.set_xlim(x0, x1)
                ax.set_ylim(y1, y0)
            else:
                ax.set_title(label)
    fig.suptitle(title, fontsize=10)
    fig.text(
        0.5,
        0.01,
        "Boxes: dataset annotations, NOT correspondence GT. "
        "RGB zoom uses its own box; other zooms use the same IR coordinates.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("experiments/antiuav300_registration_v7_pilot_e10_seed0"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("experiments/antiuav300_registration_v2_full_train_cache"),
    )
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=Path(
            "experiments/antiuav300_registration_v6_pilot_e10_seed0/antiuav300_registration_v6_e10.pth"
        ),
    )
    parser.add_argument("--arm", choices=("geometry", "geometry_mind"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("refusing to overwrite visual diagnostics")
    root = args.run_root / args.arm
    proof = verify_arm(root, args.arm, args.cache_root)
    if proof["status"] != "verified_saved_artifacts_not_replayed":
        parser.error("this arm has not completed its verified 10-epoch pilot")
    if file_sha256(args.initial_checkpoint) != INITIAL_SHA256:
        parser.error("initializer differs from the frozen v7 pilot")
    source_hashes = {
        **proof["spec"]["source_sha256"],
        "scripts/render_registration_v7.py": file_sha256(__file__),
        "scripts/verify_registration_v7.py": file_sha256(
            Path(__file__).with_name("verify_registration_v7.py")
        ),
    }
    project = Path(__file__).resolve().parents[1]
    torch.set_num_threads(1)
    device = torch.device("cpu")
    initial = SharedVelocityMatcher(
        load_superfusion_matcher(args.initial_checkpoint, device)
    ).eval()
    final, _ = load_trained(root / "shared_velocity_e10.pth", device)
    screens = [
        json.loads((root / f"{phase}_train_screen.json").read_text())
        for phase in ("initial", "final")
    ]
    groups, selected = selected_cases(*(screen["rows"] for screen in screens))
    samples = {
        r["sequence_id"]: s for r, s in zip(screens[1]["rows"], screens[1]["samples"], strict=True)
    }
    args.out_dir.mkdir(parents=True, exist_ok=False)
    outputs = []
    with torch.no_grad():
        for index, (group, sid) in enumerate(selected):
            sample = samples[sid]
            arrays, digest = read_batch(args.cache_root, sample["selection"])
            if digest != sample["array_sha256"]:
                raise ValueError("visualized observations differ from the saved screen")
            vis, ir, vb, ib = tensors(arrays, device)
            images = [arrays["visible"][0], arrays["infrared"][0]]
            measurements = {}
            for name, model in (("initial", initial), ("final", final)):
                f, r = model.fields(ir, vis)
                value = warp(vis, f)[0].permute(1, 2, 0).clamp(0, 1).numpy()
                images.append(np.rint(value * 255).astype(np.uint8))
                measurements[name] = {
                    "ir_to_rgb_points": direction_statistics(f, r, ib, vb)[0],
                    "rgb_to_ir_points": direction_statistics(r, f, vb, ib)[0],
                }
            path = args.out_dir / f"{index:02d}_{group}.png"
            render(
                path,
                images,
                arrays["source_boxes"][0],
                arrays["target_boxes"][0],
                f"{args.arm} / {group} (saved-screen stratum) / {sid}\n"
                "Train observations; shared-velocity initialization vs completed pilot",
            )
            outputs.append(
                {
                    "sequence_id": sid,
                    "group": group,
                    "selection": sample["selection"],
                    "array_sha256": digest,
                    "path": path.name,
                    "sha256": file_sha256(path),
                    "cpu_recomputed_metrics": measurements,
                }
            )
    for name, digest in proof["artifacts_sha256"].items():
        if file_sha256(root / name) != digest:
            raise ValueError("pilot artifacts changed during rendering")
    if file_sha256(args.initial_checkpoint) != INITIAL_SHA256 or any(
        file_sha256(project / name) != digest for name, digest in source_hashes.items()
    ):
        raise ValueError("initializer or code changed during rendering")
    report = {
        "kind": "v7_train_visual_diagnostic_not_qualification",
        "arm": args.arm,
        "source_sha256": source_hashes,
        "pilot_artifacts_sha256": proof["artifacts_sha256"],
        "initial_checkpoint_sha256": INITIAL_SHA256,
        "selection_rule": "first two sorted sequence IDs per saved joint-pass stratum",
        "stratum_population": {k: len(v) for k, v in groups.items()},
        "outputs": outputs,
        "device": "cpu",
        "torch": str(torch.__version__),
        "limitations": [
            "Stratified diagnostic sample, not a representative performance estimate.",
            "The reference is shared-velocity initialization, not native v6 baseline.",
            "CPU measurements are recomputed; no bitwise GPU replay is asserted.",
            "Zero warp padding is not observed image content.",
            "Only IR-domain RGB warps are pictured; both direction metrics are recorded.",
        ],
        "generator_training_eligible": "hold_not_qualified",
    }
    with (args.out_dir / "report.json").open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(f"wrote {args.out_dir / 'report.json'}; train diagnostic, not qualification")


if __name__ == "__main__":
    main()

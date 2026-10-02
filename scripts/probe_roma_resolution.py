"""Fixed processing-resolution ablation on the unchanged Anti-UAV train16 inputs."""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

from aero_ir.utils.manifest import file_sha256
from scripts import probe_minima_roma_gpu as runner
from scripts import verify_minima_roma as verifier
from scripts.fetch_minima_roma import COMMIT, ROOT
from scripts.verify_antiuav_tiled_matching import safe_artifact

BASE_PROTOCOL = runner.protocol
SETTINGS = dict(coarse_res=840, upsample_res=1152)
BASE_REPORT = Path("experiments/registration_minima_roma_gpu_01/report.json")
BASE_SHA = "53f123f1e03ac563585cd477ae7b4748af875c830bed05ca9d8b8c0dbedc367d"


def protocol():
    hashes, cases, metadata = BASE_PROTOCOL()
    if file_sha256(BASE_REPORT) != BASE_SHA:
        raise ValueError("baseline changed")
    for path in (BASE_REPORT, Path(__file__), Path("scripts/run_roma_resolution_gpu.sh")):
        hashes[str(path)] = file_sha256(path)
    return hashes, cases, metadata


def resolution_factory(factory):
    def construct(*args, **kwargs):
        if any(k in kwargs for k in SETTINGS):
            raise ValueError("unexpected preexisting resolution override")
        return factory(*args, **kwargs, **SETTINGS)

    return construct


def ungated(path, report):
    rows = []
    for row in report["rows"]:
        with np.load(safe_artifact(path.parent, row), allow_pickle=False) as data:
            images = [np.empty((*shape, 3), dtype=np.uint8) for shape in row["crop_shapes"]]
            scores = runner.evaluate(
                data["warp"],
                np.ones_like(data["certainty"]),
                images,
                row["header_rows"],
                row["boxes_xyxy"],
            )
        rows.append(
            dict(
                sequence_id=row["sequence_id"],
                iou=scores["iou"],
                box_proxy_both_ge_06=scores["joint_pass"],
            )
        )
    return dict(rows=rows, box_proxy_both_ge_06=sum(r["box_proxy_both_ge_06"] for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "run", "verify"))
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    path = args.out_dir / "report.json"
    if args.action == "verify":
        with patch.object(verifier, "protocol", protocol):
            result = verifier.verify(path)
        report = json.loads(path.read_text())
        if report["resolution_settings"] != SETTINGS or report["ungated_diagnostic"] != ungated(
            path, report
        ):
            raise ValueError("resolution or diagnostic changed")
        print(json.dumps(result))
        return
    argv = [runner.__file__, "--out-dir", str(args.out_dir)]
    if args.action == "preflight":
        with (
            patch.object(runner, "protocol", protocol),
            patch.object(sys, "argv", argv + ["--preflight"]),
        ):
            runner.main()
        return
    sys.dont_write_bytecode = True
    sys.path.insert(0, str((ROOT / f"RoMa_minima-{COMMIT}").resolve()))
    protocol()  # Validate vendored code before importing it.
    import romatch

    with (
        patch.object(runner, "protocol", protocol),
        patch.object(sys, "argv", argv),
        patch.object(romatch, "roma_outdoor", resolution_factory(romatch.roma_outdoor)),
    ):
        runner.main()
    report = json.loads(path.read_text())
    report.update(
        resolution_settings=SETTINGS,
        ungated_diagnostic=ungated(path, report),
        limitations=[
            "Processing resolution only; no new native image information.",
            "Same header-cropped RGB inputs and unchanged weights; no GT-guided crop.",
            "Float32 inference, coarse840 and fine1152.",
            "Ungated branch is diagnostic, not a replacement acceptance gate.",
            "Train16 box proxy and controls do not establish physical pixel accuracy.",
        ],
    )
    path.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(
        json.dumps(
            dict(
                primary=report["joint_passes"],
                ungated=report["ungated_diagnostic"]["box_proxy_both_ge_06"],
            )
        )
    )


if __name__ == "__main__":
    main()

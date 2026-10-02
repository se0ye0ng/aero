"""Isolated-process adapter for the optional official DiffV2IR sampling backend.

Model/runtime GPU qualification is still pending; CPU interface tests are not
diffusion results. See docs/diffv2ir_integration.md before selecting weights.
"""

from __future__ import annotations

import copy
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from aero_ir.utils.manifest import file_sha256


class DiffV2IRGenerator:
    name = "diffv2ir"

    def __init__(
        self,
        checkpoint: str,
        guidance_scale: float = 7.5,
        steps: int = 50,
        use_segmentation: bool = True,
        caption_source: str = "blip",
        batch_size: int = 1,
        *,
        runtime_python: str | None = None,
        upstream_root: str = "experiments/external/DiffV2IR",
        checkpoint_sha256: str | None = None,
        clip_root: str | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.guidance_scale = guidance_scale
        self.steps = steps
        self.use_segmentation = use_segmentation
        self.caption_source = caption_source
        self.batch_size = batch_size
        self.runtime_python = runtime_python
        self.upstream_root = upstream_root
        self.checkpoint_sha256 = checkpoint_sha256
        self.clip_root = clip_root

    def generate(
        self,
        sources,
        labels,
        *,
        segmentations,
        prompts,
        source_ids,
        output_dir,
        caption_spec_sha256,
        segmentation_spec_sha256,
        seed=0,
    ):
        """Persist a request and execute actual sampling in the configured runtime.

        Conditions must be frozen beforehand; no implicit BLIP/SAM or resize is
        performed. Returned source labels remain candidates for F4 auditing, not
        a claim that diffusion preserved the labelled object.
        """
        if self.batch_size != 1 or not self.use_segmentation or self.caption_source != "blip":
            raise ValueError("this bridge requires batch_size=1, segmentation, frozen BLIP prompts")
        if type(self.steps) is not int or self.steps <= 0:
            raise ValueError("steps must be a positive integer")
        if not math.isfinite(self.guidance_scale) or self.guidance_scale < 0:
            raise ValueError("guidance_scale must be finite and nonnegative")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        for digest in (self.checkpoint_sha256, caption_spec_sha256, segmentation_spec_sha256):
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError(
                    "explicit lowercase checkpoint/conditioning SHA256 values required"
                )
        if not self.runtime_python or not self.clip_root:
            raise ValueError("configure a separate runtime_python and local clip_root first")
        python = Path(self.runtime_python).absolute()
        if not python.is_file():
            raise FileNotFoundError(python)
        checkpoint = Path(self.checkpoint).resolve()
        if file_sha256(checkpoint) != self.checkpoint_sha256:
            raise ValueError("checkpoint SHA256 mismatch")
        sources, labels, segmentations = list(sources), list(labels), list(segmentations)
        prompts, source_ids = list(prompts), list(source_ids)
        if (
            not sources
            or len({len(v) for v in (sources, labels, segmentations, prompts, source_ids)}) != 1
        ):
            raise ValueError("sources, labels and conditioning lengths must match and be nonempty")
        if any(not isinstance(s, str) or not s for s in source_ids) or len(set(source_ids)) != len(
            source_ids
        ):
            raise ValueError("source_ids must be nonempty unique strings")
        for rgb, seg, prompt in zip(sources, segmentations, prompts, strict=True):
            if (
                not isinstance(rgb, np.ndarray)
                or not isinstance(seg, np.ndarray)
                or rgb.dtype != np.uint8
                or seg.dtype != np.uint8
                or rgb.ndim != 3
                or rgb.shape[2] != 3
                or rgb.shape != seg.shape
                or min(rgb.shape[:2]) <= 0
            ):
                raise ValueError("RGB and segmentation must be matching uint8 H,W,3 arrays")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("each source needs a frozen nonempty prompt")
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=False)
        (output / "conditioning").mkdir()
        records = []
        for index, (rgb, seg, prompt, sid) in enumerate(
            zip(sources, segmentations, prompts, source_ids, strict=True)
        ):
            record = {"source_id": sid, "prompt": prompt}
            for role, image in (("rgb", rgb), ("segmentation", seg)):
                path = output / "conditioning" / f"{index:06d}_{role}.png"
                with path.open("xb") as handle:
                    Image.fromarray(image).save(handle, format="PNG")
                record[role] = {"path": str(path.relative_to(output)), "sha256": file_sha256(path)}
            records.append(record)
        project = Path(__file__).resolve().parents[3]
        request = {
            "schema_version": 1,
            "kind": "aero_diffv2ir_inference_request",
            "backend": {
                "upstream_root": str(Path(self.upstream_root).resolve()),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": self.checkpoint_sha256,
                "clip_root": str(Path(self.clip_root).resolve()),
            },
            "seed": seed,
            "records": records,
            "sampling": {
                "steps": self.steps,
                "text_scale": self.guidance_scale,
                "image_scale": 1.5,
                "seg_scale": 1.5,
            },
            "caption_spec_sha256": caption_spec_sha256,
            "segmentation_spec_sha256": segmentation_spec_sha256,
            "prompt_source_declared_by_caller": self.caption_source,
            "bridge_source_sha256": {
                str(p.relative_to(project)): file_sha256(p)
                for p in (
                    Path(__file__),
                    project / "scripts/sample_diffv2ir.py",
                    project / "src/aero_ir/generate/diffv2ir_sampling.py",
                )
            },
        }
        request_path = output / "request.json"
        with request_path.open("x") as handle:
            json.dump(request, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        request_hash = file_sha256(request_path)
        env = dict(
            os.environ,
            PYTHONPATH=str(project / "src"),
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
        )
        env.pop("LD_LIBRARY_PATH", None)
        command = [
            str(python),
            "-u",
            str(project / "scripts/sample_diffv2ir.py"),
            "--request",
            str(request_path),
            "--request-sha256",
            request_hash,
        ]
        with (output / "worker.log").open("x") as log:
            completed = subprocess.run(
                command, cwd=project, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        if completed.returncode:
            raise RuntimeError(
                f"DiffV2IR worker failed ({completed.returncode}); see {output / 'worker.log'}"
            )
        result_path = output / "result.json"
        result_hash = file_sha256(result_path)
        result = json.loads(result_path.read_text())
        if (
            result.get("kind") != "aero_diffv2ir_inference_outputs"
            or result.get("generator_training_eligible") != "hold_not_qualified"
            or result.get("output_radiometry") != "display_rgb_not_calibrated_radiance"
        ):
            raise ValueError("unsupported sampling result or unjustified qualification claim")
        if (
            result.get("request_sha256") != request_hash
            or file_sha256(request_path) != request_hash
        ):
            raise ValueError("sampling result is not bound to the original request")
        outputs = result["outputs"]
        if [r["source_id"] for r in outputs] != source_ids:
            raise ValueError("sampling output identities/order differ from request")
        images, provenance = [], []
        for rgb, row in zip(sources, outputs, strict=True):
            if row.get("label_validity") != "unverified_requires_label_audit":
                raise ValueError("sampling cannot establish source label validity")
            path = (output / row["path"]).resolve()
            if not path.is_relative_to(output) or file_sha256(path) != row["sha256"]:
                raise ValueError("generated image path/hash mismatch")
            with Image.open(path) as image:
                if image.mode != "RGB":
                    raise ValueError("generated image must be explicitly RGB")
                arr = np.asarray(image).copy()
            if file_sha256(path) != row["sha256"]:
                raise ValueError("generated image changed while decoding")
            if arr.dtype != np.uint8 or arr.shape != rgb.shape:
                raise ValueError("generated image changed the source observation grid")
            images.append(arr)
            provenance.append(
                {
                    **row,
                    "generator": self.name,
                    "request_sha256": request_hash,
                    "result_sha256": result_hash,
                    "checkpoint_sha256": self.checkpoint_sha256,
                    "generator_training_eligible": "hold_not_qualified",
                }
            )
        if file_sha256(result_path) != result_hash:
            raise ValueError("sampling result changed while reading outputs")
        return images, copy.deepcopy(labels), provenance

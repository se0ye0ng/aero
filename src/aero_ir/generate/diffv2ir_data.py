"""Explicit train-only conditioning batches for the upstream DiffV2IR model.

This is a data interface, NOT a registration qualifier or a diffusion trainer.
Inputs must already share an IR-domain observation grid. Merely loading a batch
does not establish that those inputs are physically registered.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _read_verified(path: Path, expected: str) -> bytes:
    """Hash the exact bytes subsequently decoded, not a previous file open."""
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"SHA256 mismatch: {path}")
    return data


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


class DiffV2IRTrainDataset(Dataset):
    """Yield the upstream ``edited`` / ``edit`` BCHW conditioning contract.

    A JSON manifest contains explicit records; no random frame-level split, online
    caption generation, auto-download, registration fitting, or test access occurs.
    The expected manifest and official-train digests are supplied separately by the
    caller, so editing a manifest cannot silently change the selected experiment.

    All image assets are hashed at each access. ``rgb`` and ``seg`` are exported on
    the same lattice as ``ir``. A fully observed ``support`` mask is mandatory:
    upstream's unmasked diffusion objective must not learn registration padding.
    A common-view crop can satisfy this, but its exporter must qualify that exact
    supervision domain and record the crop/box transform before training is allowed.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        train_split_path: str | Path,
        expected_manifest_sha256: str,
        expected_train_split_sha256: str,
        output_size: tuple[int, int] = (256, 256),
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.root = self.manifest_path.parent
        self.manifest_sha256 = _digest(expected_manifest_sha256, "manifest hash")
        train_hash = _digest(expected_train_split_sha256, "official train hash")
        if len(output_size) != 2:
            raise ValueError("output_size must be (height, width)")
        self.output_size = tuple(_integer(v, "output_size", 64) for v in output_size)
        if any(v % 64 for v in self.output_size):
            raise ValueError("output_size dimensions must be multiples of 64")

        manifest = json.loads(_read_verified(self.manifest_path, self.manifest_sha256))
        train = json.loads(_read_verified(Path(train_split_path), train_hash))
        if not isinstance(train, dict) or not train:
            raise ValueError(
                "official train split must be a nonempty sequence-to-attributes object"
            )
        if not isinstance(manifest, dict):
            raise ValueError("conditioning manifest must be an object")
        required = {
            "schema_version": 1,
            "kind": "aero_diffv2ir_train_conditioning",
            "split": "train",
            "coordinate_domain": "common_ir_observation_grid",
            "official_train_sha256": train_hash,
        }
        for key, expected in required.items():
            if manifest.get(key) != expected:
                raise ValueError(f"invalid conditioning manifest {key}; expected {expected!r}")
        provenance = manifest.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError("conditioning manifest needs export provenance")
        # These identities are necessary lineage, not sufficient qualification.
        for key in (
            "pair_manifest_sha256",
            "registration_checkpoint_sha256",
            "registration_report_sha256",
            "export_spec_sha256",
            "caption_spec_sha256",
            "segmentation_spec_sha256",
        ):
            _digest(provenance.get(key), key)
        self.export_provenance = dict(provenance)

        rows = manifest.get("records")
        if not isinstance(rows, list) or not rows:
            raise ValueError("conditioning manifest has no records")
        seen_ids, seen_pairs = set(), set()
        self.records = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("conditioning record must be an object")
            pair_id, sequence = row.get("pair_id"), row.get("sequence")
            if not isinstance(pair_id, str) or not pair_id.strip() or pair_id in seen_ids:
                raise ValueError("pair_id must be nonempty and unique")
            if not isinstance(sequence, str) or sequence not in train:
                raise ValueError(
                    f"sequence is not in the pinned official train split: {sequence!r}"
                )
            visible = _integer(row.get("visible_frame_index"), "visible_frame_index")
            infrared = _integer(row.get("infrared_frame_index"), "infrared_frame_index")
            pair = (sequence, visible, infrared)
            if pair in seen_pairs:
                raise ValueError(f"duplicate source frame pair: {pair}")
            seen_ids.add(pair_id)
            seen_pairs.add(pair)
            _integer(row.get("height"), "height", 1)
            _integer(row.get("width"), "width", 1)
            if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
                raise ValueError("a frozen nonempty vision-language prompt is required")
            for role in ("rgb", "ir", "seg", "support"):
                self._asset_path(row.get(role), role)
            self.records.append(row)

    def _asset_path(self, asset: object, role: str) -> Path:
        if not isinstance(asset, dict):
            raise ValueError(f"missing {role} asset")
        _digest(asset.get("sha256"), f"{role} hash")
        name = asset.get("path")
        if not isinstance(name, str) or not name or "\n" in name or "\r" in name:
            raise ValueError(f"invalid {role} path")
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{role} path must remain inside the export directory")
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError(f"{role} path must name a file inside the export directory: {name}")
        return path

    def _image(self, row: dict, role: str) -> Image.Image:
        asset = row[role]
        data = _read_verified(self._asset_path(asset, role), asset["sha256"])
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            if image.size != (row["width"], row["height"]):
                raise ValueError(f"{row['pair_id']} {role}: observation grid size mismatch")
            modes = ("L",) if role == "support" else ("L", "RGB") if role == "ir" else ("RGB",)
            if image.mode not in modes:
                raise ValueError(f"{role} must have mode {modes}, got {image.mode}")
            if role == "support":
                if not np.all(np.asarray(image) == 255):
                    raise ValueError(
                        "unobserved registration pixels: "
                        "upstream unmasked loss cannot use this pair"
                    )
                return image.copy()
            # Native 8-bit grayscale IR becomes three equal channels. No per-image
            # contrast fitting, uint16 truncation, or silently applied colour map.
            return image.convert("RGB")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        row = self.records[index]
        self._image(row, "support")
        tensors = {}
        height, width = self.output_size
        for role in ("rgb", "ir", "seg"):
            image = self._image(row, role)
            interpolation = Image.Resampling.NEAREST if role == "seg" else Image.Resampling.BILINEAR
            image = image.resize((width, height), resample=interpolation)
            array = np.asarray(image, dtype=np.float32).copy()
            tensors[role] = torch.from_numpy(array).permute(2, 0, 1).contiguous() / 127.5 - 1.0
        return {
            "edited": tensors["ir"],
            "edit": {
                "c_concat1": tensors["rgb"],
                "c_concat2": tensors["seg"],
                "c_crossattn": row["prompt"],
            },
        }

    def sample_provenance(self, index: int) -> dict:
        """Separate metadata avoids changing the upstream model's input contract."""
        row = self.records[index]
        height, width = self.output_size
        return {
            "manifest_sha256": self.manifest_sha256,
            "pair_id": row["pair_id"],
            "sequence": row["sequence"],
            "visible_frame_index": row["visible_frame_index"],
            "infrared_frame_index": row["infrared_frame_index"],
            "input_size_hw": [row["height"], row["width"]],
            "output_size_hw": [height, width],
            "transform": "full_grid_resize_no_crop_no_flip",
            "xy_scale": [width / row["width"], height / row["height"]],
            "rgb_ir_interpolation": "bilinear",
            "seg_interpolation": "nearest",
            "registration_qualification": "not_established_by_data_loader",
        }

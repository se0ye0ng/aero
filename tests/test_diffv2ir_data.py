"""Synthetic interface tests, not evidence of RGB/IR registration accuracy."""

import copy
import hashlib
import json

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from aero_ir.generate.diffv2ir_data import DiffV2IRTrainDataset  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def bundle(tmp_path):
    train_path = tmp_path / "train.json"
    train_path.write_text(json.dumps({"train_sequence": ["day"]}))
    grid = np.zeros((16, 32, 3), dtype=np.uint8)
    grid[:, 16:] = 255
    assets = {}
    for role in ("rgb", "ir", "seg", "support"):
        path = tmp_path / f"{role}.png"
        data = np.full((16, 32), 255, np.uint8) if role == "support" else grid
        if role == "ir":
            data = grid[..., 0]
        Image.fromarray(data).save(path)
        assets[role] = {"path": path.name, "sha256": sha(path)}
    manifest = {
        "schema_version": 1,
        "kind": "aero_diffv2ir_train_conditioning",
        "split": "train",
        "coordinate_domain": "common_ir_observation_grid",
        "official_train_sha256": sha(train_path),
        "provenance": dict.fromkeys(
            (
                "pair_manifest_sha256",
                "registration_checkpoint_sha256",
                "registration_report_sha256",
                "export_spec_sha256",
                "caption_spec_sha256",
                "segmentation_spec_sha256",
            ),
            "a" * 64,
        ),
        "records": [
            {
                "pair_id": "train_sequence_v0_i1",
                "sequence": "train_sequence",
                "visible_frame_index": 0,
                "infrared_frame_index": 1,
                "height": 16,
                "width": 32,
                "prompt": "turn the visible image of a UAV into infrared",
                **assets,
            }
        ],
    }
    return tmp_path, train_path, manifest


def dataset(bundle, **kwargs):
    root, train, manifest = bundle
    path = root / "conditioning.json"
    path.write_text(json.dumps(manifest))
    return DiffV2IRTrainDataset(path, train, sha(path), sha(train), **kwargs)


def test_upstream_batch_contract_preserves_rgb_ir_grid_and_mask_palette(bundle):
    data = dataset(bundle, output_size=(64, 128))
    item = data[0]
    assert set(item) == {"edited", "edit"}
    assert set(item["edit"]) == {"c_concat1", "c_concat2", "c_crossattn"}
    for tensor in (item["edited"], item["edit"]["c_concat1"], item["edit"]["c_concat2"]):
        assert tensor.shape == (3, 64, 128)
        assert tensor.dtype == torch.float32
        assert tensor.is_contiguous()
        assert tensor.min() == -1 and tensor.max() == 1
    assert torch.equal(item["edited"], item["edit"]["c_concat1"])
    assert set(item["edit"]["c_concat2"].unique().tolist()) == {-1.0, 1.0}
    assert torch.equal(data[0]["edited"], data[0]["edited"])
    batch = next(iter(torch.utils.data.DataLoader(data, batch_size=1)))
    assert batch["edited"].shape == (1, 3, 64, 128)
    assert batch["edit"]["c_crossattn"] == [bundle[2]["records"][0]["prompt"]]
    assert len(data) == 1  # No implicit 90/5/5 split or silently dropped sample.
    provenance = data.sample_provenance(0)
    assert provenance["xy_scale"] == [4, 4]
    assert provenance["visible_frame_index"] == 0
    assert provenance["infrared_frame_index"] == 1
    assert provenance["registration_qualification"] == "not_established_by_data_loader"


@pytest.mark.parametrize(
    "key,value",
    [
        ("split", "val"),
        ("split", "test"),
        ("split", None),
        ("coordinate_domain", "native_visible"),
        ("kind", "antiuav300_raw_cache"),
        ("official_train_sha256", "b" * 64),
        ("schema_version", 2),
    ],
)
def test_rejects_wrong_protocol(bundle, key, value):
    bundle[2][key] = value
    with pytest.raises(ValueError, match="manifest"):
        dataset(bundle)


@pytest.mark.parametrize(
    "key,value,error",
    [
        ("sequence", "heldout_sequence", "official train"),
        ("prompt", "", "prompt"),
        ("visible_frame_index", -1, "integer"),
        ("infrared_frame_index", True, "integer"),
        ("width", 0, "integer"),
    ],
)
def test_rejects_bad_row(bundle, key, value, error):
    bundle[2]["records"][0][key] = value
    with pytest.raises(ValueError, match=error):
        dataset(bundle)


def test_duplicate_pair_cannot_be_hidden_under_another_id(bundle):
    row = copy.deepcopy(bundle[2]["records"][0])
    row["pair_id"] = "another_id"
    bundle[2]["records"].append(row)
    with pytest.raises(ValueError, match="duplicate source frame"):
        dataset(bundle)


@pytest.mark.parametrize("size", [(63, 64), (65, 64), (True, 64), (64,)])
def test_invalid_shape(bundle, size):
    with pytest.raises(ValueError, match="output_size"):
        dataset(bundle, output_size=size)


def test_changed_asset_rejected_after_initialization(bundle):
    data = dataset(bundle)
    (bundle[0] / "rgb.png").write_bytes(b"changed since manifest")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        data[0]


def test_pinned_manifest_and_official_split_are_verified(bundle):
    data = dataset(bundle)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        DiffV2IRTrainDataset(data.manifest_path, bundle[1], "b" * 64, sha(bundle[1]))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        DiffV2IRTrainDataset(data.manifest_path, bundle[1], sha(data.manifest_path), "b" * 64)


@pytest.mark.parametrize("role", ["rgb", "ir", "seg", "support"])
def test_mismatched_observation_grids(bundle, role):
    path = bundle[0] / f"{role}.png"
    Image.new("L" if role in ("ir", "support") else "RGB", (31, 16)).save(path)
    bundle[2]["records"][0][role]["sha256"] = sha(path)
    with pytest.raises(ValueError, match="grid size mismatch"):
        dataset(bundle)[0]


def test_registration_padding_is_not_a_diffusion_target(bundle):
    path = bundle[0] / "support.png"
    mask = np.full((16, 32), 255, np.uint8)
    mask[0, 0] = 0
    Image.fromarray(mask).save(path)
    bundle[2]["records"][0]["support"]["sha256"] = sha(path)
    with pytest.raises(ValueError, match="unobserved registration pixels"):
        dataset(bundle)[0]


def test_uint16_is_not_silently_truncated(bundle):
    path = bundle[0] / "ir.png"
    Image.fromarray(np.full((16, 32), 32768, np.uint16)).save(path)
    bundle[2]["records"][0]["ir"]["sha256"] = sha(path)
    with pytest.raises(ValueError, match="mode"):
        dataset(bundle)[0]


@pytest.mark.parametrize("path", ["../rgb.png", "/tmp/rgb.png", "rgb\n.png"])
def test_assets_cannot_escape_the_export(bundle, path):
    bundle[2]["records"][0]["rgb"]["path"] = path
    with pytest.raises(ValueError, match="path"):
        dataset(bundle)


def test_missing_provenance_rejected(bundle):
    del bundle[2]["provenance"]["caption_spec_sha256"]
    with pytest.raises(ValueError, match="caption_spec"):
        dataset(bundle)


def test_symlink_retarget_cannot_escape_export(bundle, tmp_path_factory):
    root = bundle[0]
    link = root / "rgb_link.png"
    link.symlink_to(root / "rgb.png")
    bundle[2]["records"][0]["rgb"]["path"] = link.name
    data = dataset(bundle)
    outside = tmp_path_factory.mktemp("external_assets") / "rgb.png"
    outside.write_bytes((root / "rgb.png").read_bytes())
    link.unlink()
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="inside the export"):
        data[0]

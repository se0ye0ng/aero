import copy
import shutil
import subprocess

import pytest

from aero_ir.registration.physical_review import review_template, validate_reviews


def example():
    manifest = {
        "split": "train",
        "pairs": [
            {
                "pair_id": "seq__000001",
                "images": {
                    "visible": {"shape": [1080, 1920, 3]},
                    "infrared": {"shape": [512, 640, 3]},
                },
            }
        ],
    }
    reviews = [review_template(manifest, "hash", slot) for slot in ("A", "B")]
    for i, review in enumerate(reviews):
        review["reviewer_identity"] = f"independent person {i}"
        review["independent_model_blind_attestation"] = True
        review["pairs"][0]["reviewed"] = True
        review["pairs"][0]["landmarks"] = [
            {
                "landmark_id": "physical rotor hub",
                "region": "target",
                "visible": {
                    "visibility": "visible",
                    "xy": [100.0 + i, 200.0],
                    "uncertainty_px": 1.0,
                },
                "infrared": {"visibility": "visible", "xy": [30.0, 40.0], "uncertainty_px": 0.5},
            }
        ]
    return manifest, reviews


def test_valid_structure_keeps_qualification_hold_and_disagreements():
    manifest, reviews = example()
    result = validate_reviews(manifest, "hash", reviews)
    assert result["structure_valid"]
    assert result["generator_training_eligible"] == "hold"
    assert result["reviewer_discrepancies"][0]["native_pixel_disagreement"]["visible"] == 1
    assert result["reviewer_discrepancies"][0]["requires_semantic_adjudication"]


def test_empty_templates_fail_closed():
    manifest, _ = example()
    reviews = [review_template(manifest, "hash", slot) for slot in ("A", "B")]
    assert not validate_reviews(manifest, "hash", reviews)["structure_valid"]


@pytest.mark.parametrize(
    "change", ["identity", "attest", "hash", "coverage", "duplicate", "reviewed"]
)
def test_review_provenance_failures(change):
    manifest, reviews = example()
    if change == "identity":
        reviews[1]["reviewer_identity"] = reviews[0]["reviewer_identity"].upper()
    elif change == "attest":
        reviews[1]["independent_model_blind_attestation"] = False
    elif change == "hash":
        reviews[1]["manifest_sha256"] = "different"
    elif change == "coverage":
        reviews[1]["pairs"] = []
    elif change == "duplicate":
        reviews[1]["pairs"].append(copy.deepcopy(reviews[1]["pairs"][0]))
    else:
        reviews[1]["pairs"][0]["reviewed"] = False
    assert not validate_reviews(manifest, "hash", reviews)["structure_valid"]


@pytest.mark.parametrize(
    "xy,radius",
    [
        ([1920, 10], 1),
        ([20, float("nan")], 1),
        ([True, 20], 1),
        ([20, 20], 0),
        ([20, 20], float("inf")),
    ],
)
def test_invalid_native_points(xy, radius):
    manifest, reviews = example()
    reviews[0]["pairs"][0]["landmarks"][0]["visible"].update(xy=xy, uncertainty_px=radius)
    assert not validate_reviews(manifest, "hash", reviews)["structure_valid"]


def test_unobservable_disagreement_is_preserved_not_dropped():
    manifest, reviews = example()
    reviews[1]["pairs"][0]["landmarks"][0]["infrared"] = {
        "visibility": "unobservable",
        "xy": None,
        "uncertainty_px": None,
        "reason": "thermal blooming prevents localization",
    }
    result = validate_reviews(manifest, "hash", reviews)
    assert result["structure_valid"]
    row = result["reviewer_discrepancies"][0]
    assert row["native_pixel_disagreement"]["infrared"] is None
    assert row["review_B"]["infrared"]["reason"]


def test_completely_unobservable_frame_requires_explanation():
    manifest, reviews = example()
    for review in reviews:
        review["pairs"][0].update(landmarks=[], notes="No common physical feature identifiable")
    assert validate_reviews(manifest, "hash", reviews)["structure_valid"]
    reviews[0]["pairs"][0]["notes"] = ""
    assert not validate_reviews(manifest, "hash", reviews)["structure_valid"]


def test_malformed_review_and_slot_fail_closed():
    manifest, reviews = example()
    assert not validate_reviews(manifest, "hash", list(reversed(reviews)))["structure_valid"]
    assert not validate_reviews(manifest, "hash", [None, reviews[1]])["structure_valid"]
    reviews[0]["reviewer_slot"] = []
    assert not validate_reviews(manifest, "hash", reviews)["structure_valid"]


def test_embedded_javascript_syntax():
    from scripts.prepare_registration_review import HTML

    if not shutil.which("node"):
        pytest.skip("Node unavailable; HTML runtime not installed by annotation workflow")
    script = HTML.split("<script>", 1)[1].split("</script>", 1)[0]
    script = script.replace("__MANIFEST__", "{}").replace("__TEMPLATE__", "{}")
    subprocess.run(["node", "--check"], input=script, text=True, check=True, capture_output=True)

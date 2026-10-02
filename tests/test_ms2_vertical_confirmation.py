"""Guard the confirmation protocol against retuning or relabelling approval."""

import pytest

from scripts.probe_ms2_vertical_confirmation import SOURCE_SHA, validate_protocol


def protocol():
    return dict(
        frame_ids=["000174", "000870"],
        sequence="sequence",
        source_sha256=SOURCE_SHA,
        offsets=[0.0, 0.5, -0.5],
        wrong_pair_cyclic_offset=7,
        thermal_window_dn=[3308.0, 4974.0],
        camera_fit=False,
        registration_qualified=False,
        generator_training_approved=False,
    )


def test_fixed_disjoint_protocol():
    p = protocol()
    validate_protocol(p, p.copy(), ["000000", "000696"])


@pytest.mark.parametrize(
    "key,value",
    [
        ("offsets", [0.0, 1.0, -1.0]),
        ("wrong_pair_cyclic_offset", 1),
        ("thermal_window_dn", [0.0, 65535.0]),
        ("camera_fit", True),
        ("registration_qualified", True),
        ("generator_training_approved", True),
        ("source_sha256", "changed"),
        ("sequence", "another"),
        ("frame_ids", ["000174"]),
    ],
)
def test_protocol_rejects_changes(key, value):
    p = protocol()
    plan = p.copy()
    p[key] = value
    with pytest.raises(ValueError):
        validate_protocol(p, plan, ["000000"])


def test_protocol_rejects_selection_overlap():
    p = protocol()
    with pytest.raises(ValueError):
        validate_protocol(p, p.copy(), ["000870"])

"""Rendering/selection checks only; fixture images are not registration evidence."""

import copy

import numpy as np
import pytest
import torch
from PIL import Image

from aero_ir.registration.qualification_v4 import direction_statistics
from scripts.render_registration_v7 import crop_limits, render, selected_cases


def row(sid, passing):
    field = torch.zeros(1, 2, 32, 32)
    box = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    measurement = direction_statistics(field, field, box, box)[0]
    if not passing:
        measurement["bbox_iou"] = 0.1
    return {
        "sequence_id": sid,
        "ir_to_rgb_points": measurement,
        "rgb_to_ir_points": copy.deepcopy(measurement),
    }


def test_diagnostic_selection_includes_regressions_and_persistent_failures():
    before = [row("b", True), row("a", False), row("d", True), row("c", False)]
    after = [row("a", True), row("b", False), row("c", False), row("d", True)]
    groups, selected = selected_cases(before, after)
    assert selected == [
        ("new_pass", "a"),
        ("regression", "b"),
        ("both_fail", "c"),
        ("both_pass", "d"),
    ]
    assert all(len(v) == 1 for v in groups.values())
    assert selected_cases(before[::-1], after[::-1]) == (groups, selected)


def test_no_new_pass_cases_are_not_fabricated():
    rows = [row(str(n), False) for n in range(4)]
    groups, selected = selected_cases(rows, rows)
    assert groups["new_pass"] == groups["regression"] == groups["both_pass"] == []
    assert selected == [("both_fail", "0"), ("both_fail", "1")]
    with pytest.raises(ValueError):
        selected_cases(rows, rows[:2])
    with pytest.raises(ValueError):
        selected_cases(rows + [rows[0]], rows)


def test_render_uses_image_boundary_coordinates(tmp_path, monkeypatch):
    from matplotlib.axes import Axes

    original = Axes.imshow
    extents = []

    def inspect(self, image, *args, **kwargs):
        extents.append(kwargs.get("extent"))
        return original(self, image, *args, **kwargs)

    monkeypatch.setattr(Axes, "imshow", inspect)
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    box = [0.5, 0.5, 0.2, 0.2]
    path = tmp_path / "fixture.png"
    render(path, [image] * 4, box, box, "test fixture, not model output")
    assert extents == [(0, 48, 32, 0)] * 8
    with Image.open(path) as output:
        assert output.width > 48 and output.height > 32
    x0, x1, y1, y0 = crop_limits([0.05, 0.05, 0.1, 0.1], image.shape)
    assert 0 <= x0 < x1 <= 48 and 0 <= y0 < y1 <= 32

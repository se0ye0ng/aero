import numpy as np

from scripts.probe_registration_timing import lag_screen, trajectory


def test_known_nonlinear_lag():
    t = np.linspace(0, 12, 400)
    a = np.c_[np.sin(t) + 0.2 * np.sin(7 * t), np.cos(1.7 * t) + 0.1 * np.sin(11 * t)]
    b = np.roll(a @ np.array([[0.7, 0.1], [-0.2, 0.8]]) + [0.1, 0.2], 4, axis=0)
    valid = np.ones(len(t), bool)
    result = lag_screen(a, b, valid, valid)
    assert all(x["selected_lag"] == 4 for x in result["blocks"])
    assert result["all_four_check_improve_at_least_10percent"]


def test_static_trajectory_abstains():
    a = np.zeros((400, 2))
    v = np.ones(400, bool)
    assert lag_screen(a, a, v, v)["status"] == "unobservable_degenerate_trajectory"


def test_common_support_missing_existence():
    a = np.zeros((100, 2))
    v = np.zeros(100, bool)
    assert lag_screen(a, a, v, v)["status"] == "insufficient_common_frames"


def test_boxes_are_xywh_native():
    centres, v = trajectory(
        {"gt_rect": [[10, 20, 20, 40], [0, 0, 0, 0]], "exist": [1, 0]}, [100, 200]
    )
    assert np.allclose(centres[0], [0.1, 0.4])
    assert v.tolist() == [True, False]


def test_absent_frame_empty_rectangle():
    centres, v = trajectory({"gt_rect": [[10, 20, 20, 40], []], "exist": [1, 0]}, [100, 200])
    assert np.allclose(centres[0], [0.1, 0.4])
    assert v.tolist() == [True, False]

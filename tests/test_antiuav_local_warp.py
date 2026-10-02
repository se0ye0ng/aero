import numpy as np

from scripts.probe_antiuav_local_warp import corner_iou


def test_translation_proxy():
    assert corner_iou(lambda p: p + [5, 2], [0, 0, 10, 10], [5, 2, 15, 12]) == 1


def test_unsupported_corner_rejects_whole_box():
    def predict(points):
        points = points.copy()
        points[0] = np.nan
        return points

    assert corner_iou(predict, [0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]) is None
    assert corner_iou(None, [0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]) is None

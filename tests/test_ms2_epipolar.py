import numpy as np

from scripts.analyze_ms2_epipolar import fundamental_matrix, line_distances, summary


def test_rectified_stereo_line_distances_and_reverse():
    k = np.array([[400.0, 0.0, 320.0], [0.0, 400.0, 128.0], [0.0, 0.0, 1.0]])
    t = np.eye(4)
    t[0, 3] = -0.3
    a = np.array([[100.0, 100.0], [120.0, 110.0]])
    b = np.array([[90.0, 100.0], [112.0, 114.0]])
    f = fundamental_matrix(k, k, t)
    d0, d1 = line_distances(a, b, f)
    np.testing.assert_allclose(d0, [0.0, 4.0], atol=1e-12)
    np.testing.assert_allclose(d1, [0.0, 4.0], atol=1e-12)
    reverse = fundamental_matrix(k, k, np.linalg.inv(t))
    np.testing.assert_allclose(line_distances(b, a, reverse)[0], d1, atol=1e-12)


def test_degenerate_lines_remain_failures():
    a = np.array([[1.0, 2.0]])
    d0, _ = line_distances(a, a, np.zeros((3, 3)))
    result = summary(d0)
    assert result["invalid"] == 1
    assert result["within_px"]["3.0"] == 0.0

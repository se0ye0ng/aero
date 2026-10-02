import numpy as np

from scripts.probe_external_roma import score


def test_native_pixel_centres_and_fixed_vs_ungated():
    w, h = 12, 10
    x, y = np.meshgrid(2 * (np.arange(w) + 0.5) / w - 1, 2 * (np.arange(h) + 0.5) / h - 1)
    grid = np.stack([x, y], axis=-1)
    # Normalized identity maps coordinates between different native image sizes.
    warp = np.concatenate([np.concatenate([grid, grid], -1)] * 2, axis=1)
    rgb = np.array([[2.0, 2.0], [3.0, 4.0], [5.0, 6.0], [6.0, 3.0]])
    thermal = (rgb + 0.5) * 2 - 0.5
    result = score(warp, np.full((h, 2 * w), 0.1), [(w, h), (2 * w, 2 * h)], [rgb, thermal])
    for direction in result["fixed_confidence_ge_half"]:
        assert direction["errors"] == [None] * 4
    for direction in result["ungated_diagnostic"]:
        np.testing.assert_allclose(direction["errors"], 0, atol=1e-12)

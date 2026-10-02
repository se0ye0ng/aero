import numpy as np

from aero_ir.registration.prewarp import compose, field_predictor, observed


def test_zero_field_preserves_centres_across_unequal_sizes():
    predict = field_predictor(np.zeros((2, 32, 32)), (640, 360), (640, 512))
    points = np.array([[319.5, 179.5], [99.5, 99.5]])
    expected = (points + 0.5) / [640, 360] * [640, 512] - 0.5
    np.testing.assert_allclose(predict(points), expected)
    assert np.isnan(predict(np.array([[-1.0, 50.0], [0.0, 0.0]]))).all()


def test_residual_composition_order_and_inverse():
    forward, reverse = compose(
        lambda p: p * 2,
        lambda p: p / 2,
        lambda p: p + [3, 5],
        lambda p: p - [3, 5],
        np.ones((200, 200), bool),
        rgb_top=10,
        ir_top=20,
    )
    query = np.array([[30.0, 40.0]])
    np.testing.assert_allclose(forward(query), [[63, 85]])
    np.testing.assert_allclose(reverse(forward(query)), query)
    assert np.isnan(forward(np.array([[30.0, 5.0]]))).all()


def test_unobserved_padding_cannot_be_recovered_by_matcher():
    mask = np.ones((20, 20), bool)
    mask[8, 8] = False
    assert not observed(mask, np.array([[8.2, 8.2]]))[0]
    forward, reverse = compose(
        lambda p: p, lambda p: p, lambda p: p, lambda p: p, mask, rgb_top=0, ir_top=0
    )
    for prediction in (forward, reverse):
        assert np.isnan(prediction(np.array([[8.0, 8.0]]))).all()
        assert np.isnan(prediction(np.array([[np.nan, 8.0]]))).all()

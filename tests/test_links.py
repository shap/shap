import numpy as np

from shap.links import logit


def test_logit_handles_probability_boundaries():
    values = np.array([0.0, 1.0])
    result = logit(values)

    assert np.isneginf(result[0])
    assert np.isposinf(result[1])


def test_logit_round_trips_finite_probabilities():
    values = np.array([1e-12, 1e-6, 0.1, 0.9, 1 - 1e-6, 1 - 1e-12])
    logits = logit(values)

    assert np.all(np.isfinite(logits))
    assert np.allclose(logit.inverse(logits), values, rtol=1e-12, atol=0.0)

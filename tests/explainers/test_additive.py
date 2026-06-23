"""Tests for shap.AdditiveExplainer."""

import numpy as np
import pytest

import shap


def _make_additive_model(coefficients):
    """Return a callable additive model ``f(X) = X @ coefficients``."""
    coefficients = np.asarray(coefficients, dtype=float)

    def model(X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return X @ coefficients

    return model


def _zero_centered_background(num_features, num_samples=200, seed=0):
    """Background sampled from a distribution with mean ~0 per feature.

    A near-zero background mean lets us compare SHAP values to closed-form
    expectations for additive models without subtracting a baseline shift.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=1.0, size=(num_samples, num_features))


def test_additive_explainer_recovers_known_linear_contributions():
    """For ``f(x) = c @ x`` with zero-mean background, phi_i ≈ c_i * x_i."""
    coefficients = np.array([2.0, 3.0, -1.0])
    model = _make_additive_model(coefficients)
    background = _zero_centered_background(num_features=3, num_samples=500, seed=42)
    masker = shap.maskers.Independent(background, max_samples=500)

    explainer = shap.AdditiveExplainer(model, masker)

    sample = np.array([[1.0, 1.0, 1.0]])
    explanation = explainer(sample)

    expected = coefficients * sample[0]
    np.testing.assert_allclose(explanation.values[0], expected, atol=0.2)


def test_additive_explainer_zero_contribution_for_unused_feature():
    """A feature with coefficient 0 must produce a SHAP value of 0."""
    coefficients = np.array([2.0, 0.0, 5.0])
    model = _make_additive_model(coefficients)
    background = _zero_centered_background(num_features=3, num_samples=500, seed=1)
    masker = shap.maskers.Independent(background, max_samples=500)

    explainer = shap.AdditiveExplainer(model, masker)

    sample = np.array([[7.0, 99.0, 4.0]])
    explanation = explainer(sample)

    assert abs(explanation.values[0, 1]) < 1e-8


def test_additive_explainer_explanation_shape_matches_input():
    """Returned Explanation must have shape ``(n_samples, n_features)``.

    Downstream plot functions (waterfall, beeswarm, bar) assume this
    shape contract; a silent shape regression breaks visualizations.
    """
    coefficients = np.array([1.0, -2.0, 0.5, 3.0])
    model = _make_additive_model(coefficients)
    background = _zero_centered_background(num_features=4, num_samples=200, seed=7)
    masker = shap.maskers.Independent(background, max_samples=200)

    explainer = shap.AdditiveExplainer(model, masker)

    samples = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [-1.0, 0.0, 1.0, -2.0],
            [0.5, 0.5, 0.5, 0.5],
        ]
    )
    explanation = explainer(samples)

    assert explanation.values.shape == (3, 4)
    assert explanation.data.shape == (3, 4)


def test_additive_explainer_values_sum_to_model_output_minus_baseline():
    """SHAP additivity: ``sum(phi_i) == f(x) - E[f(X)]`` for additive models."""
    coefficients = np.array([1.5, -0.5, 2.0])
    model = _make_additive_model(coefficients)
    background = _zero_centered_background(num_features=3, num_samples=500, seed=3)
    masker = shap.maskers.Independent(background, max_samples=500)

    explainer = shap.AdditiveExplainer(model, masker)

    sample = np.array([[0.7, -1.2, 2.5]])
    explanation = explainer(sample)

    model_output = float(model(sample)[0])
    baseline = float(model(background).mean())
    np.testing.assert_allclose(
        explanation.values[0].sum(),
        model_output - baseline,
        atol=0.2,
    )


def test_additive_explainer_rejects_non_independent_masker():
    """Constructor asserts when the masker is not ``shap.maskers.Independent``.

    This pins the documented limitation in the docstring and prevents silent
    misuse with maskers that would produce mathematically wrong values for
    the additive code path.
    """
    coefficients = np.array([1.0, 1.0])
    model = _make_additive_model(coefficients)

    class _NotAMasker: ...

    with pytest.raises(AssertionError, match="Tabular masker"):
        shap.AdditiveExplainer(model, _NotAMasker())


def test_additive_explainer_supports_model_with_masker_default_false():
    """``supports_model_with_masker`` rejects arbitrary callables.

    Auto-dispatch in ``shap.Explainer`` relies on this returning ``False``
    for non-EBM models so the right explainer is chosen.
    """
    model = _make_additive_model([1.0, 2.0])
    background = _zero_centered_background(num_features=2, num_samples=100, seed=0)
    masker = shap.maskers.Independent(background, max_samples=100)

    assert shap.AdditiveExplainer.supports_model_with_masker(model, masker) is False

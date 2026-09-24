"""Unit tests for the AdditiveExplainer."""

import numpy as np
import pytest

import shap
from shap.explainers._additive import AdditiveExplainer


def _sum_model(x):
    return x.sum(axis=1)


def _double_first_feature(x):
    return x[:, 0] * 2


def _make_additive_explainer(n_features=4, n_background=10):
    """Helper to create a simple AdditiveExplainer with a linear additive model."""
    background = np.random.randn(n_background, n_features)
    masker = shap.maskers.Independent(background)
    return AdditiveExplainer(_sum_model, masker), background, n_features


def test_additive_explainer_init():
    """Test that AdditiveExplainer initializes correctly."""
    explainer, background, n_features = _make_additive_explainer()
    assert explainer is not None
    assert hasattr(explainer, "_expected_value")
    assert hasattr(explainer, "_zero_offset")
    assert hasattr(explainer, "_input_offsets")
    assert len(explainer._input_offsets) == n_features


def test_additive_explainer_expected_value():
    """Test that expected value is computed correctly."""
    explainer, background, n_features = _make_additive_explainer()
    expected = background.sum(axis=1).mean()
    assert abs(float(explainer._expected_value) - expected) < 1e-6


def test_additive_explainer_shap_values_shape():
    """Test that SHAP values have correct shape."""
    explainer, background, n_features = _make_additive_explainer(n_features=4)
    X = np.random.randn(3, n_features)
    result = explainer(X, silent=True)
    assert result.values.shape == (3, n_features)


def test_additive_explainer_additivity():
    """Test that SHAP values sum to model output minus expected value."""
    np.random.seed(42)
    explainer, background, n_features = _make_additive_explainer(n_features=4)
    X = np.random.randn(5, n_features)
    result = explainer(X, silent=True)
    model_output = X.sum(axis=1)
    expected = float(explainer._expected_value)
    for i in range(len(X)):
        shap_sum = result.values[i].sum()
        diff = model_output[i] - expected
        assert abs(shap_sum - diff) < 1e-5, f"Additivity failed for sample {i}"


def test_additive_explainer_zero_input():
    """Test SHAP values for zero input."""
    np.random.seed(0)
    background = np.random.randn(10, 3)
    masker = shap.maskers.Independent(background)
    explainer = AdditiveExplainer(_sum_model, masker)
    X = np.zeros((1, 3))
    result = explainer(X, silent=True)
    assert result.values.shape == (1, 3)


def test_additive_explainer_explain_row():
    """Test explain_row returns expected keys."""
    explainer, background, n_features = _make_additive_explainer()
    x = np.random.randn(n_features)
    row_result = explainer.explain_row(
        x,
        max_evals="auto",
        main_effects=False,
        error_bounds=False,
        outputs=None,
        silent=True,
    )
    assert "values" in row_result
    assert "expected_values" in row_result
    assert "main_effects" in row_result
    assert len(row_result["values"]) == n_features


def test_additive_explainer_supports_model_with_masker_false():
    """Test that supports_model_with_masker returns False for unknown models."""
    masker = shap.maskers.Independent(np.random.randn(10, 4))
    result = AdditiveExplainer.supports_model_with_masker(_sum_model, masker)
    assert result is False


def test_additive_explainer_single_feature():
    """Test AdditiveExplainer with a single feature."""
    background = np.random.randn(10, 1)
    masker = shap.maskers.Independent(background)
    explainer = AdditiveExplainer(_double_first_feature, masker)
    X = np.random.randn(3, 1)
    result = explainer(X, silent=True)
    assert result.values.shape == (3, 1)


def test_additive_explainer_wrong_masker_raises():
    """Test that using non-Independent masker raises AssertionError."""
    background = np.random.randn(10, 4)
    masker = shap.maskers.Partition(background)
    with pytest.raises(AssertionError):
        AdditiveExplainer(_sum_model, masker)

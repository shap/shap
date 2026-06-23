import numpy as np
import pytest

from shap.explainers._additive import AdditiveExplainer
from shap.utils import safe_isinstance


class _DummyIndependentMasker:
    """Mock Independent masker for testing."""

    def __init__(self, data):
        self.data = np.asarray(data)
        self.shape = self.data.shape

    def __call__(self, mask, *args):
        """Return masked samples."""
        if isinstance(mask, np.ndarray) and mask.ndim > 1:
            results = []
            for m in mask:
                results.append(self.data[m].mean(axis=0))
            return np.array(results)
        return self.data[mask].mean(axis=0, keepdims=True)


class _DummyNonIndependentMasker:
    """Mock non-Independent masker for testing."""

    shape = (2, 3)


def dummy_model(X):
    """Simple linear model for testing."""
    return np.sum(X, axis=1)


def test_additive_with_independent_masker(monkeypatch):
    """Test AdditiveExplainer initialization with Independent masker."""
    background_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    masker = _DummyIndependentMasker(background_data)

    original_safe_isinstance = safe_isinstance

    def mock_safe_isinstance(obj, class_str):
        if class_str == "shap.maskers.Independent":
            return isinstance(obj, _DummyIndependentMasker)
        return original_safe_isinstance(obj, class_str)

    monkeypatch.setattr("shap.explainers._additive.safe_isinstance", mock_safe_isinstance)

    explainer = AdditiveExplainer(dummy_model, masker)

    assert explainer.masker is not None
    assert hasattr(explainer, "_zero_offset")
    assert hasattr(explainer, "_input_offsets")
    assert hasattr(explainer, "_expected_value")


def test_additive_with_non_independent_masker_raises_assertion(monkeypatch):
    """Test that non-Independent masker raises AssertionError."""
    masker = _DummyNonIndependentMasker()

    original_safe_isinstance = safe_isinstance

    def mock_safe_isinstance(obj, class_str):
        if class_str == "shap.maskers.Independent":
            return False
        return original_safe_isinstance(obj, class_str)

    monkeypatch.setattr("shap.explainers._additive.safe_isinstance", mock_safe_isinstance)

    with pytest.raises(AssertionError, match="only supports the Tabular masker"):
        AdditiveExplainer(dummy_model, masker)


def test_additive_ebm_without_masker_not_implemented(monkeypatch):
    """Test that EBM model without masker raises NotImplementedError."""

    class MockEBM:
        """Mock ExplainableBoostingClassifier."""

        def decision_function(self, X):
            return np.sum(X, axis=1)

        intercept_ = 0.5

    ebm_model = MockEBM()
    original_safe_isinstance = safe_isinstance

    def mock_safe_isinstance(obj, class_str):
        if class_str == "interpret.glassbox.ExplainableBoostingClassifier":
            return True
        return original_safe_isinstance(obj, class_str)

    monkeypatch.setattr("shap.explainers._additive.safe_isinstance", mock_safe_isinstance)

    with pytest.raises(NotImplementedError, match="Masker not given"):
        AdditiveExplainer(ebm_model, masker=None)


def test_additive_call_method(monkeypatch):
    """Test that __call__ method delegates to parent."""
    background_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    masker = _DummyIndependentMasker(background_data)

    original_safe_isinstance = safe_isinstance

    def mock_safe_isinstance(obj, class_str):
        if class_str == "shap.maskers.Independent":
            return isinstance(obj, _DummyIndependentMasker)
        return original_safe_isinstance(obj, class_str)

    monkeypatch.setattr("shap.explainers._additive.safe_isinstance", mock_safe_isinstance)

    explainer = AdditiveExplainer(dummy_model, masker, feature_names=["a", "b", "c"])

    # Call the explainer with test data
    X = np.array([[1.0, 2.0, 3.0]])
    result = explainer(X)

    # Result should be an Explanation object with SHAP values
    assert hasattr(result, "values")


def test_additive_supports_model_ebm_no_interactions(monkeypatch):
    """Test supports_model_with_masker returns True for EBM without interactions."""

    class MockEBM:
        """Mock ExplainableBoostingClassifier."""

        interactions = 0

    ebm_model = MockEBM()
    original_safe_isinstance = safe_isinstance

    def mock_safe_isinstance(obj, class_str):
        if class_str == "interpret.glassbox.ExplainableBoostingClassifier":
            return isinstance(obj, MockEBM)
        return original_safe_isinstance(obj, class_str)

    monkeypatch.setattr("shap.explainers._additive.safe_isinstance", mock_safe_isinstance)

    result = AdditiveExplainer.supports_model_with_masker(ebm_model, None)
    assert result is True


def test_additive_supports_model_ebm_with_interactions_raises(monkeypatch):
    """Test supports_model_with_masker raises NotImplementedError for EBM with interactions."""

    class MockEBM:
        """Mock ExplainableBoostingClassifier."""

        interactions = 2

    ebm_model = MockEBM()
    original_safe_isinstance = safe_isinstance

    def mock_safe_isinstance(obj, class_str):
        if class_str == "interpret.glassbox.ExplainableBoostingClassifier":
            return isinstance(obj, MockEBM)
        return original_safe_isinstance(obj, class_str)

    monkeypatch.setattr("shap.explainers._additive.safe_isinstance", mock_safe_isinstance)

    with pytest.raises(NotImplementedError, match="interaction effects"):
        AdditiveExplainer.supports_model_with_masker(ebm_model, None)


def test_additive_supports_model_non_ebm_returns_false(monkeypatch):
    """Test supports_model_with_masker returns False for non-EBM model."""

    def some_model(X):
        return np.sum(X, axis=1)

    original_safe_isinstance = safe_isinstance

    def mock_safe_isinstance(obj, class_str):
        if class_str == "interpret.glassbox.ExplainableBoostingClassifier":
            return False
        return original_safe_isinstance(obj, class_str)

    monkeypatch.setattr("shap.explainers._additive.safe_isinstance", mock_safe_isinstance)

    result = AdditiveExplainer.supports_model_with_masker(some_model, None)
    assert result is False

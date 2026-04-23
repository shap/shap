"""Unit tests for the AdditiveExplainer."""

import numpy as np
import pytest
from conftest import compare_numpy_outputs_against_baseline

import shap


@pytest.fixture
def create_additive_model():
    """Create a simple additive model (linear function) for testing."""

    def model(X):
        # Simple additive model: sum of features with different weights
        X = np.atleast_2d(X)
        return X[:, 0] * 2.0 + X[:, 1] * 3.0 + X[:, 2] * 1.5 + 1.0

    return model


@compare_numpy_outputs_against_baseline(func_file=__file__)
def _single_sample_with_baseline(model, background, test_sample):
    masker = shap.maskers.Independent(background)
    explainer = shap.AdditiveExplainer(model, masker)
    return explainer(test_sample)


@compare_numpy_outputs_against_baseline(func_file=__file__)
def _feature_names_with_baseline(model, background, test_data, feature_names):
    masker = shap.maskers.Independent(background)
    explainer = shap.AdditiveExplainer(model, masker, feature_names=feature_names)
    return explainer(test_data)


class TestAdditiveExplainer:
    """Test suite for AdditiveExplainer."""

    def test_basic_additivity(self, create_additive_model):
        """Test that SHAP values sum to the difference from expected value."""
        model = create_additive_model
        background = np.random.randn(50, 3)
        test_data = np.random.randn(5, 3)

        masker = shap.maskers.Independent(background)
        explainer = shap.AdditiveExplainer(model, masker)
        shap_values = explainer(test_data)

        # Check additivity: base_value + sum(shap_values) ≈ model(x)
        predictions = model(test_data)
        reconstructed = shap_values.base_values + shap_values.values.sum(axis=1)
        np.testing.assert_allclose(reconstructed, predictions, rtol=1e-5)

    def test_single_sample(self, create_additive_model):
        """Test explanation for a single sample."""
        model = create_additive_model
        background = np.random.randn(20, 3)
        test_sample = np.random.randn(1, 3)

        shap_values = _single_sample_with_baseline(model, background, test_sample)

        # Check shape
        assert shap_values.values.shape == (1, 3)

        # Check additivity
        prediction = model(test_sample)
        reconstructed = shap_values.base_values + shap_values.values.sum(axis=1)
        np.testing.assert_allclose(reconstructed, prediction, rtol=1e-5)

    def test_expected_value_property(self, create_additive_model):
        """Test that expected value is correctly computed."""
        model = create_additive_model
        background = np.random.randn(100, 3)

        masker = shap.maskers.Independent(background)
        explainer = shap.AdditiveExplainer(model, masker)

        # Expected value should be stored internally
        assert hasattr(explainer, "_expected_value")
        # And should be a scalar or 0-d array for single-output models
        assert np.isscalar(explainer._expected_value) or explainer._expected_value.ndim == 0

    def test_feature_names(self, create_additive_model):
        """Test that feature names are preserved."""
        model = create_additive_model
        background = np.random.randn(20, 3)
        test_data = np.random.randn(3, 3)
        feature_names = ["feature_a", "feature_b", "feature_c"]

        shap_values = _feature_names_with_baseline(model, background, test_data, feature_names)

        assert shap_values.feature_names == feature_names

    def test_main_effects(self, create_additive_model):
        """Test that main_effects parameter works."""
        model = create_additive_model
        background = np.random.randn(20, 3)
        test_data = np.random.randn(3, 3)

        masker = shap.maskers.Independent(background)
        explainer = shap.AdditiveExplainer(model, masker)
        shap_values = explainer(test_data, main_effects=True)

        # For additive models, main effects should equal SHAP values
        np.testing.assert_allclose(shap_values.main_effects, shap_values.values, rtol=1e-5)


class TestAdditiveExplainerMaskerValidation:
    """Tests for masker validation in AdditiveExplainer."""

    def test_requires_independent_masker(self, create_additive_model):
        """Test that AdditiveExplainer requires Independent masker."""
        model = create_additive_model
        background = np.random.randn(20, 3)

        # Using Partition masker should raise an error
        partition_masker = shap.maskers.Partition(background)
        with pytest.raises(AssertionError, match="Tabular masker"):
            shap.AdditiveExplainer(model, partition_masker)


class TestAdditiveExplainerSupportsModel:
    """Tests for the supports_model_with_masker static method."""

    def test_supports_ebm_classifier(self):
        """Test that EBM classifier is supported when available."""
        pytest.importorskip("interpret")
        from interpret.glassbox import ExplainableBoostingClassifier

        # Create a simple EBM classifier
        X = np.random.randn(50, 3)
        y = (X[:, 0] > 0).astype(int)

        ebm = ExplainableBoostingClassifier(interactions=0, max_rounds=5)
        ebm.fit(X, y)

        result = shap.AdditiveExplainer.supports_model_with_masker(ebm, None)
        assert result is True

    def test_ebm_with_interactions_raises(self):
        """Test that EBM with interactions raises NotImplementedError."""
        pytest.importorskip("interpret")
        from interpret.glassbox import ExplainableBoostingClassifier

        X = np.random.randn(50, 3)
        y = (X[:, 0] > 0).astype(int)

        ebm = ExplainableBoostingClassifier(interactions=1, max_rounds=5)
        ebm.fit(X, y)

        with pytest.raises(NotImplementedError, match="interaction effects"):
            shap.AdditiveExplainer.supports_model_with_masker(ebm, None)

    def test_generic_model_not_supported(self, create_additive_model):
        """Test that generic models return False for supports_model_with_masker."""
        model = create_additive_model
        result = shap.AdditiveExplainer.supports_model_with_masker(model, None)
        assert result is False


class TestAdditiveExplainerWithEBM:
    """Tests for AdditiveExplainer with ExplainableBoostingClassifier."""

    def test_ebm_without_masker_raises(self):
        """Test that using EBM without a masker raises NotImplementedError."""
        pytest.importorskip("interpret")
        from interpret.glassbox import ExplainableBoostingClassifier

        X = np.random.randn(50, 3)
        y = (X[:, 0] > 0).astype(int)

        ebm = ExplainableBoostingClassifier(interactions=0, max_rounds=5)
        ebm.fit(X, y)

        # Should raise because masker=None is not yet supported for EBM
        with pytest.raises(NotImplementedError, match="Masker not given"):
            shap.AdditiveExplainer(ebm, masker=None)

    def test_ebm_with_independent_masker(self):
        """Test AdditiveExplainer with EBM and Independent masker."""
        pytest.importorskip("interpret")
        from interpret.glassbox import ExplainableBoostingClassifier

        X = np.random.randn(100, 4)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        ebm = ExplainableBoostingClassifier(interactions=0, max_rounds=10)
        ebm.fit(X, y)

        masker = shap.maskers.Independent(X[:50])
        explainer = shap.AdditiveExplainer(ebm, masker)

        test_data = X[50:55]
        shap_values = explainer(test_data)

        # Check shape
        assert shap_values.values.shape[0] == 5
        assert shap_values.values.shape[1] == 4

        # Check additivity with decision_function
        predictions = ebm.decision_function(test_data)
        reconstructed = shap_values.base_values + shap_values.values.sum(axis=1)
        np.testing.assert_allclose(reconstructed, predictions, rtol=1e-4)


class TestAdditiveExplainerExplainRow:
    """Tests specifically targeting the explain_row method."""

    def test_explain_row_outputs(self, create_additive_model):
        """Test explain_row output keys, shapes, and main effects."""
        model = create_additive_model
        background = np.random.randn(20, 3)
        test_sample = np.random.randn(3)

        masker = shap.maskers.Independent(background)
        explainer = shap.AdditiveExplainer(model, masker)

        # Call explain_row directly
        result = explainer.explain_row(
            test_sample,
            max_evals="auto",
            main_effects=True,
            error_bounds=False,
            outputs=None,
            silent=True,
        )

        # Check that result contains expected keys
        assert "values" in result
        assert "expected_values" in result
        assert "mask_shapes" in result
        assert "main_effects" in result
        assert "clustering" in result

        # Check shapes
        assert result["values"].shape == (3,)
        assert result["mask_shapes"] == [(3,)]
        np.testing.assert_array_equal(result["values"], result["main_effects"])


class TestAdditiveExplainerEdgeCases:
    """Edge case tests for AdditiveExplainer."""

    def test_identical_samples(self, create_additive_model):
        """Test when all test samples are identical."""
        model = create_additive_model
        background = np.random.randn(20, 3)
        single_sample = np.random.randn(1, 3)
        test_data = np.repeat(single_sample, 5, axis=0)

        masker = shap.maskers.Independent(background)
        explainer = shap.AdditiveExplainer(model, masker)
        shap_values = explainer(test_data)

        # All SHAP values should be identical
        for i in range(1, 5):
            np.testing.assert_array_equal(shap_values.values[0], shap_values.values[i])

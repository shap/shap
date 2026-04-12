"""Additional unit tests for ExactExplainer.

These tests target code paths in `shap/explainers/_exact.py` that the existing
test suite (which depends on xgboost) leaves uncovered. They use simple Python
functions as models and the lightweight `Independent` / `Partition` maskers.
"""

import numpy as np
import pytest

import shap

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _linear_model(X):
    """Linear single-output model: y = 2*x0 - x1 + 0.5*x2."""
    return X[:, 0] * 2 + X[:, 1] * (-1) + X[:, 2] * 0.5


def _linear_multi_output(X):
    """Linear multi-output model: returns two columns."""
    a = X[:, 0] + X[:, 1]
    b = X[:, 0] - X[:, 2]
    return np.column_stack([a, b])


@pytest.fixture()
def background_data():
    rng = np.random.default_rng(0)
    return rng.standard_normal((20, 3))


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestExactExplainerConstruction:
    """Tests for ExactExplainer.__init__."""

    def test_basic_construction_independent_masker(self, background_data):
        masker = shap.maskers.Independent(background_data)
        explainer = shap.explainers.ExactExplainer(_linear_model, masker)
        assert explainer is not None
        assert explainer._gray_code_cache == {}

    def test_construction_with_partition_masker(self, background_data):
        masker = shap.maskers.Partition(background_data)
        explainer = shap.explainers.ExactExplainer(_linear_model, masker)
        # partition masker has clustering, so partition_masks should be precomputed
        assert hasattr(explainer, "_partition_masks")
        assert hasattr(explainer, "_partition_masks_inds")
        assert hasattr(explainer, "_partition_delta_indexes")


# ---------------------------------------------------------------------------
# __call__ / explain_row (standard Shapley path)
# ---------------------------------------------------------------------------


class TestExactExplainerStandardShapley:
    """Tests for the standard (non-clustered) Shapley value path."""

    def test_output_shape(self, background_data):
        masker = shap.maskers.Independent(background_data)
        explainer = shap.explainers.ExactExplainer(_linear_model, masker)

        test_X = np.array([[1.0, 2.0, 3.0]])
        sv = explainer(test_X)
        assert sv.values.shape == (1, 3)

    def test_additivity_single_output(self, background_data):
        masker = shap.maskers.Independent(background_data)
        explainer = shap.explainers.ExactExplainer(_linear_model, masker)

        rng = np.random.default_rng(1)
        test_X = rng.standard_normal((4, 3))
        sv = explainer(test_X)

        model_output = _linear_model(test_X)
        reconstructed = sv.base_values + sv.values.sum(axis=1)
        np.testing.assert_allclose(reconstructed, model_output, atol=1e-6)

    def test_additivity_multi_output(self, background_data):
        masker = shap.maskers.Independent(background_data)
        explainer = shap.explainers.ExactExplainer(_linear_multi_output, masker)

        rng = np.random.default_rng(2)
        test_X = rng.standard_normal((3, 3))
        sv = explainer(test_X)

        model_output = _linear_multi_output(test_X)
        reconstructed = sv.base_values + sv.values.sum(axis=1)
        np.testing.assert_allclose(reconstructed, model_output, atol=1e-6)

    def test_gray_code_cache_populated(self, background_data):
        """The gray code cache should be populated after a call."""
        masker = shap.maskers.Independent(background_data)
        explainer = shap.explainers.ExactExplainer(_linear_model, masker)

        test_X = np.array([[1.0, 2.0, 3.0]])
        explainer(test_X)
        # cache should now contain at least one entry
        assert len(explainer._gray_code_cache) > 0


# ---------------------------------------------------------------------------
# Interactions
# ---------------------------------------------------------------------------


class TestExactExplainerInteractions:
    """Tests for the Shapley-Taylor interaction path."""

    def test_interactions_shape(self, background_data):
        masker = shap.maskers.Independent(background_data)
        explainer = shap.explainers.ExactExplainer(_linear_model, masker)

        test_X = np.array([[1.0, 2.0, 3.0]])
        sv = explainer(test_X, interactions=2)
        # interaction values should be a square matrix per row
        assert sv.values.shape == (1, 3, 3)

    def test_interactions_true_equivalent_to_2(self, background_data):
        masker = shap.maskers.Independent(background_data)
        explainer = shap.explainers.ExactExplainer(_linear_model, masker)

        test_X = np.array([[1.0, 2.0, 3.0]])
        sv_true = explainer(test_X, interactions=True)
        sv_2 = explainer(test_X, interactions=2)
        np.testing.assert_allclose(sv_true.values, sv_2.values, atol=1e-10)

    def test_interactions_higher_than_2_raises(self, background_data):
        masker = shap.maskers.Independent(background_data)
        explainer = shap.explainers.ExactExplainer(_linear_model, masker)

        test_X = np.array([[1.0, 2.0, 3.0]])
        with pytest.raises(NotImplementedError, match="higher than order 2"):
            explainer(test_X, interactions=3)


# ---------------------------------------------------------------------------
# max_evals validation
# ---------------------------------------------------------------------------


class TestExactExplainerMaxEvals:
    """Tests for max_evals validation in explain_row."""

    def test_max_evals_too_low_raises(self, background_data):
        masker = shap.maskers.Independent(background_data)
        explainer = shap.explainers.ExactExplainer(_linear_model, masker)

        test_X = np.array([[1.0, 2.0, 3.0]])  # 3 features → needs 2^3 = 8 evals
        with pytest.raises(ValueError, match="max_evals"):
            explainer(test_X, max_evals=4)

    def test_max_evals_auto_works(self, background_data):
        masker = shap.maskers.Independent(background_data)
        explainer = shap.explainers.ExactExplainer(_linear_model, masker)

        test_X = np.array([[1.0, 2.0, 3.0]])
        sv = explainer(test_X, max_evals="auto")
        assert sv.values.shape == (1, 3)


# ---------------------------------------------------------------------------
# Partition masker path (clustered Shapley)
# ---------------------------------------------------------------------------


class TestExactExplainerPartitionPath:
    """Tests for the clustered Shapley value path."""

    def test_partition_masker_basic(self, background_data):
        masker = shap.maskers.Partition(background_data)
        explainer = shap.explainers.ExactExplainer(_linear_model, masker)

        test_X = np.array([[1.0, 2.0, 3.0]])
        sv = explainer(test_X)
        assert sv.values.shape == (1, 3)

    def test_partition_masker_additivity(self, background_data):
        masker = shap.maskers.Partition(background_data)
        explainer = shap.explainers.ExactExplainer(_linear_model, masker)

        rng = np.random.default_rng(3)
        test_X = rng.standard_normal((3, 3))
        sv = explainer(test_X)

        model_output = _linear_model(test_X)
        reconstructed = sv.base_values + sv.values.sum(axis=1)
        np.testing.assert_allclose(reconstructed, model_output, atol=1e-6)

    def test_partition_max_evals_too_low_raises(self, background_data):
        masker = shap.maskers.Partition(background_data)
        explainer = shap.explainers.ExactExplainer(_linear_model, masker)

        test_X = np.array([[1.0, 2.0, 3.0]])
        # for the clustered path we need len(fm)**2 evals (3**2 = 9)
        with pytest.raises(ValueError, match="max_evals"):
            explainer(test_X, max_evals=4)


# ---------------------------------------------------------------------------
# main_effects
# ---------------------------------------------------------------------------


class TestExactExplainerMainEffects:
    """Tests for the main_effects code path."""

    def test_main_effects_returned(self, background_data):
        masker = shap.maskers.Independent(background_data)
        explainer = shap.explainers.ExactExplainer(_linear_model, masker)

        test_X = np.array([[1.0, 2.0, 3.0]])
        sv = explainer(test_X, main_effects=True)
        # additivity should still hold
        reconstructed = sv.base_values + sv.values.sum(axis=1)
        np.testing.assert_allclose(reconstructed, _linear_model(test_X), atol=1e-6)


# ---------------------------------------------------------------------------
# Helper functions: gray_code_indexes / gray_code_masks
# ---------------------------------------------------------------------------


class TestGrayCodeHelpers:
    """Tests for the module-level gray code helper functions."""

    def test_gray_code_masks_shape(self):
        from shap.explainers._exact import gray_code_masks

        out = gray_code_masks(3)
        assert out.shape == (8, 3)
        assert out.dtype == bool

    def test_gray_code_masks_unique(self):
        """All 2^n binary patterns should appear exactly once."""
        from shap.explainers._exact import gray_code_masks

        out = gray_code_masks(4)
        rows = {tuple(r) for r in out}
        assert len(rows) == 16

    def test_gray_code_masks_single_bit_flip(self):
        """Adjacent rows should differ in exactly one bit (gray code property)."""
        from shap.explainers._exact import gray_code_masks

        out = gray_code_masks(4)
        for i in range(1, len(out)):
            diff = np.sum(out[i] != out[i - 1])
            assert diff == 1

    def test_gray_code_indexes_length(self):
        from shap.explainers._exact import gray_code_indexes

        out = gray_code_indexes(3)
        assert len(out) == 8

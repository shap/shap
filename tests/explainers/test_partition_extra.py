"""Additional unit tests for PartitionExplainer.

These tests target code paths in `shap/explainers/_partition.py` that the
existing test suite leaves uncovered.
"""

import numpy as np
import pytest

import shap


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


class TestPartitionExplainerConstruction:
    """Tests for PartitionExplainer.__init__."""

    def test_basic_construction(self, background_data):
        masker = shap.maskers.Partition(background_data)
        explainer = shap.explainers.PartitionExplainer(_linear_model, masker)
        assert explainer is not None
        assert explainer.expected_value is None

    def test_rejects_masker_without_clustering(self, background_data):
        """A masker without a `clustering` attribute should raise ValueError."""
        masker = shap.maskers.Independent(background_data)
        with pytest.raises(ValueError, match="clustering"):
            shap.explainers.PartitionExplainer(_linear_model, masker)

    def test_input_shape_set(self, background_data):
        masker = shap.maskers.Partition(background_data)
        explainer = shap.explainers.PartitionExplainer(_linear_model, masker)
        assert explainer.input_shape == (3,)

    def test_call_args_default_override(self, background_data):
        """Passing extra call_args should rewrap __call__ with new defaults."""
        masker = shap.maskers.Partition(background_data)
        explainer = shap.explainers.PartitionExplainer(_linear_model, masker, max_evals=200)

        # the new default for max_evals should be 200 on this instance
        assert explainer.__call__.__kwdefaults__["max_evals"] == 200


# __call__ / explain_row


class TestPartitionExplainerCall:
    """Tests for PartitionExplainer.__call__ output."""

    def test_output_shape_single_output(self, background_data):
        masker = shap.maskers.Partition(background_data)
        explainer = shap.explainers.PartitionExplainer(_linear_model, masker)

        test_X = np.array([[1.0, 2.0, 3.0]])
        sv = explainer(test_X)
        assert sv.values.shape == (1, 3)

    def test_output_shape_multi_row(self, background_data):
        masker = shap.maskers.Partition(background_data)
        explainer = shap.explainers.PartitionExplainer(_linear_model, masker)

        rng = np.random.default_rng(1)
        test_X = rng.standard_normal((4, 3))
        sv = explainer(test_X)
        assert sv.values.shape == (4, 3)

    def test_additivity_single_output(self, background_data):
        masker = shap.maskers.Partition(background_data)
        explainer = shap.explainers.PartitionExplainer(_linear_model, masker)

        rng = np.random.default_rng(2)
        test_X = rng.standard_normal((4, 3))
        sv = explainer(test_X)

        model_output = _linear_model(test_X)
        reconstructed = sv.base_values + sv.values.sum(axis=1)
        np.testing.assert_allclose(reconstructed, model_output, atol=1e-6)

    def test_additivity_multi_output(self, background_data):
        masker = shap.maskers.Partition(background_data)
        explainer = shap.explainers.PartitionExplainer(_linear_multi_output, masker)

        rng = np.random.default_rng(3)
        test_X = rng.standard_normal((3, 3))
        sv = explainer(test_X)

        model_output = _linear_multi_output(test_X)
        reconstructed = sv.base_values + sv.values.sum(axis=1)
        np.testing.assert_allclose(reconstructed, model_output, atol=1e-6)


# fixed_context


class TestPartitionFixedContext:
    """Tests for the fixed_context parameter."""

    @pytest.mark.parametrize("fixed_context", [None, 0, 1])
    def test_valid_fixed_context_values(self, background_data, fixed_context):
        masker = shap.maskers.Partition(background_data)
        explainer = shap.explainers.PartitionExplainer(_linear_model, masker)

        test_X = np.array([[1.0, 2.0, 3.0]])
        sv = explainer(test_X, fixed_context=fixed_context)
        assert sv.values.shape == (1, 3)

    def test_invalid_fixed_context_raises(self, background_data):
        masker = shap.maskers.Partition(background_data)
        explainer = shap.explainers.PartitionExplainer(_linear_model, masker)

        test_X = np.array([[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError, match="fixed_context"):
            explainer(test_X, fixed_context=42)


# Misc


class TestPartitionMisc:
    """Miscellaneous tests for PartitionExplainer."""

    def test_str_repr(self, background_data):
        masker = shap.maskers.Partition(background_data)
        explainer = shap.explainers.PartitionExplainer(_linear_model, masker)
        assert "PartitionExplainer" in str(explainer)

    def test_max_evals_auto(self, background_data):
        masker = shap.maskers.Partition(background_data)
        explainer = shap.explainers.PartitionExplainer(_linear_model, masker)

        test_X = np.array([[1.0, 2.0, 3.0]])
        sv = explainer(test_X, max_evals="auto")
        assert sv.values.shape == (1, 3)

    def test_dvalues_populated(self, background_data):
        """After a call, the explainer should have populated dvalues."""
        masker = shap.maskers.Partition(background_data)
        explainer = shap.explainers.PartitionExplainer(_linear_model, masker)

        test_X = np.array([[1.0, 2.0, 3.0]])
        explainer(test_X)
        assert explainer.dvalues is not None
        assert explainer.values is not None

    def test_batch_size_explicit(self, background_data):
        masker = shap.maskers.Partition(background_data)
        explainer = shap.explainers.PartitionExplainer(_linear_model, masker)

        test_X = np.array([[1.0, 2.0, 3.0]])
        sv = explainer(test_X, batch_size=2)
        assert sv.values.shape == (1, 3)

"""Unit tests for the Permutation explainer."""

import pickle

import numpy as np
import pytest

import shap

from . import common


def simple_sum_model(x):
    return np.sum(x, axis=1)


def test_exact_second_order():
    rs = np.random.RandomState(42)
    data = rs.randint(0, 2, size=(100, 5))

    def model(data):
        return data[:, 0] * data[:, 2] + data[:, 1] + data[:, 2] + data[:, 2] * data[:, 3]

    right_answer = np.zeros(data.shape)
    right_answer[:, 0] += (data[:, 0] * data[:, 2]) / 2
    right_answer[:, 2] += (data[:, 0] * data[:, 2]) / 2
    right_answer[:, 1] += data[:, 1]
    right_answer[:, 2] += data[:, 2]
    right_answer[:, 2] += (data[:, 2] * data[:, 3]) / 2
    right_answer[:, 3] += (data[:, 2] * data[:, 3]) / 2

    shap_values = shap.explainers.PermutationExplainer(model, np.zeros((1, 5)))(data)

    assert np.allclose(right_answer, shap_values.values)  # type: ignore[union-attr]


def test_multi_output_parity_with_invariant_feature():
    background = np.array([[0.0, 5.0, 2.0], [2.0, 5.0, 0.0]])
    rows = np.array([[4.0, 5.0, 3.0]])

    def model(data):
        left, invariant, right = data.T
        interaction = left * right
        return np.column_stack((interaction + 2 * left + invariant, 2 * interaction + right))

    # Coalition means from replacing the selected columns of the background.
    base = np.array([7.0, 1.0])
    left_only = np.array([17.0, 9.0])
    right_only = np.array([10.0, 9.0])
    both = np.array([25.0, 27.0])
    interaction = (both - left_only - right_only + base) / 2

    permutation = shap.PermutationExplainer(model, background, seed=42)(rows, max_evals=15, error_bounds=True)
    assert isinstance(permutation, shap.Explanation)
    expected = np.zeros((1, 3, 2))
    expected[0, 0] = left_only - base + interaction
    expected[0, 2] = right_only - base + interaction
    expected_std = np.zeros_like(expected)
    expected_std[0, [0, 2]] = np.abs(interaction)
    np.testing.assert_array_equal(permutation.base_values, base[None, :])
    np.testing.assert_array_equal(permutation.values, expected)
    np.testing.assert_array_equal(permutation.error_std, expected_std)

    exact = shap.ExactExplainer(model, background)(rows)
    np.testing.assert_array_equal(exact.base_values, base[None, :])
    np.testing.assert_array_equal(exact.values, expected)


# TODO: add baseline comparison once PermutationExplainer supports passing a numpy.random.Generator
# for reproducible results (currently uses global np.random state)


@pytest.mark.parametrize(
    "predict_fn, masker",
    [
        ("predict", None),
        ("predict_proba", None),
        ("predict", shap.maskers.Partition),
        ("predict_proba", shap.maskers.Partition),
        ("predict", shap.maskers.Independent),
        ("predict_proba", shap.maskers.Independent),
    ],
)
def test_tabular_additivity(predict_fn, masker):
    model, data = common.basic_xgboost_scenario(100)
    fn = getattr(model, predict_fn)

    masker_instance = data if masker is None else masker(data)

    common.test_additivity(
        shap.explainers.PermutationExplainer,
        fn,
        masker_instance,
        data,
    )


def test_serialization():
    model, data = common.basic_xgboost_scenario()
    common.test_serialization(
        shap.explainers.PermutationExplainer, model.predict, data, data, rtol=0.1, atol=0.05, max_evals=100000
    )


def test_serialization_no_model_or_masker():
    model, data = common.basic_xgboost_scenario()
    common.test_serialization(
        shap.explainers.PermutationExplainer,
        model.predict,
        data,
        data,
        model_saver=False,
        masker_saver=False,
        model_loader=lambda _: model.predict,
        masker_loader=lambda _: data,
        rtol=0.1,
        atol=0.05,
        max_evals=100000,
    )


def test_serialization_custom_model_save():
    model, data = common.basic_xgboost_scenario()
    common.test_serialization(
        shap.explainers.PermutationExplainer,
        model.predict,
        data,
        data,
        model_saver=pickle.dump,
        model_loader=pickle.load,
        rtol=0.1,
        atol=0.05,
        max_evals=100000,
    )


def test_max_evals_too_low():
    rs = np.random.RandomState(42)
    data = rs.rand(10, 3)

    explainer = shap.explainers.PermutationExplainer(simple_sum_model, data)

    with pytest.raises(ValueError):
        explainer(data, max_evals=1)


def test_seed_reproducibility():
    rs = np.random.RandomState(42)
    data = rs.rand(10, 4)

    explainer1 = shap.explainers.PermutationExplainer(simple_sum_model, data, seed=42)
    explainer2 = shap.explainers.PermutationExplainer(simple_sum_model, data, seed=42)

    vals1 = explainer1(data).values
    vals2 = explainer2(data).values

    assert np.allclose(vals1, vals2)


def test_main_effects():
    rs = np.random.RandomState(42)
    data = rs.rand(5, 3)

    explainer = shap.explainers.PermutationExplainer(simple_sum_model, data)
    explanation = explainer(data, main_effects=True)

    # Just verify no crash and shape consistency if present
    if explanation.main_effects is not None:
        assert explanation.main_effects.shape == data.shape


def test_error_bounds():
    rs = np.random.RandomState(42)
    data = rs.rand(5, 3)

    explainer = shap.explainers.PermutationExplainer(simple_sum_model, data)
    explanation = explainer(data, error_bounds=True)

    assert explanation.error_std is not None


def test_constant_features():
    rs = np.random.RandomState(42)  # Adding rs here for consistency
    data = np.ones((10, 4)) * rs.uniform(1.0, 1.0)

    explainer = shap.explainers.PermutationExplainer(simple_sum_model, data)
    explanation = explainer(data)

    assert np.allclose(explanation.values, 0)

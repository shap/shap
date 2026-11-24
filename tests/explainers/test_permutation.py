"""Unit tests for the Permutation explainer."""

import pickle

import numpy as np

import shap

from . import common


def test_exact_second_order():
    """This tests that the Perumtation explain gives exact answers for second order functions."""
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
def test_tabular_single_output_auto_masker():
    model, data = common.basic_xgboost_scenario(100)
    common.test_additivity(shap.explainers.PermutationExplainer, model.predict, data, data)


def test_tabular_multi_output_auto_masker():
    model, data = common.basic_xgboost_scenario(100)
    common.test_additivity(shap.explainers.PermutationExplainer, model.predict_proba, data, data)


def test_tabular_single_output_partition_masker():
    model, data = common.basic_xgboost_scenario(100)
    common.test_additivity(shap.explainers.PermutationExplainer, model.predict, shap.maskers.Partition(data), data)


def test_tabular_multi_output_partition_masker():
    model, data = common.basic_xgboost_scenario(100)
    common.test_additivity(
        shap.explainers.PermutationExplainer, model.predict_proba, shap.maskers.Partition(data), data
    )


def test_tabular_single_output_independent_masker():
    model, data = common.basic_xgboost_scenario(100)
    common.test_additivity(shap.explainers.PermutationExplainer, model.predict, shap.maskers.Independent(data), data)


def test_tabular_multi_output_independent_masker():
    model, data = common.basic_xgboost_scenario(100)
    common.test_additivity(
        shap.explainers.PermutationExplainer, model.predict_proba, shap.maskers.Independent(data), data
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

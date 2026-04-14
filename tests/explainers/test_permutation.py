"""Unit tests for the Permutation explainer."""

import pickle

import numpy as np
import pytest

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


def test_permutation_explainer_output_names_passthrough():

    # Simple dummy model that returns 2 columns
    def dummy_model(x):
        return np.ones((x.shape[0], 2))

    X = np.zeros((5, 3))
    names = ["Class A", "Class B"]

    # Initialize with output_names
    explainer = shap.PermutationExplainer(dummy_model, X, output_names=names)
    explanation = explainer(X[:1])

    assert list(explanation.output_names) == names, f"Expected {names}, but got {explanation.output_names}"


def test_permutation_explainer_slicing_by_output_name():

    def dummy_model(x):
        return np.random.rand(x.shape[0], 3)

    X = np.random.rand(10, 4)
    names = ["Setosa", "Versicolor", "Virginica"]

    explainer = shap.PermutationExplainer(dummy_model, X, output_names=names)
    explanation = explainer(X[:2])

    try:
        sliced_ext = explanation[:, :, "Versicolor"]
        assert sliced_ext.shape == (2, 4)
    except KeyError:
        pytest.fail("Explanation could not be sliced by output_name. Metadata mapping failed.")

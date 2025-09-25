"""Unit tests for the Permutation explainer."""

import pickle

import numpy as np

import shap

from . import common


def test_exact_second_order(random_seed):
    """This tests that the Perumtation explain gives exact answers for second order functions."""
    rs = np.random.RandomState(random_seed)
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

    assert np.allclose(right_answer, shap_values.values)


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

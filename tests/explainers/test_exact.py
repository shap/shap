"""Unit tests for the Exact explainer."""

import pickle

from conftest import compare_numpy_outputs_against_baseline

import shap

from . import common


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_interactions():
    model, data = common.basic_xgboost_scenario(100)
    return common.test_interactions_additivity(shap.explainers.ExactExplainer, model.predict, data, data)


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_tabular_single_output_auto_masker():
    model, data = common.basic_xgboost_scenario(100)
    return common.test_additivity(shap.explainers.ExactExplainer, model.predict, data, data)


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_tabular_multi_output_auto_masker():
    model, data = common.basic_xgboost_scenario(100)
    return common.test_additivity(shap.explainers.ExactExplainer, model.predict_proba, data, data)


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_tabular_single_output_partition_masker():
    model, data = common.basic_xgboost_scenario(100)
    return common.test_additivity(shap.explainers.ExactExplainer, model.predict, shap.maskers.Partition(data), data)


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_tabular_multi_output_partition_masker():
    model, data = common.basic_xgboost_scenario(100)
    return common.test_additivity(
        shap.explainers.ExactExplainer, model.predict_proba, shap.maskers.Partition(data), data
    )


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_tabular_single_output_independent_masker():
    model, data = common.basic_xgboost_scenario(100)
    return common.test_additivity(shap.explainers.ExactExplainer, model.predict, shap.maskers.Independent(data), data)


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_tabular_multi_output_independent_masker():
    model, data = common.basic_xgboost_scenario(100)
    return common.test_additivity(
        shap.explainers.ExactExplainer, model.predict_proba, shap.maskers.Independent(data), data
    )


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_serialization():
    model, data = common.basic_xgboost_scenario()
    return common.test_serialization(shap.explainers.ExactExplainer, model.predict, data, data)


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_serialization_no_model_or_masker():
    model, data = common.basic_xgboost_scenario()
    return common.test_serialization(
        shap.explainers.ExactExplainer,
        model.predict,
        data,
        data,
        model_saver=False,
        masker_saver=False,
        model_loader=lambda _: model.predict,
        masker_loader=lambda _: data,
    )


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_serialization_custom_model_save():
    model, data = common.basic_xgboost_scenario()
    return common.test_serialization(
        shap.explainers.ExactExplainer, model.predict, data, data, model_saver=pickle.dump, model_loader=pickle.load
    )

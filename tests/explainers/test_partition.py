"""This file contains tests for partition explainer."""

import pickle

import numpy as np
from conftest import compare_numpy_outputs_against_baseline

import shap

from . import common


def test_translation(basic_translation_scenario):
    model, tokenizer, data = basic_translation_scenario
    return common.test_additivity(shap.explainers.PartitionExplainer, model, tokenizer, data)


def test_translation_auto(basic_translation_scenario):
    model, tokenizer, data = basic_translation_scenario
    return common.test_additivity(shap.Explainer, model, tokenizer, data)


def test_translation_algorithm_arg(basic_translation_scenario):
    model, tokenizer, data = basic_translation_scenario
    return common.test_additivity(shap.Explainer, model, tokenizer, data, algorithm="partition")


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_tabular_single_output():
    model, data = common.basic_xgboost_scenario(100)
    return common.test_additivity(shap.explainers.PartitionExplainer, model.predict, shap.maskers.Partition(data), data)


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_tabular_multi_output():
    model, data = common.basic_xgboost_scenario(100)
    return common.test_additivity(
        shap.explainers.PartitionExplainer, model.predict_proba, shap.maskers.Partition(data), data
    )


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_serialization(basic_translation_scenario):
    model, tokenizer, data = basic_translation_scenario
    return common.test_serialization(shap.explainers.PartitionExplainer, model, tokenizer, data)


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_serialization_no_model_or_masker(basic_translation_scenario):
    model, tokenizer, data = basic_translation_scenario
    return common.test_serialization(
        shap.explainers.Partition,
        model,
        tokenizer,
        data,
        model_saver=None,
        masker_saver=None,
        model_loader=lambda _: model,
        masker_loader=lambda _: tokenizer,
    )


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_serialization_custom_model_save(basic_translation_scenario):
    model, tokenizer, data = basic_translation_scenario
    return common.test_serialization(
        shap.explainers.PartitionExplainer, model, tokenizer, data, model_saver=pickle.dump, model_loader=pickle.load
    )


def test_partition_linearized_logit_link():
    background = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.1]])
    samples = np.array([[2.0, 3.0], [1.5, 2.0]])

    def model(x):
        return 0.2 + 0.1 * x.sum(axis=1)

    masker = shap.maskers.Partition(background, clustering=np.array([[0, 1, 1, 2]]))
    explainer = shap.Explainer(model, masker, algorithm="partition", link=shap.links.logit)
    explanation = explainer(samples)

    assert np.isfinite(explanation.values).all()
    np.testing.assert_allclose(explanation.base_values, shap.links.logit(model(background).mean()))
    np.testing.assert_allclose(
        explanation.base_values + explanation.values.sum(axis=1), shap.links.logit(model(samples))
    )

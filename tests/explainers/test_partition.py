"""This file contains tests for partition explainer."""

import pickle

import numpy as np
from conftest import compare_numpy_outputs_against_baseline

import shap
from shap.explainers._partition import lower_credit

from . import common


def test_lower_credit_matches_baseline_operation_order():
    """``lower_credit`` must reproduce the numba baseline's arithmetic exactly.

    The baseline computed the credit handed to a child as
    ``values[parent] * lsize / group_size`` -- multiply, then divide. The C++
    version hoists ``lsize / group_size`` into a ``factor`` argument and computes
    ``factor * values[parent]`` -- divide, then multiply. Floating point is not
    associative, so the two round differently, and ``lower_credit`` compounds the
    difference down every level of the clustering tree.

    Here node 3 = {0, 1} and node 4 = {node 3, leaf 2}, so leaf 2 receives a
    factor of 1/3 -- inexact in binary, which is what exposes the reassociation.
    """
    clustering = np.array([[0.0, 1.0, 0.0, 2.0], [3.0, 2.0, 0.0, 3.0]])
    M = 3
    values = np.zeros(2 * M - 1, dtype=np.float64)
    values[-1] = 4.2

    lower_credit(
        i=len(values) - 1,
        M=M,
        prev_i=len(values) - 1,
        factor=0.0,
        values=values,
        clustering=clustering,
    )

    # leaf 2 is a returned SHAP value: values[:M] is what explain_row hands back
    assert values[2] == 4.2 * 1 / 3


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

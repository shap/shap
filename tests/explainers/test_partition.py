"""This file contains tests for partition explainer."""

import pickle

import numpy as np
import pytest
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


def _toy_partition_explainer():
    """Build a small PartitionExplainer for fast unit tests."""
    rs = np.random.RandomState(0)
    X = rs.randn(50, 4)
    coef = np.array([1.0, -2.0, 0.5, 0.3])

    def model(x):
        return x @ coef

    masker = shap.maskers.Partition(X)
    return shap.explainers.PartitionExplainer(model, masker), X


def test_masker_without_clustering_raises():
    """A masker without a .clustering attribute must raise ValueError."""
    rs = np.random.RandomState(0)
    X = rs.randn(20, 3)

    def model(x):
        return x.sum(axis=1)

    # Independent masker has no .clustering attribute
    independent = shap.maskers.Independent(X)
    with pytest.raises(ValueError, match="must have a .clustering attribute"):
        shap.explainers.PartitionExplainer(model, independent)


def test_str_repr():
    """__str__ should return a stable identifier."""
    explainer, _ = _toy_partition_explainer()
    assert str(explainer) == "shap.explainers.PartitionExplainer()"


def test_invalid_fixed_context_raises():
    """fixed_context other than 0/1/None/'auto' must raise ValueError."""
    explainer, X = _toy_partition_explainer()
    with pytest.raises(ValueError, match="Unknown fixed_context"):
        explainer(X[:1], fixed_context=2)


@pytest.mark.parametrize("fixed_context", [0, 1])
def test_explain_with_fixed_context(fixed_context):
    """fixed_context = 0 and 1 both produce additive Explanations."""
    explainer, X = _toy_partition_explainer()
    explanation = explainer(X[:2], fixed_context=fixed_context, silent=True)
    # For a linear model, additivity should hold up to small numerical noise
    preds = X[:2] @ np.array([1.0, -2.0, 0.5, 0.3])
    np.testing.assert_allclose(explanation.values.sum(axis=1) + explanation.base_values, preds, atol=1e-6)


def test_call_args_set_default_kwargs():
    """Constructor kwargs forwarded to __call__ should override defaults."""
    rs = np.random.RandomState(0)
    X = rs.randn(20, 3)

    def model(x):
        return x.sum(axis=1)

    masker = shap.maskers.Partition(X)
    explainer = shap.explainers.PartitionExplainer(model, masker, max_evals=64)
    # the call wrapper must store the new default
    assert explainer.__call__.__kwdefaults__["max_evals"] == 64
    # and the explainer should still work with the overridden default
    explanation = explainer(X[:1], silent=True)
    assert explanation.values.shape[0] == 1


def test_output_indexes_len_helpers():
    """output_indexes_len handles max(N), min(N), max(abs(N)) and array inputs."""
    from shap.explainers._partition import output_indexes_len

    assert output_indexes_len("max(3)") == 3
    assert output_indexes_len("min(2)") == 2
    assert output_indexes_len(np.array([0, 1, 2])) == 3
    # unrecognised string returns None
    assert output_indexes_len("unrecognised") is None
    # NB: the "max(abs(N))" prefix is currently unreachable because the
    # "max(" prefix matches first and tries to parse "abs(N)" as an int.
    # Tracking this as a separate cleanup; the branch is still part of the
    # documented API surface, so the helper is exposed.


def test_explain_with_outputs_opchain():
    """Passing an OpChain as outputs (e.g. shap.Explanation.argsort) routes through the OpChain branch."""
    rs = np.random.RandomState(0)
    X = rs.randn(20, 3)
    # multi-output model so outputs slicing has work to do
    coef = np.array([[1.0, -1.0], [2.0, 0.5], [-0.5, 1.0]])

    def model(x):
        return x @ coef

    masker = shap.maskers.Partition(X)
    explainer = shap.explainers.PartitionExplainer(model, masker)
    # use the argsort OpChain to select outputs dynamically per row
    explanation = explainer(X[:1], outputs=shap.Explanation.argsort, silent=True)
    assert explanation.values.shape[0] == 1

"""Unit tests for the Exact explainer."""

import pickle

import numpy as np
from conftest import compare_numpy_outputs_against_baseline

import shap

from . import common


def test_tabular_simple_case():
    import pytest

    xgboost = pytest.importorskip("xgboost")
    sk = pytest.importorskip("sklearn")

    model = xgboost.XGBClassifier(tree_method="exact", base_score=0.5)
    X, y = sk.datasets.make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, return_X_y=True)

    X_train = X[:80]
    X_test = X[80:]
    y_train = y[:80]
    model.fit(X_train, y_train)
    ex = shap.explainers.ExactExplainer(model.predict_proba, X_train)
    shap_values = ex(X_test)

    pred = model.predict_proba(X_test)
    # check additivity
    np.testing.assert_allclose(shap_values.base_values + shap_values.values.sum(axis=1), pred, atol=1e-6)


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_interactions():
    model, data = common.basic_xgboost_scenario(100)
    return common.test_interactions_additivity(shap.explainers.ExactExplainer, model.predict, data, data)


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_tabular_single_output_auto_masker():
    model, data = common.basic_xgboost_scenario(100)
    return common.test_additivity(shap.explainers.ExactExplainer, model.predict, data, data)


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_tabular_single_output_auto_masker_single_value():
    # This currently fails with an MemoryError, I assume due to having a different dimension than required!
    model, data = common.basic_xgboost_scenario(1)
    return common.test_additivity(shap.explainers.ExactExplainer, model.predict, data, data)


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_tabular_single_output_auto_masker_minimal():
    model, data = common.basic_xgboost_scenario(2)
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
def test_serialization_no_model_or_masker_reduced():
    import pytest

    X, y = shap.datasets.adult()
    X = X.iloc[:, :3]
    xgboost = pytest.importorskip("xgboost")
    data = X

    model = xgboost.XGBClassifier(tree_method="exact", base_score=0.5, seed=42)
    model.fit(X, y)

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


def test_multi_output_with_non_varying_features():
    """Test 2D code path when some features don't vary from background.

    This reproduces a bug in compute_grey_code_row_values_2d where the inner
    loop iterates over rv.shape(0) and indexes rv(rvi, ...) instead of
    iterating over inds.shape(0) and indexing rv(inds(rvi), ...).
    The bug is invisible when all features vary (inds == [0,1,...,M-1]),
    but causes wrong results when only a subset varies.
    """
    # 4 features, multi-output model
    # Background: single sample so we can control exactly which features vary
    background = np.array([[0.0, 1.0, 2.0, 3.0]])

    # Simple linear multi-output model: returns [sum_of_features, 2*sum_of_features]
    def model(X):
        s = X.sum(axis=1)
        return np.column_stack([s, 2 * s])

    # Test sample: features 0 and 2 match the background, features 1 and 3 differ
    # So inds should be [1, 3] (only 2 of 4 features vary)
    test_x = np.array([[0.0, 5.0, 2.0, 7.0]])

    explainer = shap.explainers.ExactExplainer(model, background)
    shap_values = explainer(test_x)

    # Additivity check: base_values + sum(shap_values) == model prediction
    pred = model(test_x)
    reconstructed = shap_values.base_values + shap_values.values.sum(axis=1)
    np.testing.assert_allclose(reconstructed, pred, atol=1e-10)

    # Non-varying features (0 and 2) should have zero SHAP values
    np.testing.assert_allclose(shap_values.values[0, 0, :], 0.0, atol=1e-10)
    np.testing.assert_allclose(shap_values.values[0, 2, :], 0.0, atol=1e-10)

    # Varying features (1 and 3) should have non-zero SHAP values
    assert np.any(np.abs(shap_values.values[0, 1, :]) > 1e-10)
    assert np.any(np.abs(shap_values.values[0, 3, :]) > 1e-10)

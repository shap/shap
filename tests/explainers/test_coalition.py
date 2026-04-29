"""This file contains tests for coalition explainer."""

import numpy as np
import pandas as pd
from conftest import compare_numpy_outputs_against_baseline

import shap
from shap.explainers._coalition import create_partition_hierarchy

from . import common
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_tabular_coalition_single_output():
    coalition_tree = {
        "Demographics": ["Sex", "Age", "Race", "Marital Status", "Education-Num"],
        "Work": ["Occupation", "Workclass", "Hours per week"],
        "Finance": ["Capital Gain", "Capital Loss"],
        "Residence": ["Country"],
    }
    model, data = common.basic_xgboost_scenario(100)
    X, _ = shap.datasets.adult()
    features = X.columns.tolist()
    masker = shap.maskers.Partition(data)
    masker.feature_names = features
    return common.test_additivity(
        shap.explainers.CoalitionExplainer, model.predict, masker, data, partition_tree=coalition_tree
    )


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_tabular_coalition_multiple_output():
    coalition_tree = {
        "Demographics": ["Sex", "Age", "Race", "Marital Status", "Education-Num"],
        "Work": ["Occupation", "Workclass", "Hours per week"],
        "Finance": ["Capital Gain", "Capital Loss"],
        "Residence": ["Country"],
    }
    model, data = common.basic_xgboost_scenario(100)
    X, _ = shap.datasets.adult()
    features = X.columns.tolist()
    masker = shap.maskers.Partition(data)
    masker.feature_names = features
    return common.test_additivity(
        shap.explainers.CoalitionExplainer, model.predict_proba, masker, data, partition_tree=coalition_tree
    )


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_tabular_coalition_exact_match():
    model, data = common.basic_xgboost_scenario(50)
    X, _ = shap.datasets.adult()
    features = X.columns.tolist()
    data = pd.DataFrame(data, columns=features)
    exact_explainer = shap.explainers.ExactExplainer(model.predict, data)
    shap_values = exact_explainer(data)

    flat_hierarchy = {}
    for name in features:
        flat_hierarchy[name] = name

    partition_masker = shap.maskers.Partition(data)
    partition_masker.feature_names = features
    partition_explainer_f = shap.CoalitionExplainer(model.predict, partition_masker, partition_tree=flat_hierarchy)
    flat_winter_values = partition_explainer_f(data)
    assert np.allclose(shap_values.values, flat_winter_values.values)
    return shap_values


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_tabular_coalition_partition_match():
    model, data = common.basic_xgboost_scenario(50)
    X, _ = shap.datasets.adult()
    features = X.columns.tolist()
    data = pd.DataFrame(data, columns=features)
    partition_tree = shap.utils.partition_tree(data)
    partition_masker = shap.maskers.Partition(data, clustering=partition_tree)
    partition_masker.feature_names = features
    partition_explainer = shap.explainers.PartitionExplainer(model.predict, partition_masker)
    binary_values = partition_explainer(data)

    hierarchy_binary = create_partition_hierarchy(partition_tree, features)

    coalition_masker = shap.maskers.Partition(data)
    partition_explainer_b = shap.CoalitionExplainer(model.predict, coalition_masker, partition_tree=hierarchy_binary)  # type: ignore[arg-type]
    binary_winter_values = partition_explainer_b(data)

    assert np.allclose(binary_values.values, binary_winter_values.values)  # type: ignore[union-attr]
    return binary_values


def test_coalition_explainer_raises_on_overlapping_features():
    """CoalitionExplainer should raise ValueError for overlapping features."""
    X, y = load_iris(return_X_y=True)
    feature_names = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    masker = shap.maskers.Partition(X)
    masker.feature_names = feature_names

    overlapping_tree = {
        "Sepal": ["sepal length (cm)", "sepal width (cm)"],
        "Petal": ["petal length (cm)", "petal width (cm)"],
        "Extra": ["sepal length (cm)"],  # duplicate
    }

    with pytest.raises(ValueError, match="overlapping features"):
        shap.CoalitionExplainer(model.predict, masker, partition_tree=overlapping_tree)


def test_coalition_explainer_valid_tree_no_error():
    """CoalitionExplainer should not raise for a valid non-overlapping tree."""
    X, y = load_iris(return_X_y=True)
    feature_names = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    masker = shap.maskers.Partition(X)
    masker.feature_names = feature_names

    valid_tree = {
        "Sepal": ["sepal length (cm)", "sepal width (cm)"],
        "Petal": ["petal length (cm)", "petal width (cm)"],
    }

    explainer = shap.CoalitionExplainer(model.predict, masker, partition_tree=valid_tree)
    shap_values = explainer(X[:3]).values
    assert shap_values.shape == (3, 4)

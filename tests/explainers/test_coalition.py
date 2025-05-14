"""This file contains tests for coalition explainer."""

import numpy as np
import pandas as pd

import shap
from shap.explainers._coalition import create_partition_hierarchy

from . import common


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
    common.test_additivity(
        shap.explainers.CoalitionExplainer, model.predict, masker, data, partition_tree=coalition_tree
    )


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
    common.test_additivity(
        shap.explainers.CoalitionExplainer, model.predict_proba, masker, data, partition_tree=coalition_tree
    )


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
    partition_explainer_b = shap.CoalitionExplainer(model.predict, coalition_masker, partition_tree=hierarchy_binary)
    binary_winter_values = partition_explainer_b(data)

    assert np.allclose(binary_values.values, binary_winter_values.values)

"""Tests for categorical splits in the native Tree SHAP implementation."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

import shap
from shap.explainers._tree import SingleTree, TreeEnsemble


def _categorical_training_data(n_categories, *, include_missing=False):
    categories = np.tile(np.arange(n_categories, dtype=np.float64), 12)
    continuous = np.linspace(-1.0, 1.0, categories.size)
    missing = np.zeros(categories.size, dtype=bool)
    if include_missing:
        missing[::13] = True
        categories[missing] = np.nan

    category_codes = np.nan_to_num(categories, nan=0).astype(int)
    target = 3.0 * np.isin(category_codes % 5, [0, 2]) + 0.2 * continuous
    target[missing] = 8.0
    return np.column_stack([categories, continuous]), target


def _train_categorical_booster(lightgbm, n_categories=33, *, include_missing=False, num_boost_round=4):
    X, y = _categorical_training_data(n_categories, include_missing=include_missing)
    dataset = lightgbm.Dataset(X, label=y, categorical_feature=[0])
    model = lightgbm.train(
        {
            "objective": "regression",
            "verbosity": -1,
            "num_threads": 1,
            "seed": 0,
            "max_depth": 3,
            "num_leaves": 8,
            "min_data_in_leaf": 1,
            "min_data_per_group": 1,
            "max_cat_to_onehot": 1,
            "cat_smooth": 0,
            "cat_l2": 0,
            "learning_rate": 0.3,
            "force_col_wise": True,
        },
        dataset,
        num_boost_round=num_boost_round,
    )
    return model, X


def _assert_interaction_additivity(model, X):
    explainer = shap.TreeExplainer(model)
    interactions = explainer.shap_interaction_values(X)
    pred_contrib = model.predict(X, pred_contrib=True)
    predictions = model.predict(X)

    np.testing.assert_allclose(interactions.sum(axis=2), pred_contrib[:, :-1], atol=1e-6, rtol=0)
    np.testing.assert_allclose(
        interactions.sum(axis=(1, 2)) + explainer.expected_value,
        predictions,
        atol=1e-6,
        rtol=0,
    )
    return explainer


def _legacy_categorical_tree(threshold):
    tree = SingleTree(
        {
            "children_left": np.array([1, -1, -1]),
            "children_right": np.array([2, -1, -1]),
            "children_default": np.array([2, -1, -1]),
            "features": np.array([0, -1, -1]),
            "thresholds": np.array([threshold, -1.0, -1.0]),
            "values": np.array([[0.0], [1.0], [-1.0]]),
            "node_sample_weight": np.array([2.0, 1.0, 1.0]),
        }
    )
    tree.threshold_types[0] = 1
    return tree


@pytest.mark.parametrize("n_categories", [8, 33, 300])
def test_lightgbm_native_categorical_interactions(n_categories):
    lightgbm = pytest.importorskip("lightgbm")
    model, X = _train_categorical_booster(lightgbm, n_categories)
    X_test = X[[0, min(32, n_categories - 1), n_categories - 1]]

    assert X_test[0, 0] == 0
    assert any(tree["tree_structure"]["decision_type"] == "==" for tree in model.dump_model()["tree_info"])
    if n_categories == 33:
        assert any("32" in tree["tree_structure"]["threshold"].split("||") for tree in model.dump_model()["tree_info"])

    _assert_interaction_additivity(model, X_test)


def test_lightgbm_native_categorical_missing_routing():
    lightgbm = pytest.importorskip("lightgbm")
    model, X = _train_categorical_booster(lightgbm, include_missing=True)
    X_missing = X[np.isnan(X[:, 0])][:6]

    assert len(X_missing) == 6
    assert np.isnan(X_missing[:, 0]).all()
    _assert_interaction_additivity(model, X_missing)


def test_lightgbm_mixed_categorical_and_continuous_trees():
    lightgbm = pytest.importorskip("lightgbm")
    X, y = _categorical_training_data(33)
    y = y + 2.0 * (X[:, 1] > 0)
    dataset = lightgbm.Dataset(X, label=y, categorical_feature=[0])
    model = lightgbm.train(
        {
            "objective": "regression",
            "verbosity": -1,
            "num_threads": 1,
            "seed": 0,
            "max_depth": 2,
            "num_leaves": 4,
            "min_data_in_leaf": 1,
            "min_data_per_group": 1,
            "max_cat_to_onehot": 1,
            "cat_smooth": 0,
            "cat_l2": 0,
            "learning_rate": 0.3,
            "force_col_wise": True,
        },
        dataset,
        num_boost_round=3,
        callbacks=[lightgbm.reset_parameter(feature_contri=lambda i: [1, 0] if i != 1 else [0, 1])],
    )

    explainer = _assert_interaction_additivity(model, X[[0, 32, 64]])
    trees = explainer.model.trees
    assert trees is not None
    assert [np.any(tree.threshold_types == 1) for tree in trees] == [True, False, True]
    bitset_sizes = [tree.cat_bitsets.size for tree in trees]
    assert bitset_sizes[0] > 0
    assert bitset_sizes[1] == 0
    assert bitset_sizes[2] > 0

    second_tree_cat_nodes = np.flatnonzero(trees[2].threshold_types == 1)
    np.testing.assert_array_equal(
        explainer.model.thresholds[2, second_tree_cat_nodes],
        trees[2].thresholds[second_tree_cat_nodes] + trees[0].cat_bitsets.size,
    )


def test_lightgbm_categorical_saabas_reconstructs_prediction():
    lightgbm = pytest.importorskip("lightgbm")
    model, X = _train_categorical_booster(lightgbm, n_categories=8)
    X_test = X[[0, 7, 15, 31]]
    # Supplying data bypasses LightGBM's native pred_contrib shortcut and exercises
    # SHAP's categorical-aware dense_tree_saabas implementation.
    explainer = shap.TreeExplainer(model, data=X)

    shap_values = explainer.shap_values(X_test, approximate=True)

    np.testing.assert_allclose(
        shap_values.sum(axis=1) + explainer.expected_value,
        model.predict(X_test),
        atol=1e-6,
        rtol=0,
    )


def test_continuous_random_forest_values_and_interactions():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(80, 4))
    y = X[:, 0] ** 2 + X[:, 1] - 0.5 * X[:, 2]
    model = RandomForestRegressor(n_estimators=6, max_depth=3, random_state=0).fit(X, y)
    X_test = X[:8]
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_test)
    interactions = explainer.shap_interaction_values(X_test)

    assert not np.any(explainer.model.threshold_types == 1)
    np.testing.assert_allclose(interactions.sum(axis=2), shap_values, atol=1e-6, rtol=0)
    np.testing.assert_allclose(
        interactions.sum(axis=(1, 2)) + explainer.expected_value,
        model.predict(X_test),
        atol=1e-6,
        rtol=0,
    )


def test_legacy_packed_categorical_mask_migration():
    ensemble = TreeEnsemble([_legacy_categorical_tree(np.int32(-1))], model_output="raw")
    categories = np.arange(34, dtype=np.float64).reshape(-1, 1)

    predictions = ensemble.predict(categories)

    np.testing.assert_array_equal(ensemble.cat_bitsets, np.array([2, 0xFFFFFFFE, 1], dtype=np.uint32))
    np.testing.assert_array_equal(predictions[1:33], np.ones(32))
    np.testing.assert_array_equal(predictions[[0, 33]], -np.ones(2))


def test_invalid_legacy_packed_categorical_mask_raises():
    with pytest.raises(ValueError, match="cannot be represented as a packed 32-bit mask"):
        TreeEnsemble([_legacy_categorical_tree(np.nan)], model_output="raw")


def test_global_path_dependent_rejects_categorical_splits():
    lightgbm = pytest.importorskip("lightgbm")
    model, X = _train_categorical_booster(lightgbm, n_categories=8, num_boost_round=2)
    categorical_explainer = shap.TreeExplainer(model, data=X[:40])
    categorical_explainer.feature_perturbation = "global_path_dependent"

    with pytest.raises(ValueError, match="global_path_dependent.*does not support categorical splits"):
        categorical_explainer.shap_values(X[:2])

    continuous_model = DecisionTreeRegressor(max_depth=1, random_state=0).fit(X[:, 1:], X[:, 1] > 0)
    continuous_explainer = shap.TreeExplainer(continuous_model, data=X[:40, 1:])
    continuous_explainer.feature_perturbation = "global_path_dependent"
    # The constructor no longer exposes this legacy mode, so discard the expected value
    # computed for its initial interventional mode and let the native call set the right one.
    continuous_explainer.expected_value = None

    shap_values = continuous_explainer.shap_values(X[:4, 1:])
    np.testing.assert_allclose(
        shap_values.sum(axis=1) + continuous_explainer.expected_value,
        continuous_model.predict(X[:4, 1:]),
        atol=1e-6,
        rtol=0,
    )


def test_malformed_categorical_bitset_offset_routes_right():
    ensemble = TreeEnsemble([_legacy_categorical_tree(1)], model_output="raw")
    ensemble.thresholds[0, 0] = ensemble.cat_bitsets.size + 10
    X = np.array([[1.0]])

    np.testing.assert_array_equal(ensemble.predict(X), np.array([-1.0]))

    explainer = shap.TreeExplainer(ensemble)
    shap_values = explainer.shap_values(X)
    np.testing.assert_allclose(shap_values.sum() + explainer.expected_value, -1.0, atol=1e-6, rtol=0)

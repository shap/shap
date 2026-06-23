import sys
import types

import numpy as np
import pandas as pd
import pytest

import shap.explainers.pytree as pytree
from shap.utils._exceptions import ExplainerError


class _FakeTensor:
    def __init__(self, data):
        self.data = np.array(data)

    @property
    def shape(self):
        return self.data.shape

    def to(self, device):
        return self

    def long(self):
        return _FakeTensor(self.data.astype(np.int64))

    def cumsum(self, axis):
        return _FakeTensor(np.cumsum(self.data, axis=axis))

    def masked_fill_(self, mask, value):
        mask_data = mask.data if isinstance(mask, _FakeTensor) else mask
        self.data = np.where(mask_data, value, self.data)
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.data

    def astype(self, dtype):
        return self.data.astype(dtype)

    def __getitem__(self, item):
        return _FakeTensor(self.data[item])

    def __setitem__(self, key, value):
        self.data[key] = value.data if isinstance(value, _FakeTensor) else value

    def __eq__(self, other):
        other_data = other.data if isinstance(other, _FakeTensor) else other
        return _FakeTensor(self.data == other_data)

    def __sub__(self, other):
        other_data = other.data if isinstance(other, _FakeTensor) else other
        return _FakeTensor(self.data - other_data)

    def __rsub__(self, other):
        other_data = other.data if isinstance(other, _FakeTensor) else other
        return _FakeTensor(other_data - self.data)

    def __array__(self, dtype=None):
        return np.asarray(self.data, dtype=dtype)


class _FakeEstimator:
    def __init__(self, tree):
        self.tree_ = tree


class _FakeSklearnTree:
    pass


_FakeSklearnTree.__module__ = "sklearn.tree._tree"
_FakeSklearnTree.__qualname__ = "Tree"


class _FakeRFRegressor:
    def __init__(self, estimators_):
        self.estimators_ = estimators_


_FakeRFRegressor.__module__ = "sklearn.ensemble.forest"
_FakeRFRegressor.__qualname__ = "RandomForestRegressor"


class _FakeRFClassifier:
    def __init__(self, estimators_):
        self.estimators_ = estimators_


_FakeRFClassifier.__module__ = "sklearn.ensemble.forest"
_FakeRFClassifier.__qualname__ = "RandomForestClassifier"


class _FakeXGBDMatrix:
    def __init__(self, data):
        self.data = np.array(data)


_FakeXGBDMatrix.__module__ = "xgboost.core"
_FakeXGBDMatrix.__qualname__ = "DMatrix"


class _FakeXGBBooster:
    def __init__(self):
        self.calls = []

    def predict(self, X, ntree_limit=None, pred_contribs=False, pred_interactions=False):
        self.calls.append(
            {
                "X": X,
                "ntree_limit": ntree_limit,
                "pred_contribs": pred_contribs,
                "pred_interactions": pred_interactions,
            }
        )
        if pred_interactions:
            return np.array([[[1.0, 2.0], [3.0, 4.0]]])
        if pred_contribs:
            return np.array([[0.1, 0.2, 0.3]])
        return np.array([0.5])


_FakeXGBBooster.__module__ = "xgboost.core"
_FakeXGBBooster.__qualname__ = "Booster"


class _FakeLGBMBooster:
    def __init__(self):
        self.calls = []

    def predict(self, X, num_iteration=None, pred_contrib=False):
        self.calls.append({"X": X, "num_iteration": num_iteration, "pred_contrib": pred_contrib})
        return np.array([[0.4, 0.6]])


_FakeLGBMBooster.__module__ = "lightgbm.basic"
_FakeLGBMBooster.__qualname__ = "Booster"


def _make_fake_sklearn_tree(
    values,
    children_left,
    children_right,
    features,
    thresholds,
    weighted_n_node_samples,
    missing_go_to_left=None,
):
    tree = _FakeSklearnTree()
    tree.children_left = np.array(children_left, dtype=np.int32)
    tree.children_right = np.array(children_right, dtype=np.int32)
    tree.feature = np.array(features, dtype=np.int32)
    tree.threshold = np.array(thresholds, dtype=np.float64)
    tree.value = np.array(values, dtype=np.float64)
    tree.weighted_n_node_samples = np.array(weighted_n_node_samples, dtype=np.float64)
    if missing_go_to_left is not None:
        tree.missing_go_to_left = np.array(missing_go_to_left)
    return tree


def _make_internal_regressor_tree():
    return _make_fake_sklearn_tree(
        values=[[[0.0]], [[1.0]], [[3.0]]],
        children_left=[1, -1, -1],
        children_right=[2, -1, -1],
        features=[0, -2, -2],
        thresholds=[0.5, -2.0, -2.0],
        weighted_n_node_samples=[10.0, 4.0, 6.0],
        missing_go_to_left=[True, False, False],
    )


def _make_internal_classifier_tree():
    return _make_fake_sklearn_tree(
        values=[[[2.0, 6.0]], [[1.0, 3.0]], [[4.0, 2.0]]],
        children_left=[1, -1, -1],
        children_right=[2, -1, -1],
        features=[0, -2, -2],
        thresholds=[0.5, -2.0, -2.0],
        weighted_n_node_samples=[10.0, 4.0, 6.0],
    )


def _fake_xgboost_module():
    module = types.ModuleType("xgboost")
    module.DMatrix = _FakeXGBDMatrix
    return module


def test_tree_wrapper_and_expectations():
    tree = _make_internal_regressor_tree()

    wrapped = pytree.Tree(tree)

    assert wrapped.max_depth == 1
    assert wrapped.children_default.tolist() == [1, -1, -1]
    np.testing.assert_allclose(wrapped.values[0, 0], 2.2)


def test_tree_wrapper_normalizes_classifier_values():
    tree = _make_internal_classifier_tree()

    wrapped = pytree.Tree(tree, normalize=True)

    np.testing.assert_allclose(wrapped.values.sum(axis=1), 1.0)


def test_path_helpers_nonzero_branch():
    feature_indexes = np.zeros(6, dtype=np.int32)
    zero_fractions = np.zeros(6, dtype=np.float64)
    one_fractions = np.zeros(6, dtype=np.float64)
    pweights = np.zeros(6, dtype=np.float64)

    pytree.extend_path(feature_indexes, zero_fractions, one_fractions, pweights, 0, 0.25, 0.75, 3)
    pytree.extend_path(feature_indexes, zero_fractions, one_fractions, pweights, 1, 0.5, 0.5, 4)

    assert feature_indexes[:2].tolist() == [3, 4]
    assert pweights[0] > 0

    total = pytree.unwound_path_sum(feature_indexes, zero_fractions, one_fractions, pweights, 1, 0)
    assert total > 0


def test_path_helpers_zero_fraction_branch():
    feature_indexes = np.array([5, 6, 7, 0], dtype=np.int32)
    zero_fractions = np.array([0.4, 0.6, 0.2, 0.0], dtype=np.float64)
    one_fractions = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    pweights = np.array([0.2, 0.3, 0.4, 0.5], dtype=np.float64)

    pytree.unwind_path(feature_indexes, zero_fractions, one_fractions, pweights, 2, 1)
    total = pytree.unwound_path_sum(feature_indexes, zero_fractions, one_fractions, pweights, 2, 1)

    assert total > 0


def test_tree_shap_recursive_leaf_missing_and_condition_branches():
    children_left = np.array([1, -1, -1], dtype=np.int32)
    children_right = np.array([2, -1, -1], dtype=np.int32)
    children_default = np.array([1, -1, -1], dtype=np.int32)
    features = np.array([0, -1, -1], dtype=np.int32)
    thresholds = np.array([0.5, 0.0, 0.0], dtype=np.float64)
    values = np.array([[0.0], [1.0], [3.0]], dtype=np.float64)
    weights = np.array([10.0, 4.0, 6.0], dtype=np.float64)
    x = np.array([0.1], dtype=np.float64)
    phi = np.zeros((2, 1), dtype=np.float64)
    parent_feature_indexes = np.zeros(8, dtype=np.int32)
    parent_zero_fractions = np.zeros(8, dtype=np.float64)
    parent_one_fractions = np.zeros(8, dtype=np.float64)
    parent_pweights = np.zeros(8, dtype=np.float64)

    pytree.tree_shap_recursive(
        children_left,
        children_right,
        children_default,
        features,
        thresholds,
        values,
        weights,
        x,
        np.array([0], dtype=np.int32),
        phi,
        0,
        0,
        parent_feature_indexes,
        parent_zero_fractions,
        parent_one_fractions,
        parent_pweights,
        1.0,
        1.0,
        -1,
        0,
        0,
        1.0,
    )

    pytree.tree_shap_recursive(
        children_left,
        children_right,
        children_default,
        features,
        thresholds,
        values,
        weights,
        x,
        np.array([1], dtype=np.int32),
        phi,
        0,
        0,
        parent_feature_indexes.copy(),
        parent_zero_fractions.copy(),
        parent_one_fractions.copy(),
        parent_pweights.copy(),
        1.0,
        1.0,
        -1,
        1,
        0,
        1.0,
    )

    pytree.tree_shap_recursive(
        children_left,
        children_right,
        children_default,
        features,
        thresholds,
        values,
        weights,
        x,
        np.array([1], dtype=np.int32),
        phi,
        0,
        0,
        parent_feature_indexes.copy(),
        parent_zero_fractions.copy(),
        parent_one_fractions.copy(),
        parent_pweights.copy(),
        1.0,
        1.0,
        -1,
        -1,
        0,
        1.0,
    )

    assert phi[0, 0] != 0


def test_tree_shap_recursive_repeated_feature_unwinds_path():
    children_left = np.array([1, -1, 3, -1, -1], dtype=np.int32)
    children_right = np.array([2, -1, 4, -1, -1], dtype=np.int32)
    children_default = np.array([1, -1, 3, -1, -1], dtype=np.int32)
    features = np.array([0, -1, 0, -1, -1], dtype=np.int32)
    thresholds = np.array([0.5, 0.0, 0.5, 0.0, 0.0], dtype=np.float64)
    values = np.array([[0.0], [1.0], [0.0], [2.0], [3.0]], dtype=np.float64)
    weights = np.array([10.0, 4.0, 6.0, 2.0, 4.0], dtype=np.float64)
    x = np.array([0.9], dtype=np.float64)
    phi = np.zeros((2, 1), dtype=np.float64)
    parent_feature_indexes = np.zeros(16, dtype=np.int32)
    parent_zero_fractions = np.zeros(16, dtype=np.float64)
    parent_one_fractions = np.zeros(16, dtype=np.float64)
    parent_pweights = np.zeros(16, dtype=np.float64)

    pytree.tree_shap_recursive(
        children_left,
        children_right,
        children_default,
        features,
        thresholds,
        values,
        weights,
        x,
        np.array([0], dtype=np.int32),
        phi,
        0,
        0,
        parent_feature_indexes,
        parent_zero_fractions,
        parent_one_fractions,
        parent_pweights,
        1.0,
        1.0,
        -1,
        0,
        0,
        1.0,
    )

    assert phi[0, 0] != 0


def test_tree_explainer_internal_and_shap_values(monkeypatch):
    model = _FakeRFRegressor([_FakeEstimator(_make_internal_regressor_tree())])
    explainer = pytree.TreeExplainer(model)

    assert explainer.model_type == "internal"
    assert hasattr(explainer, "feature_indexes")

    monkeypatch.setattr(explainer, "tree_shap", lambda *args, **kwargs: None)

    original_zeros = pytree.np.zeros

    def zeros(shape, dtype=float, order="C"):
        if isinstance(shape, (int, np.integer)) and isinstance(dtype, (int, np.integer)):
            return original_zeros((shape, dtype), order=order)
        return original_zeros(shape, dtype=dtype, order=order)

    monkeypatch.setattr(pytree.np, "zeros", zeros)

    df_X = pd.DataFrame([[0.1], [0.9]], columns=["feature_0"])
    shap_values_2d = explainer.shap_values(df_X)
    shap_values_1d = explainer.shap_values(np.array([0.1], dtype=np.float64))

    assert shap_values_2d.shape == (2, 2)
    assert shap_values_1d.shape == (2,)


def test_tree_explainer_classifier_branch_and_multioutput_returns(monkeypatch):
    model = _FakeRFClassifier([_FakeEstimator(_make_internal_classifier_tree())])
    explainer = pytree.TreeExplainer(model)

    assert explainer.model_type == "internal"

    monkeypatch.setattr(explainer, "tree_shap", lambda *args, **kwargs: None)

    original_zeros = pytree.np.zeros

    def zeros(shape, dtype=float, order="C"):
        if isinstance(shape, (int, np.integer)) and isinstance(dtype, (int, np.integer)):
            return original_zeros((shape, dtype), order=order)
        return original_zeros(shape, dtype=dtype, order=order)

    monkeypatch.setattr(pytree.np, "zeros", zeros)

    df_X = pd.DataFrame([[0.1], [0.9]], columns=["feature_0"])
    shap_values_2d = explainer.shap_values(df_X)
    shap_values_1d = explainer.shap_values(np.array([0.1], dtype=np.float64))

    assert isinstance(shap_values_2d, list)
    assert len(shap_values_2d) == 2
    assert all(value.shape == (2, 2) for value in shap_values_2d)
    assert isinstance(shap_values_1d, list)
    assert len(shap_values_1d) == 2
    assert all(value.shape == (2,) for value in shap_values_1d)


def test_tree_shap_method_updates_bias_term(monkeypatch):
    model = _FakeRFRegressor([_FakeEstimator(_make_internal_regressor_tree())])
    explainer = pytree.TreeExplainer(model)
    tree = explainer.trees[0]
    phi = np.zeros((2, 1), dtype=np.float64)

    monkeypatch.setattr(pytree, "tree_shap_recursive", lambda *args, **kwargs: None)

    explainer.tree_shap(tree, np.array([0.1], dtype=np.float64), np.array([0], dtype=np.int32), phi)

    assert phi[-1, 0] == tree.values[0, 0]


def test_tree_explainer_xgboost_and_lightgbm_shortcuts(monkeypatch):
    xgb_module = _fake_xgboost_module()
    monkeypatch.setitem(sys.modules, "xgboost", xgb_module)

    xgb_model = _FakeXGBBooster()
    xgb_explainer = pytree.TreeExplainer(xgb_model)
    xgb_values = xgb_explainer.shap_values(np.array([[1.0, 2.0]], dtype=np.float64))
    xgb_interactions = xgb_explainer.shap_interaction_values(np.array([[1.0, 2.0]], dtype=np.float64))

    assert xgb_explainer.model_type == "xgboost"
    assert isinstance(xgb_values, np.ndarray)
    assert xgb_values.shape == (1, 3)
    assert xgb_interactions.shape == (1, 2, 2)
    assert isinstance(xgb_model.calls[0]["X"], _FakeXGBDMatrix)
    assert xgb_model.calls[0]["ntree_limit"] == 0
    assert xgb_model.calls[0]["pred_contribs"] is True
    assert xgb_model.calls[1]["pred_interactions"] is True

    lgbm_model = _FakeLGBMBooster()
    lgbm_explainer = pytree.TreeExplainer(lgbm_model)
    lgbm_values = lgbm_explainer.shap_values(np.array([[1.0, 2.0]], dtype=np.float64))

    assert lgbm_explainer.model_type == "lightgbm"
    assert isinstance(lgbm_values, np.ndarray)
    assert lgbm_values.shape == (1, 2)
    assert lgbm_model.calls[0]["num_iteration"] == -1
    assert lgbm_model.calls[0]["pred_contrib"] is True


def test_tree_explainer_unsupported_model_and_interactions_error():
    with pytest.raises(ExplainerError):
        pytree.TreeExplainer(object())

    model = _FakeRFRegressor([_FakeEstimator(_make_internal_regressor_tree())])
    explainer = pytree.TreeExplainer(model)

    with pytest.raises(NotImplementedError):
        explainer.shap_interaction_values(np.array([[0.1]], dtype=np.float64))

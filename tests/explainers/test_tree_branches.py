from __future__ import annotations

import builtins
import importlib.util
import json
import sys
import types

import numpy as np
import pandas as pd
import pytest
import sklearn.ensemble
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

from shap.explainers import _tree as tree_module
from shap.utils._exceptions import DimensionError, ExplainerError, InvalidMaskerError, InvalidModelError
from shap.utils._legacy import DenseData
from shap.utils._warnings import ExperimentalWarning


def _make_explainer_for_validate_inputs(
    *,
    model_output: str = "raw",
    model_type: str = "internal",
    fully_defined_weighting: bool = True,
):
    explainer = tree_module.TreeExplainer.__new__(tree_module.TreeExplainer)
    explainer.model = types.SimpleNamespace(
        tree_limit=None,
        values=np.zeros((3, 1, 1), dtype=float),
        input_dtype=np.float64,
        model_output=model_output,
        fully_defined_weighting=fully_defined_weighting,
        model_type=model_type,
    )
    explainer.feature_perturbation = "interventional"
    explainer.data = None
    explainer.expected_value = None
    return explainer


def _make_explainer_for_fastpath(model_type: str, original_model, *, cat_feature_indices=None):
    explainer = tree_module.TreeExplainer.__new__(tree_module.TreeExplainer)
    explainer.feature_perturbation = "tree_path_dependent"
    explainer.data = None
    explainer.expected_value = None
    explainer.model = types.SimpleNamespace(
        tree_limit=None,
        model_type=model_type,
        original_model=original_model,
        model_output="raw",
        num_stacked_models=1,
        cat_feature_indices=cat_feature_indices,
    )
    return explainer


def test_safe_check_tree_instance_experimental_warns_for_supported_library():
    CausalTreeLike = type("CausalTreeLike", (), {})
    CausalTreeLike.__module__ = "causalml.inference.tree"

    with pytest.warns(ExperimentalWarning, match="experimental integration with causalml"):
        tree_module._safe_check_tree_instance_experimental(CausalTreeLike())


def test_safe_check_tree_instance_experimental_warns_when_uninspectable():
    class Uninspectable:
        def __getattribute__(self, name):
            if name == "__class__":
                raise AttributeError("hidden class")
            return super().__getattribute__(name)

    with pytest.warns(ExperimentalWarning, match="Unable to check experimental integration status"):
        tree_module._safe_check_tree_instance_experimental(Uninspectable())


def test_xgboost_helpers_cover_iteration_and_categorical_guard():
    assert tree_module._xgboost_n_iterations(-1, 4) == 0
    assert tree_module._xgboost_n_iterations(10, 3) == 3

    with pytest.raises(NotImplementedError, match="Categorical split is not yet supported"):
        tree_module._xgboost_cat_unsupported(
            types.SimpleNamespace(model_type="xgboost", cat_feature_indices=np.array([0]))
        )

    tree_module._xgboost_cat_unsupported(
        types.SimpleNamespace(model_type="internal", cat_feature_indices=np.array([0]))
    )


def test_validate_inputs_requires_y_for_log_loss():
    explainer = _make_explainer_for_validate_inputs(model_output="log_loss")
    X = np.array([[1.0, 2.0]], dtype=np.float64)

    with pytest.raises(ExplainerError, match='Both samples and labels must be provided when model_output = "log_loss"'):
        explainer._validate_inputs(X, y=None, tree_limit=None, check_additivity=True)


def test_validate_inputs_checks_label_count_for_log_loss():
    explainer = _make_explainer_for_validate_inputs(model_output="log_loss")
    X = np.array([[1.0, 2.0]], dtype=np.float64)

    with pytest.raises(DimensionError, match="does not match the number of samples to explain"):
        explainer._validate_inputs(X, y=np.array([0, 1]), tree_limit=None, check_additivity=True)


def test_validate_inputs_requires_fully_defined_weighting_for_tree_path_dependent():
    explainer = _make_explainer_for_validate_inputs(fully_defined_weighting=False)
    explainer.feature_perturbation = "tree_path_dependent"
    X = np.array([[1.0, 2.0]], dtype=np.float64)

    with pytest.raises(ExplainerError, match="does not cover all the leaves"):
        explainer._validate_inputs(X, y=None, tree_limit=None, check_additivity=True)


def test_validate_inputs_disables_additivity_for_pyspark_with_warning():
    explainer = _make_explainer_for_validate_inputs(model_type="pyspark")
    X = np.array([[1.0, 2.0]], dtype=np.float64)

    with pytest.warns(UserWarning, match="check_additivity requires us to run predictions"):
        _X, _y, _missing, _flat, _tree_limit, check_additivity = explainer._validate_inputs(
            X, y=None, tree_limit=None, check_additivity=True
        )

    assert check_additivity is False


def test_get_xgboost_dmatrix_properties_converts_n_jobs_to_nthread():
    model = types.SimpleNamespace(
        missing=np.nan,
        n_jobs=8,
        enable_categorical=True,
        feature_types=["q", "c"],
    )

    out = tree_module.get_xgboost_dmatrix_properties(model)

    assert out["nthread"] == 8
    assert "n_jobs" not in out
    assert out["enable_categorical"] is True
    assert out["feature_types"] == ["q", "c"]


class _FakeLightGBMModel:
    def __init__(self, phi: np.ndarray, objective: str = "binary"):
        self._phi = phi
        self.params = {"objective": objective}

    def predict(self, X, num_iteration=None, pred_contrib=False):
        return self._phi


def test_shap_values_lightgbm_fastpath_warns_and_reshapes_output():
    X = np.arange(6, dtype=float).reshape(2, 3)
    phi = np.arange(16, dtype=float).reshape(2, 8)
    explainer = _make_explainer_for_fastpath("lightgbm", _FakeLightGBMModel(phi, objective="binary"))

    with pytest.warns(UserWarning, match="LightGBM binary classifier"):
        out = explainer.shap_values(X)

    assert out.shape == (2, 3, 2)
    assert np.allclose(explainer.expected_value, [3.0, 7.0])


def test_shap_values_lightgbm_fastpath_raises_with_bad_phi_shape():
    X = np.arange(6, dtype=float).reshape(2, 3)
    phi = np.arange(14, dtype=float).reshape(2, 7)
    explainer = _make_explainer_for_fastpath("lightgbm", _FakeLightGBMModel(phi, objective="regression"))

    with pytest.raises(ValueError, match="This reshape error is often caused by passing a bad data matrix"):
        explainer.shap_values(X)


def test_shap_values_catboost_fastpath_with_pool_conversion(monkeypatch):
    class FakePool:
        def __init__(self, data, cat_features=None):
            self.data = np.asarray(data)
            self.cat_features = cat_features

    class FakeCatBoostModel:
        def __init__(self, phi):
            self._phi = phi
            self.seen_data = None

        def get_feature_importance(self, data, fstr_type):
            self.seen_data = data
            assert fstr_type == "ShapValues"
            return self._phi

    fake_catboost = types.SimpleNamespace(Pool=FakePool)
    monkeypatch.setitem(sys.modules, "catboost", fake_catboost)

    X = np.arange(6, dtype=float).reshape(2, 3)
    phi = np.arange(8, dtype=float).reshape(2, 4)
    original_model = FakeCatBoostModel(phi)
    explainer = _make_explainer_for_fastpath("catboost", original_model, cat_feature_indices=[1])

    out = explainer.shap_values(X)

    assert isinstance(original_model.seen_data, FakePool)
    assert original_model.seen_data.cat_features == [1]
    assert out.shape == (2, 3)
    assert explainer.expected_value == 3.0


def test_xgb_loader_parse_categories_handles_empty_and_nonempty_nodes():
    left_children = np.array([1, 2, 3, 4, 5], dtype=np.int32)

    parsed = tree_module.XGBTreeModelLoader.parse_categories(
        cat_nodes=[1, 3],
        cat_segments=[0, 2],
        cat_sizes=[2, 1],
        cats=[4, 5, 7],
        left_children=left_children,
    )
    assert parsed == [[], [4, 5], [], [7], []]

    parsed_empty = tree_module.XGBTreeModelLoader.parse_categories(
        cat_nodes=[],
        cat_segments=[],
        cat_sizes=[],
        cats=[],
        left_children=left_children,
    )
    assert parsed_empty == [[], [], [], [], []]


def test_xgb_loader_get_trees_and_print_info(capsys):
    loader = tree_module.XGBTreeModelLoader.__new__(tree_module.XGBTreeModelLoader)
    loader.num_trees = 1
    loader.node_cleft = [np.array([1, -1, -1], dtype=np.int32)]
    loader.node_cright = [np.array([2, -1, -1], dtype=np.int32)]
    loader.children_default = [np.array([1, -1, -1], dtype=np.int64)]
    loader.features = [np.array([0, -1, -1], dtype=np.int64)]
    loader.thresholds = [np.array([0.5, 0.0, 0.0], dtype=np.float32)]
    loader.threshold_types = [np.array([0, 0, 0], dtype=np.int32)]
    loader.values = [np.array([[0.0], [1.0], [-1.0]], dtype=np.float64)]
    loader.sum_hess = [np.array([2.0, 1.0, 1.0], dtype=np.float64)]
    loader.base_score = 0.25
    loader.num_feature = 1
    loader.num_class = 1
    loader.name_obj = "reg:squarederror"
    loader.name_gbm = "gbtree"

    trees = loader.get_trees()
    assert len(trees) == 1
    assert isinstance(trees[0], tree_module.SingleTree)

    loader.print_info()
    stdout = capsys.readouterr().out
    assert "--- global parameters ---" in stdout
    assert "base_score = 0.25" in stdout


def test_catboost_tree_model_loader_loads_json_and_builds_tree():
    class FakeCatBoostModel:
        def __init__(self, payload):
            self.payload = payload

        def save_model(self, path, format):
            assert format == "json"
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(self.payload, fh)

    payload = {
        "oblivious_trees": [
            {
                "leaf_weights": [1, 2, 3, 4, 5, 6, 7, 8],
                "leaf_values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                "splits": [
                    {"split_type": "FloatFeature", "float_feature_index": 1, "border": 0.5},
                    {"split_type": "OneHotFeature", "cat_feature_index": 2, "value": 1.0},
                    {"split_type": "CtrFeature", "ctr_target_border_idx": 3, "border": 2.5},
                ],
            }
        ],
        "model_info": {"params": {"tree_learner_options": {"depth": 3}}},
    }

    loader = tree_module.CatBoostTreeModelLoader(FakeCatBoostModel(payload))
    trees = loader.get_trees()

    assert loader.num_trees == 1
    assert loader.max_depth == 3
    assert len(trees) == 1
    assert trees[0].features.shape[0] == 15
    assert trees[0].thresholds.shape[0] == 15


def test_tree_module_import_records_missing_cext_and_pyspark(monkeypatch):
    import shap
    from shap.utils import _general as general_utils

    module_name = "shap.explainers._tree_import_fallback"
    real_import = builtins.__import__
    previous_import_errors = dict(general_utils.import_errors)

    previous_cext_module = sys.modules.pop("shap._cext", None)
    had_cext_attr = hasattr(shap, "_cext")
    previous_cext_attr = getattr(shap, "_cext", None)
    if had_cext_attr:
        delattr(shap, "_cext")

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.endswith("._cext") or (fromlist and "_cext" in fromlist):
            raise ImportError("forced missing dependency: shap._cext")
        if name == "pyspark":
            raise ImportError(f"forced missing dependency: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    spec = importlib.util.spec_from_file_location(module_name, tree_module.__file__)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        for dep in ("cext", "pyspark"):
            try:
                module.assert_import(dep)
            except ImportError:
                pass
    finally:
        sys.modules.pop(module_name, None)
        general_utils.import_errors = previous_import_errors
        if previous_cext_module is not None:
            sys.modules["shap._cext"] = previous_cext_module
        if had_cext_attr:
            setattr(shap, "_cext", previous_cext_attr)


def test_treeexplainer_call_stacks_list_outputs_and_tiling_paths():
    explainer = tree_module.TreeExplainer.__new__(tree_module.TreeExplainer)
    explainer.expected_value = [0.1, 0.2]
    explainer.data_feature_names = ["f0", "f1"]
    explainer.shap_values = lambda X, **kwargs: [np.ones((2, 2)), np.zeros((2, 2))]
    explainer.shap_interaction_values = lambda X: [np.ones((2, 2, 2)), np.zeros((2, 2, 2))]

    out_regular = explainer(np.array([[1.0, 2.0], [3.0, 4.0]]), interactions=False)
    assert out_regular.values.shape == (2, 2, 2)
    assert out_regular.base_values.shape == (2, 2)

    out_interactions = explainer(np.array([[1.0, 2.0], [3.0, 4.0]]), interactions=True)
    assert isinstance(out_interactions.values, list)
    assert len(out_interactions.values) == 2
    assert out_interactions.base_values.shape == (2, 2)


def test_treeexplainer_call_rejects_approximate_interactions():
    explainer = tree_module.TreeExplainer.__new__(tree_module.TreeExplainer)
    explainer.expected_value = 0.0

    with pytest.raises(NotImplementedError, match="Approximate computation not yet supported for interaction effects"):
        explainer(np.array([[1.0, 2.0]]), interactions=True, approximate=True)


def test_get_shap_output_sets_expected_value_for_single_and_multi_output():
    explainer_single = tree_module.TreeExplainer.__new__(tree_module.TreeExplainer)
    explainer_single.model = types.SimpleNamespace(num_outputs=1, model_output="raw")
    explainer_single.expected_value = None
    phi_single = np.array([[[1.0], [2.0], [3.0]]])

    out_single = explainer_single._get_shap_output(phi_single, flat_output=False)
    assert explainer_single.expected_value == 3.0
    np.testing.assert_allclose(out_single, np.array([[1.0, 2.0]]))

    explainer_multi = tree_module.TreeExplainer.__new__(tree_module.TreeExplainer)
    explainer_multi.model = types.SimpleNamespace(num_outputs=2, model_output="raw")
    explainer_multi.expected_value = None
    phi_multi = np.array([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]])

    out_multi = explainer_multi._get_shap_output(phi_multi, flat_output=False)
    assert explainer_multi.expected_value == [3.0, 30.0]
    assert isinstance(out_multi, list)
    assert len(out_multi) == 2


def test_shap_interaction_values_fastpaths_xgboost_and_catboost(monkeypatch):
    class FakeDMatrix:
        def __init__(self, data):
            self.data = np.asarray(data)

    fake_xgb = types.SimpleNamespace(
        core=types.SimpleNamespace(DMatrix=FakeDMatrix),
        DMatrix=FakeDMatrix,
    )
    monkeypatch.setitem(sys.modules, "xgboost", fake_xgb)

    class FakeXGBModel:
        def predict(self, X, iteration_range=None, pred_interactions=False, validate_features=False):
            assert pred_interactions is True
            return np.arange(9, dtype=float).reshape(1, 3, 3)

    explainer_xgb = tree_module.TreeExplainer.__new__(tree_module.TreeExplainer)
    explainer_xgb.model = types.SimpleNamespace(
        model_output="raw",
        model_type="xgboost",
        tree_limit=None,
        num_stacked_models=1,
        original_model=FakeXGBModel(),
    )
    explainer_xgb.feature_perturbation = "tree_path_dependent"

    out_xgb = explainer_xgb.shap_interaction_values(np.array([[1.0, 2.0]]))
    assert out_xgb.shape == (1, 2, 2)
    assert explainer_xgb.expected_value == 8.0

    class FakePool:
        def __init__(self, data, cat_features=None):
            self.data = np.asarray(data)
            self.cat_features = cat_features

    class FakeCatModel:
        def __init__(self, phi):
            self._phi = phi

        def get_feature_importance(self, data, fstr_type):
            assert fstr_type == "ShapInteractionValues"
            return self._phi

    fake_catboost = types.SimpleNamespace(Pool=FakePool)
    monkeypatch.setitem(sys.modules, "catboost", fake_catboost)

    explainer_cat_4d = tree_module.TreeExplainer.__new__(tree_module.TreeExplainer)
    explainer_cat_4d.model = types.SimpleNamespace(
        model_output="raw",
        model_type="catboost",
        tree_limit=None,
        cat_feature_indices=[0],
        original_model=FakeCatModel(np.arange(18, dtype=float).reshape(1, 2, 3, 3)),
    )
    explainer_cat_4d.feature_perturbation = "tree_path_dependent"

    out_cat_4d = explainer_cat_4d.shap_interaction_values(np.array([[1.0, 2.0]]))
    assert isinstance(out_cat_4d, list)
    assert len(out_cat_4d) == 2

    explainer_cat_3d = tree_module.TreeExplainer.__new__(tree_module.TreeExplainer)
    explainer_cat_3d.model = types.SimpleNamespace(
        model_output="raw",
        model_type="catboost",
        tree_limit=None,
        cat_feature_indices=[0],
        original_model=FakeCatModel(np.arange(9, dtype=float).reshape(1, 3, 3)),
    )
    explainer_cat_3d.feature_perturbation = "tree_path_dependent"

    out_cat_3d = explainer_cat_3d.shap_interaction_values(np.array([[1.0, 2.0]]))
    assert out_cat_3d.shape == (1, 2, 2)


def test_assert_additivity_and_supports_model_with_masker(monkeypatch):
    explainer = tree_module.TreeExplainer.__new__(tree_module.TreeExplainer)
    explainer.expected_value = 0.0
    explainer.feature_perturbation = "tree_path_dependent"

    with pytest.raises(ExplainerError, match="Additivity check failed in TreeExplainer"):
        explainer.assert_additivity(np.array([[0.0, 0.0]]), np.array([10.0]))

    assert tree_module.TreeExplainer.supports_model_with_masker(object(), masker=object()) is False

    monkeypatch.setattr(tree_module, "TreeEnsemble", lambda model: object())
    assert tree_module.TreeExplainer.supports_model_with_masker(object(), masker=None) is True


def test_treeensemble_num_outputs_get_transform_and_predict_errors(monkeypatch):
    ensemble_bad_count = tree_module.TreeEnsemble.__new__(tree_module.TreeEnsemble)
    ensemble_bad_count.model_type = "internal"
    ensemble_bad_count.num_stacked_models = 2
    ensemble_bad_count.trees = [
        types.SimpleNamespace(values=np.zeros((1, 1))),
        types.SimpleNamespace(values=np.zeros((1, 1))),
        types.SimpleNamespace(values=np.zeros((1, 1))),
    ]

    with pytest.raises(ValueError, match="equal numbers of trees"):
        _ = ensemble_bad_count.num_outputs

    ensemble_bad_width = tree_module.TreeEnsemble.__new__(tree_module.TreeEnsemble)
    ensemble_bad_width.model_type = "internal"
    ensemble_bad_width.num_stacked_models = 2
    ensemble_bad_width.trees = [
        types.SimpleNamespace(values=np.zeros((1, 2))),
        types.SimpleNamespace(values=np.zeros((1, 2))),
    ]

    with pytest.raises(ValueError, match="single outputs per model"):
        _ = ensemble_bad_width.num_outputs

    ensemble_transform = tree_module.TreeEnsemble.__new__(tree_module.TreeEnsemble)
    ensemble_transform.model_output = "probability"
    ensemble_transform.tree_output = "raw_value"

    with pytest.raises(NotImplementedError, match='model_output = "probability" is not yet supported'):
        ensemble_transform.get_transform()

    ensemble_transform.model_output = "log_loss"
    ensemble_transform.objective = "unsupported"
    with pytest.raises(NotImplementedError, match='model_output = "log_loss" is not yet supported'):
        ensemble_transform.get_transform()

    ensemble_transform.model_output = "predict_fn"
    with pytest.raises(ValueError, match="Unrecognized model_output parameter value"):
        ensemble_transform.get_transform()

    ensemble_predict = tree_module.TreeEnsemble.__new__(tree_module.TreeEnsemble)
    ensemble_predict.model_type = "pyspark"
    ensemble_predict.model_output = "raw"

    with pytest.raises(NotImplementedError, match="Predict with pyspark isn't implemented"):
        ensemble_predict.predict(np.array([[1.0, 2.0]]))

    ensemble_predict = tree_module.TreeEnsemble.__new__(tree_module.TreeEnsemble)
    ensemble_predict.model_type = "xgboost"
    ensemble_predict.model_output = "raw"
    ensemble_predict.num_stacked_models = 2
    ensemble_predict._xgboost_n_outputs = 1
    ensemble_predict.trees = [types.SimpleNamespace(values=np.zeros((1, 1)))]

    with pytest.raises(NotImplementedError, match="boosted random forest is not yet supported"):
        ensemble_predict.predict(np.array([[1.0, 2.0]]))

    ensemble_predict = tree_module.TreeEnsemble.__new__(tree_module.TreeEnsemble)
    ensemble_predict.model_type = "internal"
    ensemble_predict.model_output = "raw"
    ensemble_predict.tree_limit = None
    ensemble_predict.values = np.zeros((1, 1, 2))
    ensemble_predict.input_dtype = np.float64

    with pytest.raises(ValueError, match="Both samples and labels must be provided"):
        ensemble_predict.predict(np.array([[1.0, 2.0]]), output="logloss", y=None)

    with pytest.raises(ValueError, match="does not match the number of samples to explain"):
        ensemble_predict.predict(np.array([[1.0, 2.0]]), output="logloss", y=np.array([1, 2]))


def test_treeensemble_predict_flat_output_multioutput_branch(monkeypatch):
    ensemble = tree_module.TreeEnsemble.__new__(tree_module.TreeEnsemble)
    ensemble.model_type = "internal"
    ensemble.model_output = "raw"
    ensemble.tree_output = "raw_value"
    ensemble.objective = "squared_error"
    ensemble.tree_limit = None
    ensemble.input_dtype = np.float64
    ensemble.values = np.zeros((1, 1, 2))
    ensemble.children_left = np.array([[-1]], dtype=np.int32)
    ensemble.children_right = np.array([[-1]], dtype=np.int32)
    ensemble.children_default = np.array([[-1]], dtype=np.int32)
    ensemble.features = np.array([[-1]], dtype=np.int32)
    ensemble.thresholds = np.array([[0.0]], dtype=np.float64)
    ensemble.threshold_types = np.array([[0]], dtype=np.int32)
    ensemble.max_depth = 0
    ensemble.base_offset = np.array([0.0, 0.0], dtype=np.float64)
    ensemble.num_stacked_models = 2
    ensemble.trees = [
        types.SimpleNamespace(values=np.zeros((1, 1))),
        types.SimpleNamespace(values=np.zeros((1, 1))),
    ]

    original_predict = tree_module._cext.dense_tree_predict

    def fake_dense_tree_predict(
        children_left,
        children_right,
        children_default,
        features,
        thresholds,
        threshold_types,
        values,
        max_depth,
        tree_limit,
        base_offset,
        transform_code,
        X,
        X_missing,
        y,
        output_array,
    ):
        output_array[:] = np.array([[1.0, 2.0]])

    monkeypatch.setattr(tree_module._cext, "dense_tree_predict", fake_dense_tree_predict)

    try:
        out = ensemble.predict(np.array([1.0, 2.0]))
    finally:
        monkeypatch.setattr(tree_module._cext, "dense_tree_predict", original_predict)

    np.testing.assert_allclose(out, np.array([[1.0, 2.0]]))


def test_single_tree_parses_lightgbm_structure_and_invalid_threshold_type():
    left_branch = {
        "split_index": 1,
        "split_feature": 1,
        "threshold": 0.7,
        "default_left": True,
        "internal_value": 0.2,
        "internal_count": 4,
        "left_child": {"leaf_index": 0, "leaf_value": 0.1, "leaf_count": 2},
        "right_child": {"leaf_index": 1, "leaf_value": -0.1, "leaf_count": 2},
    }
    right_branch = {
        "split_index": 2,
        "split_feature": 2,
        "threshold": 1.3,
        "default_left": True,
        "internal_value": -0.3,
        "internal_count": 4,
        "left_child": {"leaf_index": 2, "leaf_value": 0.2, "leaf_count": 2},
        "right_child": {"leaf_index": 3, "leaf_value": -0.2, "leaf_count": 2},
    }
    tree = {
        "num_leaves": 4,
        "tree_structure": {
            "split_index": 0,
            "split_feature": 0,
            "threshold": "1||3",
            "default_left": False,
            "internal_value": 0.5,
            "internal_count": 8,
            "left_child": left_branch,
            "right_child": right_branch,
        },
    }

    parsed = tree_module.SingleTree(tree)
    assert parsed.threshold_types[0] == 1
    assert parsed.children_default[0] == parsed.children_right[0]

    bad_tree = {
        "num_leaves": 2,
        "tree_structure": {
            "split_index": 0,
            "split_feature": 0,
            "threshold": {"unsupported": True},
            "default_left": True,
            "internal_value": 0.5,
            "internal_count": 3,
            "left_child": {"leaf_index": 0, "leaf_value": 0.1, "leaf_count": 1},
            "right_child": {"leaf_index": 1, "leaf_value": -0.1, "leaf_count": 2},
        },
    }

    with pytest.raises(TypeError, match="Threshold type"):
        tree_module.SingleTree(bad_tree)


def test_single_tree_parses_xgboost_json_and_text_and_rejects_unknown_input():
    json_tree = {
        "nodeid": 0,
        "cover": 6.0,
        "yes": 1,
        "no": 2,
        "missing": 1,
        "split": 0,
        "split_condition": 0.5,
        "children": [
            {"nodeid": 1, "cover": 3.0, "leaf": 0.1},
            {
                "nodeid": 2,
                "cover": 3.0,
                "yes": 3,
                "no": 4,
                "missing": 3,
                "split": 1,
                "split_condition": 1.5,
                "children": [
                    {"nodeid": 3, "cover": 2.0, "leaf": -0.2},
                    {"nodeid": 4, "cover": 1.0, "leaf": 0.3},
                ],
            },
        ],
    }
    parsed_json = tree_module.SingleTree(json_tree)
    assert parsed_json.children_left[0] == 1
    assert parsed_json.features[2] == 1

    text_tree = (
        "0:[f0<0.5] yes=1,no=2,missing=1,cover=6\n"
        "1:[f1=1.5] yes=3,no=4,missing=3,cover=3\n"
        "2:leaf=0.7,cover=3\n"
        "3:leaf=-0.2,cover=2\n"
        "4:leaf=0.1,cover=1\n"
    )
    parsed_text = tree_module.SingleTree(text_tree)
    assert parsed_text.features[0] == 0
    assert parsed_text.features[1] == 1
    assert parsed_text.values.shape[0] == 5

    with pytest.raises(TypeError, match="Unknown input to SingleTree constructor"):
        tree_module.SingleTree(object())


def test_single_tree_pyspark_branch_builds_tree(monkeypatch):
    class FakeStats:
        def __init__(self, values, count):
            self._values = values
            self._count = count

        def stats(self):
            return self._values

        def count(self):
            return self._count

    class FakeSplit:
        def featureIndex(self):
            return 0

        def threshold(self):
            return 0.5

        def getClass(self):
            return "org.apache.spark.ml.tree.ContinuousSplit"

    class FakeNode:
        def __init__(self, depth):
            self._depth = depth

        def subtreeDepth(self):
            return self._depth

        def leftChild(self):
            return FakeNode(0)

        def rightChild(self):
            return FakeNode(0)

        def prediction(self):
            return 0.3

        def impurityStats(self):
            return FakeStats([1.0, 2.0], 3.0)

        def split(self):
            return FakeSplit()

    class FakeJavaObj:
        def rootNode(self):
            return FakeNode(1)

        def getImpurity(self):
            return "gini"

    class FakeSparkTree:
        __module__ = "pyspark.ml.classification"

        def __init__(self):
            self._java_obj = FakeJavaObj()

    original_safe_isinstance = tree_module.safe_isinstance

    def fake_safe_isinstance(obj, class_path_str):
        class_path_strs = [class_path_str] if isinstance(class_path_str, str) else list(class_path_str)
        if isinstance(obj, FakeSparkTree):
            return any(
                c in class_path_strs
                for c in (
                    "pyspark.ml.classification.DecisionTreeClassificationModel",
                    "pyspark.ml.regression.DecisionTreeRegressionModel",
                )
            )
        return original_safe_isinstance(obj, class_path_str)

    monkeypatch.setattr(tree_module, "safe_isinstance", fake_safe_isinstance)

    parsed = tree_module.SingleTree(FakeSparkTree())
    assert parsed.features[0] == 0
    assert parsed.children_left[0] == 1


def test_isotree_normalize_branch():
    rs = np.random.RandomState(0)
    X = rs.randn(30, 3)
    model = sklearn.ensemble.IsolationForest(n_estimators=1, random_state=0).fit(X)
    tree = model.estimators_[0].tree_
    tree_features = model.estimators_features_[0]

    iso_tree = tree_module.IsoTree(tree, tree_features=tree_features, normalize=True)
    assert iso_tree.values.shape[0] == tree.node_count


def test_treeensemble_dict_list_and_pyod_paths(monkeypatch):
    tree_dict = {
        "children_left": np.array([-1], dtype=np.int32),
        "children_right": np.array([-1], dtype=np.int32),
        "children_default": np.array([-1], dtype=np.int32),
        "features": np.array([-1], dtype=np.int32),
        "thresholds": np.array([0.0], dtype=np.float64),
        "values": np.array([[0.4]], dtype=np.float64),
        "node_sample_weight": np.array([1.0], dtype=np.float64),
    }
    model_dict = {
        "internal_dtype": np.float32,
        "input_dtype": np.float32,
        "objective": "binary_crossentropy",
        "tree_output": "probability",
        "base_offset": 0.25,
        "trees": [tree_dict],
    }

    ensemble_from_dict = tree_module.TreeEnsemble(model_dict)
    assert ensemble_from_dict.internal_dtype == np.float32
    assert ensemble_from_dict.input_dtype == np.float32
    assert ensemble_from_dict.objective == "binary_crossentropy"

    one_tree = tree_module.SingleTree(tree_dict)
    ensemble_from_list = tree_module.TreeEnsemble([one_tree])
    assert ensemble_from_list.trees == [one_tree]

    module_pyod = types.ModuleType("pyod.models.iforest")
    IForest = type("IForest", (), {})
    IForest.__module__ = "pyod.models.iforest"
    module_pyod.IForest = IForest
    monkeypatch.setitem(sys.modules, "pyod.models.iforest", module_pyod)

    X = np.random.RandomState(1).randn(20, 2)
    iso = sklearn.ensemble.IsolationForest(n_estimators=2, random_state=1).fit(X)
    fake_pyod = IForest()
    fake_pyod.estimators_ = iso.estimators_
    fake_pyod.detector_ = types.SimpleNamespace(
        estimators_=iso.estimators_,
        estimators_features_=iso.estimators_features_,
    )

    ensemble_pyod = tree_module.TreeEnsemble(fake_pyod)
    assert ensemble_pyod.tree_output == "raw_value"


def test_treeensemble_gradient_boosting_init_branches(monkeypatch):
    X = np.random.RandomState(2).randn(30, 3)
    y_reg = np.random.RandomState(3).randn(30)

    reg = GradientBoostingRegressor(n_estimators=2, max_depth=1, random_state=0).fit(X, y_reg)

    MeanEstimator = type("MeanEstimator", (), {})
    MeanEstimator.__module__ = "sklearn.ensemble"
    QuantileEstimator = type("QuantileEstimator", (), {})
    QuantileEstimator.__module__ = "sklearn.ensemble"
    monkeypatch.setattr(sklearn.ensemble, "MeanEstimator", MeanEstimator, raising=False)
    monkeypatch.setattr(sklearn.ensemble, "QuantileEstimator", QuantileEstimator, raising=False)

    reg_mean = reg
    reg_mean.init_ = MeanEstimator()
    reg_mean.init_.mean = 1.5
    ensemble_mean = tree_module.TreeEnsemble(reg_mean)
    assert ensemble_mean.base_offset == 1.5

    reg_quantile = GradientBoostingRegressor(n_estimators=2, max_depth=1, random_state=1).fit(X, y_reg)
    reg_quantile.init_ = QuantileEstimator()
    reg_quantile.init_.quantile = -0.25
    ensemble_quantile = tree_module.TreeEnsemble(reg_quantile)
    assert ensemble_quantile.base_offset == -0.25

    reg_invalid = GradientBoostingRegressor(n_estimators=2, max_depth=1, random_state=2).fit(X, y_reg)
    reg_invalid.init_ = object()
    with pytest.raises(InvalidModelError, match="Unsupported init model type"):
        tree_module.TreeEnsemble(reg_invalid)


def test_treeensemble_hist_and_gradient_boosting_classifier_branches(monkeypatch):
    rs = np.random.RandomState(4)
    X = rs.randn(80, 4)
    y_multi = rs.randint(0, 3, size=80)

    hist_reg = HistGradientBoostingRegressor(max_iter=5, max_depth=2, random_state=0).fit(X, rs.randn(80))
    ensemble_hist_reg = tree_module.TreeEnsemble(hist_reg, model_output="predict")
    assert ensemble_hist_reg.model_output == "raw"

    hist_clf_multi = HistGradientBoostingClassifier(max_iter=5, max_depth=2, random_state=0).fit(X, y_multi)
    with pytest.raises(NotImplementedError, match="Multi-output HistGradientBoostingClassifier"):
        tree_module.TreeEnsemble(hist_clf_multi, model_output="predict_proba")

    hist_clf_for_prob = HistGradientBoostingClassifier(max_iter=5, max_depth=2, random_state=1).fit(X, y_multi)
    hist_clf_for_prob._baseline_prediction = np.array([[0.0]])
    ensemble_hist_clf = tree_module.TreeEnsemble(hist_clf_for_prob, model_output="predict_proba")
    assert ensemble_hist_clf.model_output == "probability"

    y_bin = (rs.randn(80) > 0).astype(int)
    gb_clf = GradientBoostingClassifier(n_estimators=2, max_depth=1, random_state=0).fit(X, y_bin)

    LogOddsEstimator = type("LogOddsEstimator", (), {})
    LogOddsEstimator.__module__ = "sklearn.ensemble"
    monkeypatch.setattr(sklearn.ensemble, "LogOddsEstimator", LogOddsEstimator, raising=False)

    gb_clf.init_ = LogOddsEstimator()
    gb_clf.init_.prior = 0.75
    ensemble_gb_clf = tree_module.TreeEnsemble(gb_clf)
    assert ensemble_gb_clf.base_offset == 0.75

    gb_multi = GradientBoostingClassifier(n_estimators=2, max_depth=1, random_state=1).fit(X, y_multi)
    with pytest.raises(InvalidModelError, match="only supported for binary classification"):
        tree_module.TreeEnsemble(gb_multi)


class _StubParsedTree:
    def __init__(self, value=1.0, weight=1.0):
        self.children_left = np.array([-1], dtype=np.int32)
        self.children_right = np.array([-1], dtype=np.int32)
        self.children_default = np.array([-1], dtype=np.int32)
        self.features = np.array([-1], dtype=np.int32)
        self.thresholds = np.array([0.0], dtype=np.float64)
        self.threshold_types = np.array([0], dtype=np.int32)
        self.values = np.array([[value]], dtype=np.float64)
        self.node_sample_weight = np.array([weight], dtype=np.float64)
        self.max_depth = 0


def test_treeensemble_fake_pyspark_branches(monkeypatch):
    monkeypatch.setattr(tree_module, "assert_import", lambda name: None)

    class RandomForestClassificationModel:
        __module__ = "pyspark.ml.classification"

        def __init__(self):
            self._java_obj = types.SimpleNamespace(getImpurity=lambda: "gini")
            self.treeWeights = [1.0, 1.0]
            self.trees = [types.SimpleNamespace(_zero_weight=True), types.SimpleNamespace(_zero_weight=False)]

    class GBTRegressionModel:
        __module__ = "pyspark.ml.regression"

        def __init__(self):
            self._java_obj = types.SimpleNamespace(getImpurity=lambda: "variance")
            self.treeWeights = [0.7, 0.3]
            self.trees = [object(), object()]

    class DecisionTreeRegressionModel:
        __module__ = "pyspark.ml.regression"

        def __init__(self):
            self._java_obj = types.SimpleNamespace(getImpurity=lambda: "variance")

    class UnknownSparkRegressionModel:
        __module__ = "pyspark.ml.regression"

        def __init__(self):
            self._java_obj = types.SimpleNamespace(getImpurity=lambda: "variance")

    original_safe_isinstance = tree_module.safe_isinstance

    def fake_safe_isinstance(obj, class_path_str):
        class_paths = [class_path_str] if isinstance(class_path_str, str) else list(class_path_str)
        if isinstance(obj, RandomForestClassificationModel):
            return any(
                cp
                in (
                    "pyspark.ml.classification.RandomForestClassificationModel",
                    "pyspark.ml.regression.RandomForestRegressionModel",
                )
                for cp in class_paths
            )
        if isinstance(obj, GBTRegressionModel):
            return any(
                cp
                in (
                    "pyspark.ml.classification.GBTClassificationModel",
                    "pyspark.ml.regression.GBTRegressionModel",
                )
                for cp in class_paths
            )
        if isinstance(obj, DecisionTreeRegressionModel):
            return any(
                cp
                in (
                    "pyspark.ml.classification.DecisionTreeClassificationModel",
                    "pyspark.ml.regression.DecisionTreeRegressionModel",
                )
                for cp in class_paths
            )
        if isinstance(obj, UnknownSparkRegressionModel):
            return False
        return original_safe_isinstance(obj, class_path_str)

    def fake_single_tree(tree, normalize=False, scaling=1.0, data=None, data_missing=None):
        weight = 0.0 if getattr(tree, "_zero_weight", False) else 1.0
        return _StubParsedTree(value=scaling, weight=weight)

    monkeypatch.setattr(tree_module, "safe_isinstance", fake_safe_isinstance)
    monkeypatch.setattr(tree_module, "SingleTree", fake_single_tree)

    rf = tree_module.TreeEnsemble(RandomForestClassificationModel())
    assert rf.model_type == "pyspark"
    assert rf.tree_output == "probability"
    assert rf.fully_defined_weighting is False

    gbt = tree_module.TreeEnsemble(GBTRegressionModel())
    assert gbt.objective == "squared_error"
    assert gbt.tree_output == "raw_value"

    dt = tree_module.TreeEnsemble(DecisionTreeRegressionModel())
    assert dt.tree_output == "raw_value"

    with pytest.raises(NotImplementedError, match="Unsupported Spark model type"):
        tree_module.TreeEnsemble(UnknownSparkRegressionModel())


def test_treeensemble_fake_xgboost_wrapper_branches(monkeypatch):
    module_xgb_sklearn = types.ModuleType("xgboost.sklearn")
    XGBClassifier = type("XGBClassifier", (), {"__module__": "xgboost.sklearn"})
    XGBRegressor = type("XGBRegressor", (), {"__module__": "xgboost.sklearn"})
    XGBRanker = type("XGBRanker", (), {"__module__": "xgboost.sklearn"})
    module_xgb_sklearn.XGBClassifier = XGBClassifier
    module_xgb_sklearn.XGBRegressor = XGBRegressor
    module_xgb_sklearn.XGBRanker = XGBRanker
    monkeypatch.setitem(sys.modules, "xgboost.sklearn", module_xgb_sklearn)

    def fake_set_xgb_attrs(self, data, data_missing, objective_name_map, tree_output_name_map):
        self.model_type = "xgboost"
        self.trees = [_StubParsedTree()]
        self.base_offset = 0.0
        self.objective = "squared_error"
        self.tree_output = "raw_value"
        self.num_stacked_models = getattr(self.original_model, "_stacks", 1)
        self.cat_feature_indices = None
        self.tree_limit = 1
        self._xgboost_n_outputs = max(1, self.num_stacked_models)

    monkeypatch.setattr(tree_module.TreeEnsemble, "_set_xgboost_model_attributes", fake_set_xgb_attrs)

    clf_binary = XGBClassifier()
    clf_binary._stacks = 1
    clf_binary.get_booster = lambda: types.SimpleNamespace(_stacks=1)
    clf_binary.missing = np.nan
    clf_binary.n_jobs = 2
    clf_binary.enable_categorical = False
    clf_binary.feature_types = ["q"]

    ensemble_binary = tree_module.TreeEnsemble(clf_binary, model_output="predict_proba")
    assert ensemble_binary.model_output == "probability_doubled"
    assert ensemble_binary._xgb_dmatrix_props["nthread"] == 2

    clf_multi = XGBClassifier()
    clf_multi._stacks = 3
    clf_multi.get_booster = lambda: types.SimpleNamespace(_stacks=3)
    clf_multi.missing = np.nan
    clf_multi.n_jobs = 1
    clf_multi.enable_categorical = False
    clf_multi.feature_types = ["q"]

    ensemble_multi = tree_module.TreeEnsemble(clf_multi, model_output="predict_proba")
    assert ensemble_multi.model_output == "probability"

    reg = XGBRegressor()
    reg._stacks = 1
    reg.get_booster = lambda: types.SimpleNamespace(_stacks=1)
    reg.missing = np.nan
    reg.n_jobs = 4
    reg.enable_categorical = True
    reg.feature_types = ["q", "c"]

    ensemble_reg = tree_module.TreeEnsemble(reg)
    assert ensemble_reg._xgb_dmatrix_props["nthread"] == 4

    ranker = XGBRanker()
    ranker._stacks = 1
    ranker.get_booster = lambda: types.SimpleNamespace(_stacks=1)
    ranker.missing = np.nan
    ranker.n_jobs = 3
    ranker.enable_categorical = False
    ranker.feature_types = ["q"]
    tree_module.TreeEnsemble(ranker)


def test_treeensemble_fake_lightgbm_gpboost_catboost_imblearn_ngboost_branches(monkeypatch):
    monkeypatch.setattr(tree_module, "assert_import", lambda name: None)

    module_lgb_basic = types.ModuleType("lightgbm.basic")
    LGBBooster = type("Booster", (), {"__module__": "lightgbm.basic"})
    module_lgb_basic.Booster = LGBBooster
    monkeypatch.setitem(sys.modules, "lightgbm.basic", module_lgb_basic)

    module_gp_basic = types.ModuleType("gpboost.basic")
    GPBooster = type("Booster", (), {"__module__": "gpboost.basic"})
    module_gp_basic.Booster = GPBooster
    monkeypatch.setitem(sys.modules, "gpboost.basic", module_gp_basic)

    module_lgb_sklearn = types.ModuleType("lightgbm.sklearn")
    LGBMRegressor = type("LGBMRegressor", (), {"__module__": "lightgbm.sklearn"})
    LGBMRanker = type("LGBMRanker", (), {"__module__": "lightgbm.sklearn"})
    LGBMClassifier = type("LGBMClassifier", (), {"__module__": "lightgbm.sklearn"})
    module_lgb_sklearn.LGBMRegressor = LGBMRegressor
    module_lgb_sklearn.LGBMRanker = LGBMRanker
    module_lgb_sklearn.LGBMClassifier = LGBMClassifier
    monkeypatch.setitem(sys.modules, "lightgbm.sklearn", module_lgb_sklearn)

    module_cat_core = types.ModuleType("catboost.core")
    CatBoostRegressor = type("CatBoostRegressor", (), {"__module__": "catboost.core"})
    CatBoostClassifier = type("CatBoostClassifier", (), {"__module__": "catboost.core"})
    CatBoost = type("CatBoost", (), {"__module__": "catboost.core"})
    module_cat_core.CatBoostRegressor = CatBoostRegressor
    module_cat_core.CatBoostClassifier = CatBoostClassifier
    module_cat_core.CatBoost = CatBoost
    monkeypatch.setitem(sys.modules, "catboost.core", module_cat_core)

    module_imb = types.ModuleType("imblearn.ensemble._forest")
    BRF = type("BalancedRandomForestClassifier", (), {"__module__": "imblearn.ensemble._forest"})
    module_imb.BalancedRandomForestClassifier = BRF
    monkeypatch.setitem(sys.modules, "imblearn.ensemble._forest", module_imb)

    module_ngb = types.ModuleType("ngboost.api")
    NGBRegressor = type("NGBRegressor", (), {"__module__": "ngboost.api"})
    module_ngb.NGBRegressor = NGBRegressor
    monkeypatch.setitem(sys.modules, "ngboost.api", module_ngb)

    def fake_single_tree(tree, normalize=False, scaling=1.0, data=None, data_missing=None):
        if isinstance(tree, dict) and tree.get("raise_single_tree", False):
            raise RuntimeError("forced parse failure")
        return _StubParsedTree(value=scaling)

    class FailingCatLoader:
        def __init__(self, model):
            raise RuntimeError("forced cat loader failure")

    monkeypatch.setattr(tree_module, "SingleTree", fake_single_tree)
    monkeypatch.setattr(tree_module, "CatBoostTreeModelLoader", FailingCatLoader)

    booster = LGBBooster()
    booster.params = {"objective": "regression"}
    booster.dump_model = lambda: {"tree_info": [{"raise_single_tree": True}]}
    ens_lgb_basic = tree_module.TreeEnsemble(booster)
    assert ens_lgb_basic.trees is None

    gp_booster = GPBooster()
    gp_booster.params = {"objective": "binary"}
    gp_booster.dump_model = lambda: {"tree_info": [{"raise_single_tree": True}]}
    ens_gp = tree_module.TreeEnsemble(gp_booster)
    assert ens_gp.trees is None

    lgb_reg = LGBMRegressor()
    lgb_reg.objective = None
    lgb_reg.booster_ = booster
    ens_lgb_reg = tree_module.TreeEnsemble(lgb_reg)
    assert ens_lgb_reg.objective == "squared_error"
    assert ens_lgb_reg.tree_output == "raw_value"

    lgb_ranker = LGBMRanker()
    lgb_ranker.booster_ = booster
    lgb_ranker.objective = "lambdarank"
    ens_lgb_rank = tree_module.TreeEnsemble(lgb_ranker)
    assert ens_lgb_rank.model_type == "lightgbm"

    lgb_clf = LGBMClassifier()
    lgb_clf.n_classes_ = 3
    lgb_clf.objective = None
    lgb_clf.booster_ = booster
    ens_lgb_clf = tree_module.TreeEnsemble(lgb_clf)
    assert ens_lgb_clf.num_stacked_models == 3
    assert ens_lgb_clf.objective == "binary_crossentropy"
    assert ens_lgb_clf.tree_output == "log_odds"

    cat_reg = CatBoostRegressor()
    cat_reg.get_cat_feature_indices = lambda: [0]
    ens_cat_reg = tree_module.TreeEnsemble(cat_reg)
    assert ens_cat_reg.trees is None

    cat_clf = CatBoostClassifier()
    cat_clf.get_cat_feature_indices = lambda: [0, 2]
    ens_cat_clf = tree_module.TreeEnsemble(cat_clf)
    assert ens_cat_clf.tree_output == "log_odds"
    assert ens_cat_clf.objective == "binary_crossentropy"

    cat_base = CatBoost()
    cat_base.get_cat_feature_indices = lambda: [1]
    ens_cat_base = tree_module.TreeEnsemble(cat_base)
    assert ens_cat_base.cat_feature_indices == [1]

    brf = BRF()
    brf.criterion = "gini"
    brf.estimators_ = [types.SimpleNamespace(tree_=object()), types.SimpleNamespace(tree_=object())]
    ens_brf = tree_module.TreeEnsemble(brf)
    assert ens_brf.tree_output == "probability"

    X = np.random.RandomState(12).randn(40, 3)
    y = np.random.RandomState(13).randn(40)
    dt1 = sklearn.tree.DecisionTreeRegressor(max_depth=2, random_state=0).fit(X, y)
    dt2 = sklearn.tree.DecisionTreeRegressor(max_depth=2, random_state=1).fit(X, y)

    ngb = NGBRegressor()
    ngb.base_models = [[dt1], [dt2]]
    ngb.learning_rate = 0.1
    ngb.scalings = [1.0, 0.5]
    ngb.n_features = 3
    ngb.col_idxs = [np.array([0, 1]), np.array([1, 2])]
    ngb.init_params = [0.2]

    with pytest.warns(UserWarning, match='Translating model_output="raw"'):
        ens_ng_raw = tree_module.TreeEnsemble(ngb, model_output="raw")
    assert ens_ng_raw.tree_output == "raw_value"

    ens_ng_idx = tree_module.TreeEnsemble(ngb, model_output=0)
    assert ens_ng_idx.model_output == "raw"


def test_supports_model_with_masker_returns_false_on_treeensemble_exception(monkeypatch):
    def raise_treeensemble(model):
        raise RuntimeError("boom")

    monkeypatch.setattr(tree_module, "TreeEnsemble", raise_treeensemble)
    assert tree_module.TreeExplainer.supports_model_with_masker(object(), masker=None) is False


def test_treeensemble_get_transform_logloss_modes_and_predict_single_flat(monkeypatch):
    ensemble = tree_module.TreeEnsemble.__new__(tree_module.TreeEnsemble)
    ensemble.model_output = "log_loss"
    ensemble.objective = "squared_error"
    assert ensemble.get_transform() == "squared_loss"

    ensemble.objective = "binary_crossentropy"
    assert ensemble.get_transform() == "logistic_nlogloss"

    ensemble_pred = tree_module.TreeEnsemble.__new__(tree_module.TreeEnsemble)
    ensemble_pred.model_type = "internal"
    ensemble_pred.model_output = "raw"
    ensemble_pred.tree_output = "raw_value"
    ensemble_pred.objective = "squared_error"
    ensemble_pred.tree_limit = None
    ensemble_pred.input_dtype = np.float64
    ensemble_pred.values = np.zeros((1, 1, 1))
    ensemble_pred.children_left = np.array([[-1]], dtype=np.int32)
    ensemble_pred.children_right = np.array([[-1]], dtype=np.int32)
    ensemble_pred.children_default = np.array([[-1]], dtype=np.int32)
    ensemble_pred.features = np.array([[-1]], dtype=np.int32)
    ensemble_pred.thresholds = np.array([[0.0]], dtype=np.float64)
    ensemble_pred.threshold_types = np.array([[0]], dtype=np.int32)
    ensemble_pred.max_depth = 0
    ensemble_pred.base_offset = np.array([0.0], dtype=np.float64)
    ensemble_pred.num_stacked_models = 1
    ensemble_pred.trees = [types.SimpleNamespace(values=np.zeros((1, 1)))]

    original_predict = tree_module._cext.dense_tree_predict

    def fake_dense_tree_predict(
        children_left,
        children_right,
        children_default,
        features,
        thresholds,
        threshold_types,
        values,
        max_depth,
        tree_limit,
        base_offset,
        transform_code,
        X,
        X_missing,
        y,
        output_array,
    ):
        output_array[:] = np.array([[5.0]])

    monkeypatch.setattr(tree_module._cext, "dense_tree_predict", fake_dense_tree_predict)
    try:
        pred = ensemble_pred.predict(np.array([1.0, 2.0]))
    finally:
        monkeypatch.setattr(tree_module._cext, "dense_tree_predict", original_predict)

    assert pred == 5.0


def test_treeexplainer_init_invalid_masker_and_clustering_paths(monkeypatch):
    monkeypatch.setattr(tree_module, "TreeEnsemble", lambda *args, **kwargs: object())

    def fake_init_bad_masker(self, model, masker, feature_names=None):
        self.masker = types.SimpleNamespace(clustering=None)

    monkeypatch.setattr(tree_module.Explainer, "__init__", fake_init_bad_masker)
    with pytest.raises(InvalidMaskerError, match="Unsupported masker type"):
        tree_module.TreeExplainer(object(), data=np.zeros((2, 2)))

    def fake_init_clustered(self, model, masker, feature_names=None):
        self.masker = types.SimpleNamespace(clustering=np.array([0]))

    monkeypatch.setattr(tree_module.Explainer, "__init__", fake_init_clustered)
    with pytest.raises(ExplainerError, match="does not support clustered data inputs"):
        tree_module.TreeExplainer(object(), data=None)


def test_treeexplainer_init_dataframe_and_dense_data_conversions(monkeypatch):
    def fake_init_independent(self, model, masker, feature_names=None):
        independent = tree_module.maskers.Independent.__new__(tree_module.maskers.Independent)
        independent.data = masker
        independent.clustering = None
        self.masker = independent

    monkeypatch.setattr(tree_module.Explainer, "__init__", fake_init_independent)

    def fake_tree_ensemble(model, data, data_missing, model_output):
        return types.SimpleNamespace(
            model_output="raw",
            objective="squared_error",
            tree_output="raw_value",
            model_type="internal",
            predict=lambda X: np.full((len(X), 1), 7.0),
        )

    monkeypatch.setattr(tree_module, "TreeEnsemble", fake_tree_ensemble)

    df = pd.DataFrame(np.arange(6, dtype=float).reshape(3, 2), columns=["a", "b"])
    with pytest.warns(DeprecationWarning, match="approximate argument has been deprecated"):
        explainer_df = tree_module.TreeExplainer(
            object(),
            data=df,
            feature_names=["f0", "f1"],
            approximate=True,
        )

    assert isinstance(explainer_df.data, np.ndarray)
    assert explainer_df.data.shape == (3, 2)
    assert explainer_df.data_feature_names == ["f0", "f1"]
    assert explainer_df.expected_value == 7.0

    dense = DenseData(np.arange(6, dtype=float).reshape(3, 2), ["d0", "d1"])
    explainer_dense = tree_module.TreeExplainer(object(), data=dense)
    assert isinstance(explainer_dense.data, np.ndarray)
    assert explainer_dense.data.shape == (3, 2)


def test_treeexplainer_init_expected_value_and_validation_error_paths(monkeypatch):
    def fake_init_independent(self, model, masker, feature_names=None):
        independent = tree_module.maskers.Independent.__new__(tree_module.maskers.Independent)
        independent.data = masker
        independent.clustering = None
        self.masker = independent

    monkeypatch.setattr(tree_module.Explainer, "__init__", fake_init_independent)

    def fake_tree_ensemble_valueerror(model, data, data_missing, model_output):
        def fail_predict(_):
            raise ValueError("forced categorical split failure")

        return types.SimpleNamespace(
            model_output="raw",
            objective="squared_error",
            tree_output="raw_value",
            model_type="internal",
            predict=fail_predict,
        )

    monkeypatch.setattr(tree_module, "TreeEnsemble", fake_tree_ensemble_valueerror)
    with pytest.raises(ExplainerError, match="categorical splits"):
        tree_module.TreeExplainer(object(), data=np.zeros((2, 2)))

    def fake_tree_ensemble_unknown_output(model, data, data_missing, model_output):
        return types.SimpleNamespace(
            model_output="probability",
            objective=None,
            tree_output=None,
            model_type="internal",
            predict=lambda X: np.zeros((len(X), 1)),
        )

    monkeypatch.setattr(tree_module, "TreeEnsemble", fake_tree_ensemble_unknown_output)
    with pytest.raises(Exception, match="known objective or output type"):
        tree_module.TreeExplainer(object(), data=np.zeros((2, 2)))


def test_treeexplainer_init_feature_perturbation_edge_paths(monkeypatch):
    def fake_init_no_masker(self, model, masker, feature_names=None):
        self.masker = types.SimpleNamespace(clustering=None)

    monkeypatch.setattr(tree_module.Explainer, "__init__", fake_init_no_masker)

    def fake_tree_ensemble_prob(model, data, data_missing, model_output):
        return types.SimpleNamespace(
            model_output="probability",
            objective="binary_crossentropy",
            tree_output="log_odds",
            model_type="internal",
            predict=lambda X: np.zeros((1, 1)),
        )

    monkeypatch.setattr(tree_module, "TreeEnsemble", fake_tree_ensemble_prob)
    with pytest.raises(ValueError, match='Only model_output="raw" is supported'):
        tree_module.TreeExplainer(object(), data=None, feature_perturbation="tree_path_dependent")

    def fake_tree_ensemble_raw(model, data, data_missing, model_output):
        return types.SimpleNamespace(
            model_output="raw",
            objective="squared_error",
            tree_output="raw_value",
            model_type="internal",
            predict=lambda X: np.zeros((len(X), 1)),
        )

    monkeypatch.setattr(tree_module, "TreeEnsemble", fake_tree_ensemble_raw)

    class MissingBackgroundToken:
        def __eq__(self, other):
            return False

        def __ne__(self, other):
            if other == "tree_path_dependent":
                return False
            return True

    with pytest.raises(ValueError, match="background dataset must be provided"):
        tree_module.TreeExplainer(object(), data=None, feature_perturbation=MissingBackgroundToken())


def test_treeexplainer_init_slow_background_logloss_and_node_weight_paths(monkeypatch):
    def fake_init_independent(self, model, masker, feature_names=None):
        independent = tree_module.maskers.Independent.__new__(tree_module.maskers.Independent)
        independent.data = masker
        independent.clustering = None
        self.masker = independent

    monkeypatch.setattr(tree_module.Explainer, "__init__", fake_init_independent)

    class SlowBackgroundToken:
        def __init__(self):
            self._interventional_checks = 0

        def __eq__(self, other):
            if other == "auto":
                return False
            if other == "interventional":
                self._interventional_checks += 1
                return self._interventional_checks >= 2
            if other == "tree_path_dependent":
                return False
            return False

        def __ne__(self, other):
            if other == "tree_path_dependent":
                return False
            return not self.__eq__(other)

    def fake_tree_ensemble_raw(model, data, data_missing, model_output):
        return types.SimpleNamespace(
            model_output="raw",
            objective="squared_error",
            tree_output="raw_value",
            model_type="internal",
            predict=lambda X: np.zeros((len(X), 1)),
        )

    monkeypatch.setattr(tree_module, "TreeEnsemble", fake_tree_ensemble_raw)
    with pytest.warns(UserWarning, match="1001 background samples"):
        tree_module.TreeExplainer(
            object(),
            data=np.zeros((1001, 2)),
            feature_perturbation=SlowBackgroundToken(),
        )

    def fake_tree_ensemble_logloss(model, data, data_missing, model_output):
        def predict(X, y=None):
            return y.reshape(-1, 1)

        return types.SimpleNamespace(
            model_output="log_loss",
            objective="squared_error",
            tree_output="raw_value",
            model_type="internal",
            predict=predict,
        )

    monkeypatch.setattr(tree_module, "TreeEnsemble", fake_tree_ensemble_logloss)
    explainer_logloss = tree_module.TreeExplainer(object(), data=np.zeros((3, 1)))
    out = explainer_logloss.expected_value(2.0)
    np.testing.assert_allclose(out, np.array([2.0]))

    def fake_init_no_masker(self, model, masker, feature_names=None):
        self.masker = types.SimpleNamespace(clustering=None)

    monkeypatch.setattr(tree_module.Explainer, "__init__", fake_init_no_masker)

    class ToggleRawModelOutput:
        def __init__(self):
            self.raw_checks = 0

        def __eq__(self, other):
            if other == "raw":
                self.raw_checks += 1
                return self.raw_checks <= 2
            return False

    def fake_tree_ensemble_node_weight(model, data, data_missing, model_output):
        return types.SimpleNamespace(
            model_output=ToggleRawModelOutput(),
            objective="squared_error",
            tree_output="raw_value",
            model_type="internal",
            node_sample_weight=np.array([1.0]),
            values=np.array([[[2.0]]]),
            base_offset=0.5,
        )

    monkeypatch.setattr(tree_module, "TreeEnsemble", fake_tree_ensemble_node_weight)
    explainer_node_weight = tree_module.TreeExplainer(
        object(),
        data=None,
        feature_perturbation="tree_path_dependent",
    )
    assert explainer_node_weight.expected_value is None


def test_single_tree_pyspark_variance_normalize_and_categorical_split(monkeypatch):
    class FakeStats:
        def __init__(self, values, count):
            self._values = values
            self._count = count

        def stats(self):
            return self._values

        def count(self):
            return self._count

    class ContinuousSplit:
        def featureIndex(self):
            return 0

        def threshold(self):
            return 0.25

        def getClass(self):
            return "org.apache.spark.ml.tree.ContinuousSplit"

    class CategoricalSplit(ContinuousSplit):
        def getClass(self):
            return "org.apache.spark.ml.tree.CategoricalSplit"

    class FakeNode:
        def __init__(self, depth, split_obj):
            self._depth = depth
            self._split_obj = split_obj

        def subtreeDepth(self):
            return self._depth

        def leftChild(self):
            return FakeNode(0, self._split_obj)

        def rightChild(self):
            return FakeNode(0, self._split_obj)

        def prediction(self):
            return 2.0

        def impurityStats(self):
            return FakeStats([2.0], 5.0)

        def split(self):
            return self._split_obj

    class FakeJavaObj:
        def __init__(self, split_obj):
            self._split_obj = split_obj

        def rootNode(self):
            return FakeNode(1, self._split_obj)

        def getImpurity(self):
            return "variance"

    class FakeSparkTree:
        __module__ = "pyspark.ml.regression"

        def __init__(self, split_obj):
            self._java_obj = FakeJavaObj(split_obj)

    original_safe_isinstance = tree_module.safe_isinstance

    def fake_safe_isinstance(obj, class_path_str):
        class_paths = [class_path_str] if isinstance(class_path_str, str) else list(class_path_str)
        if isinstance(obj, FakeSparkTree):
            return any(
                cp
                in (
                    "pyspark.ml.classification.DecisionTreeClassificationModel",
                    "pyspark.ml.regression.DecisionTreeRegressionModel",
                )
                for cp in class_paths
            )
        return original_safe_isinstance(obj, class_path_str)

    monkeypatch.setattr(tree_module, "safe_isinstance", fake_safe_isinstance)

    parsed = tree_module.SingleTree(FakeSparkTree(ContinuousSplit()), normalize=True)
    np.testing.assert_allclose(parsed.values.sum(1), np.ones(parsed.values.shape[0]))

    with pytest.raises(NotImplementedError, match="CategoricalSplit are not yet implemented"):
        tree_module.SingleTree(FakeSparkTree(CategoricalSplit()))


def test_single_tree_lightgbm_duplicate_split_index_continue_path():
    shared_branch = {
        "split_index": 1,
        "split_feature": 1,
        "threshold": 0.8,
        "default_left": True,
        "internal_value": 0.1,
        "internal_count": 3,
        "left_child": {"leaf_index": 0, "leaf_value": -0.2, "leaf_count": 2},
        "right_child": {"leaf_index": 1, "leaf_value": 0.3, "leaf_count": 1},
    }
    tree = {
        "num_leaves": 3,
        "tree_structure": {
            "split_index": 0,
            "split_feature": 0,
            "threshold": 0.5,
            "default_left": True,
            "internal_value": 0.0,
            "internal_count": 4,
            "left_child": shared_branch,
            "right_child": shared_branch,
        },
    }

    with pytest.raises(ValueError, match="inhomogeneous"):
        tree_module.SingleTree(tree)


def test_treeensemble_catboost_success_loader_paths(monkeypatch):
    monkeypatch.setattr(tree_module, "assert_import", lambda name: None)

    module_cat_core = types.ModuleType("catboost.core")
    CatBoostRegressor = type("CatBoostRegressor", (), {"__module__": "catboost.core"})
    CatBoostClassifier = type("CatBoostClassifier", (), {"__module__": "catboost.core"})
    module_cat_core.CatBoostRegressor = CatBoostRegressor
    module_cat_core.CatBoostClassifier = CatBoostClassifier
    monkeypatch.setitem(sys.modules, "catboost.core", module_cat_core)

    class WorkingCatLoader:
        def __init__(self, model):
            self.model = model

        def get_trees(self, data=None, data_missing=None):
            return [_StubParsedTree(value=2.0)]

    monkeypatch.setattr(tree_module, "CatBoostTreeModelLoader", WorkingCatLoader)

    reg = CatBoostRegressor()
    reg.get_cat_feature_indices = lambda: [0]
    reg_ens = tree_module.TreeEnsemble(reg)
    assert len(reg_ens.trees) == 1

    clf = CatBoostClassifier()
    clf.get_cat_feature_indices = lambda: [1]
    clf_ens = tree_module.TreeEnsemble(clf)
    assert len(clf_ens.trees) == 1


def _make_xgb_ubjson_payload(
    *,
    base_score,
    feature_types,
    trees,
    iteration_indptr=None,
    num_parallel_tree="1",
):
    model_dict = {
        "gbtree_model_param": {"num_parallel_tree": num_parallel_tree},
        "trees": trees,
    }
    if iteration_indptr is not None:
        model_dict["iteration_indptr"] = iteration_indptr

    return {
        "learner": {
            "learner_model_param": {
                "num_class": "1",
                "num_target": "1",
                "base_score": base_score,
                "num_feature": "2",
            },
            "objective": {"name": "reg:squarederror"},
            "gradient_booster": {
                "name": "gbtree",
                "model": model_dict,
            },
        }
    }


def _make_fake_xgb_model(payload, feature_types, boosted_rounds=1):
    class FakeBooster:
        def __init__(self):
            self.feature_types = feature_types

        def save_raw(self, raw_format="ubj"):
            assert raw_format == "ubj"
            return b"fake"

        def num_boosted_rounds(self):
            return boosted_rounds

    return FakeBooster()


def test_xgb_loader_init_parallel_tree_and_cat_indices(monkeypatch):
    trees = [
        {
            "parents": [-1],
            "left_children": [-1],
            "right_children": [-1],
            "split_indices": [0],
            "base_weights": [0.2],
            "default_left": [1],
            "sum_hessian": [1.0],
            "split_conditions": [0.2],
            "split_type": [0],
            "categories_segments": [],
            "categories_sizes": [],
            "categories_nodes": [],
            "categories": [],
        }
    ]
    payload = _make_xgb_ubjson_payload(
        base_score="0.5",
        feature_types=["q", "c"],
        trees=trees,
        iteration_indptr=None,
    )
    monkeypatch.setattr(tree_module, "decode_ubjson_buffer", lambda fd: payload)

    model = _make_fake_xgb_model(payload, feature_types=["q", "c"], boosted_rounds=2)
    loader = tree_module.XGBTreeModelLoader(model)

    assert loader.n_trees_per_iter == 1
    np.testing.assert_array_equal(loader.cat_feature_indices, np.array([1]))


def test_xgb_loader_init_rejects_vector_leaf_diff(monkeypatch):
    trees = [
        {
            "parents": [-1],
            "left_children": [-1],
            "right_children": [-1],
            "split_indices": [0],
            "base_weights": [0.2],
            "default_left": [1],
            "sum_hessian": [1.0],
            "split_conditions": [0.2],
            "split_type": [0],
            "categories_segments": [],
            "categories_sizes": [],
            "categories_nodes": [],
            "categories": [],
        }
    ]
    payload = _make_xgb_ubjson_payload(
        base_score="0.5",
        feature_types=None,
        trees=trees,
        iteration_indptr=[0, 1, 3],
    )
    monkeypatch.setattr(tree_module, "decode_ubjson_buffer", lambda fd: payload)

    model = _make_fake_xgb_model(payload, feature_types=None)
    with pytest.raises(ValueError, match="vector-leaf is not yet supported"):
        tree_module.XGBTreeModelLoader(model)


def test_xgb_loader_init_rejects_invalid_base_score_literal(monkeypatch):
    trees = [
        {
            "parents": [-1],
            "left_children": [-1],
            "right_children": [-1],
            "split_indices": [0],
            "base_weights": [0.2],
            "default_left": [1],
            "sum_hessian": [1.0],
            "split_conditions": [0.2],
            "split_type": [0],
            "categories_segments": [],
            "categories_sizes": [],
            "categories_nodes": [],
            "categories": [],
        }
    ]
    payload = _make_xgb_ubjson_payload(
        base_score="'bad'",
        feature_types=None,
        trees=trees,
        iteration_indptr=[0, 1],
    )
    monkeypatch.setattr(tree_module, "decode_ubjson_buffer", lambda fd: payload)

    model = _make_fake_xgb_model(payload, feature_types=None)
    with pytest.raises(ValueError, match="Expected the base_score to contain a list or float"):
        tree_module.XGBTreeModelLoader(model)


def test_xgb_loader_init_rejects_vector_leaf_base_weight_shape(monkeypatch):
    trees = [
        {
            "parents": [-1, 0],
            "left_children": [1, -1],
            "right_children": [-1, -1],
            "split_indices": [0, 0],
            "base_weights": [0.2],
            "default_left": [1, 1],
            "sum_hessian": [1.0, 0.5],
            "split_conditions": [0.2, 0.0],
            "split_type": [0, 0],
            "categories_segments": [],
            "categories_sizes": [],
            "categories_nodes": [],
            "categories": [],
        }
    ]
    payload = _make_xgb_ubjson_payload(
        base_score="0.5",
        feature_types=None,
        trees=trees,
        iteration_indptr=[0, 1],
    )
    monkeypatch.setattr(tree_module, "decode_ubjson_buffer", lambda fd: payload)

    model = _make_fake_xgb_model(payload, feature_types=None)
    with pytest.raises(ValueError, match="vector-leaf is not yet supported"):
        tree_module.XGBTreeModelLoader(model)

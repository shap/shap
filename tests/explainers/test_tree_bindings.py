"""Exercise array conversion and in-place writes at the native tree boundary."""

import numpy as np
import pandas as pd
import pytest

import shap
from shap import _cext  # type: ignore[attr-defined]
from shap.explainers._tree import SingleTree


@pytest.fixture
def tree_args():
    return {
        "children_left": np.array([[1, -1, -1]], dtype=np.int32),
        "children_right": np.array([[2, -1, -1]], dtype=np.int32),
        "children_default": np.array([[1, -1, -1]], dtype=np.int32),
        "features": np.array([[0, -1, -1]], dtype=np.int32),
        "thresholds": np.array([[0.5, 0, 0]], dtype=np.float64),
        "threshold_types": np.zeros((1, 3), dtype=np.int32),
        "values": np.array([[[3], [1], [5]]], dtype=np.float64),
        "node_sample_weights": np.array([[2, 1, 1]], dtype=np.float64),
        "max_depth": 1,
        "tree_limit": 1,
        "base_offset": np.zeros(1),
        "X": np.array([[0], [1]], dtype=np.float64),
        "X_missing": np.zeros((2, 1), dtype=bool),
        "y": None,
        "R": None,
        "R_missing": None,
        "feature_dependence": 1,
        "model_output": 0,
        "interactions": False,
    }


def call_binding(name, args, out, module=_cext):
    # Keep positional calls covered: TreeExplainer uses this internal interface.
    tree = [
        args[k]
        for k in (
            "children_left",
            "children_right",
            "children_default",
            "features",
            "thresholds",
            "threshold_types",
            "values",
        )
    ]
    if name == "compute_expectations":
        return module.compute_expectations(tree[0][0], tree[1][0], args["node_sample_weights"][0], out)
    if name == "dense_tree_update_weights":
        return module.dense_tree_update_weights(*[a[0] for a in tree], 1, out, args["X"], args["X_missing"])
    if name == "dense_tree_shap":
        return module.dense_tree_shap(
            *tree,
            args["node_sample_weights"],
            args["max_depth"],
            args["X"],
            args["X_missing"],
            args["y"],
            args["R"],
            args["R_missing"],
            args["tree_limit"],
            args["base_offset"],
            out,
            args["feature_dependence"],
            args["model_output"],
            args["interactions"],
        )
    return getattr(module, name)(
        *tree,
        args["max_depth"],
        args["tree_limit"],
        args["base_offset"],
        args["model_output"],
        args["X"],
        args["X_missing"],
        args["y"],
        out,
    )


CASES = [
    ("compute_expectations", (3, 1), [[3], [1], [5]]),
    ("dense_tree_update_weights", (3,), [2, 1, 1]),
    ("dense_tree_predict", (2, 1), [[1], [5]]),
    ("dense_tree_saabas", (2, 2, 1), [[[-2], [0]], [[2], [0]]]),
    ("dense_tree_shap", (2, 2, 1), [[[-2], [3]], [[2], [3]]]),
]


@pytest.mark.parametrize("name,shape,expected", CASES)
def test_in_place_results(tree_args, name, shape, expected):
    out = np.zeros(shape)
    if name == "compute_expectations":
        out[1:, 0] = [1, 5]
    result = call_binding(name, tree_args, out)
    np.testing.assert_allclose(out, expected)
    assert result == (1 if name in ("compute_expectations", "dense_tree_update_weights") else 3)


@pytest.mark.parametrize("name,shape,_expected", CASES)
@pytest.mark.parametrize("invalid", ["dtype", "strided", "readonly"])
def test_reject_output_copies(tree_args, name, shape, _expected, invalid):
    if invalid == "dtype":
        out = np.zeros(shape, dtype=np.float32)
    elif invalid == "strided":
        out = np.zeros((*shape, 2))[..., 0]
    else:
        out = np.zeros(shape)
        out.flags.writeable = False
    before = out.copy()
    with pytest.raises(TypeError):
        call_binding(name, tree_args, out)
    np.testing.assert_array_equal(out, before)


def test_readonly_and_converted_inputs(tree_args):
    for name, value in tree_args.items():
        if isinstance(value, np.ndarray):
            if value.dtype == np.float64:
                value = value.astype(np.float32)
            elif value.dtype == np.int32:
                value = value.astype(np.int64)
            # Use non-contiguous, read-only input views.
            view = np.zeros((*value.shape, 2), dtype=value.dtype)[..., 0]
            view[...] = value
            view.flags.writeable = False
            tree_args[name] = view
    out = np.zeros((2, 2, 1))
    call_binding("dense_tree_shap", tree_args, out)
    np.testing.assert_allclose(out, [[[-2], [3]], [[2], [3]]])


@pytest.mark.parametrize(
    "name,value,match",
    [
        ("base_offset", np.zeros(2), "base_offset"),
        ("X_missing", np.zeros((1, 1), dtype=bool), "X_missing"),
        ("children_left", np.zeros((1, 2), dtype=np.int32), "children_left"),
        ("y", np.zeros(1), "y"),
        ("R", np.zeros((2, 1)), "R and R_missing"),
        ("R_missing", np.zeros((2, 1), dtype=bool), "R and R_missing"),
        ("tree_limit", 2, "tree_limit"),
        ("max_depth", -1, "max_depth"),
    ],
)
def test_reject_inconsistent_shapes(tree_args, name, value, match):
    tree_args[name] = value
    with pytest.raises(ValueError, match=match):
        call_binding("dense_tree_shap", tree_args, np.zeros((2, 2, 1)))


def test_interaction_output_shape(tree_args):
    tree_args["interactions"] = True
    with pytest.raises(ValueError, match="out_contribs"):
        call_binding("dense_tree_shap", tree_args, np.zeros((2, 2, 1)))
    out = np.zeros((2, 2, 2, 1))
    call_binding("dense_tree_shap", tree_args, out)
    np.testing.assert_allclose(out[:, 0, 0, 0], [-2, 2])
    np.testing.assert_allclose(out[:, 1, 1, 0], 3)


def test_background_and_labels(tree_args):
    tree_args.update(feature_dependence=0, R=tree_args["X"], R_missing=tree_args["X_missing"])
    out = np.zeros((2, 2, 1))
    call_binding("dense_tree_shap", tree_args, out)
    np.testing.assert_allclose(out, [[[-2], [3]], [[2], [3]]])
    tree_args["model_output"] = 3
    with pytest.raises(ValueError, match="labels y"):
        call_binding("dense_tree_shap", tree_args, np.zeros_like(out))
    tree_args["y"] = np.zeros(2)
    out.fill(0)
    call_binding("dense_tree_shap", tree_args, out)
    np.testing.assert_allclose(out.sum(axis=1), [[1], [25]], atol=1e-12)


@pytest.mark.parametrize("name,shape,_expected", CASES)
def test_reject_output_shape(tree_args, name, shape, _expected):
    with pytest.raises(ValueError):
        call_binding(name, tree_args, np.zeros((*shape, 1)))


@pytest.mark.parametrize("name", ["y", "R", "R_missing", "base_offset"])
def test_invalid_input_conversion(tree_args, name):
    tree_args[name] = np.array(["invalid"])
    with pytest.raises(TypeError):
        call_binding("dense_tree_shap", tree_args, np.zeros((2, 2, 1)))


def test_single_tree_normalizes_mutable_buffers(tree_args):
    tree = {
        name: tree_args[name][0]
        for name in ("children_left", "children_right", "children_default", "features", "thresholds", "values")
    }
    tree["node_sample_weight"] = tree_args["node_sample_weights"][0].astype(np.float32)
    tree["node_sample_weight"].flags.writeable = False
    values = np.zeros((3, 2), dtype=np.float32)
    values[:, 0] = [0, 1, 5]
    tree["values"] = values[:, :1]
    parsed = SingleTree(tree, data=tree_args["X"], data_missing=tree_args["X_missing"])
    np.testing.assert_allclose(parsed.values[:, 0], [3, 1, 5])
    np.testing.assert_allclose(parsed.node_sample_weight, [2, 1, 1])
    assert parsed.values.dtype == parsed.node_sample_weight.dtype == np.float64
    assert parsed.values.flags.c_contiguous and parsed.values.flags.writeable


@pytest.mark.parametrize("kind", ["list", "series"])
def test_array_like_labels(tree_args, kind):
    X = np.array([[0.0], [1.0]])
    y = np.array([1.0, 5.0])
    tree = {
        name: tree_args[name][0]
        for name in ("children_left", "children_right", "children_default", "features", "thresholds", "values")
    }
    tree["node_sample_weight"] = tree_args["node_sample_weights"][0]
    model = {"trees": [tree], "objective": "squared_error", "tree_output": "raw_value"}
    explainer = shap.TreeExplainer(model, X, model_output="log_loss")
    labels = y.tolist() if kind == "list" else pd.Series(y)
    np.testing.assert_allclose(explainer.shap_values(X, labels), explainer.shap_values(X, y))
    np.testing.assert_allclose(
        explainer.model.predict(X, y=labels, output="logloss"),
        explainer.model.predict(X, y=y, output="logloss"),
    )

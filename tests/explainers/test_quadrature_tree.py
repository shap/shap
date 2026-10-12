"""Quadrature TreeSHAP against exhaustive coalitions and analytic deep trees."""

import math

import numpy as np
import pytest

import shap
from shap.explainers._tree import TreeEnsemble


def random_tree(seed, outputs=2):
    rng = np.random.default_rng(seed)
    nodes = 31
    left = np.full(nodes, -1, dtype=np.int32)
    right = left.copy()
    left[:15] = np.arange(15) * 2 + 1
    right[:15] = left[:15] + 1
    cover = np.ones(nodes)
    cover[15:] = rng.uniform(0.1, 10, 16)
    for node in range(14, -1, -1):
        cover[node] = cover[left[node]] + cover[right[node]]
    return {
        "children_left": left,
        "children_right": right,
        "children_default": np.where(rng.random(nodes) < 0.5, left, right),
        "features": np.where(left >= 0, rng.integers(0, 4, nodes), -1),
        "thresholds": rng.normal(size=nodes),
        "values": rng.normal(size=(nodes, outputs)),
        "node_sample_weight": cover,
    }


def coalition_value(model, row, coalition, tree=0, node=0):
    left = model.children_left[tree, node]
    if left < 0:
        return model.values[tree, node]
    right = model.children_right[tree, node]
    feature = model.features[tree, node]
    if coalition & (1 << feature):
        value = row[feature]
        if np.isnan(value):
            child = model.children_default[tree, node]
        else:
            threshold = model.thresholds[tree, node]
            goes_left = (
                bool(int(threshold) & (1 << (int(value) - 1)))
                if model.threshold_types[tree, node] == 1
                else value <= threshold
            )
            child = left if goes_left else right
        return coalition_value(model, row, coalition, tree, child)
    weights = model.node_sample_weight[tree]
    return (
        weights[left] * coalition_value(model, row, coalition, tree, left)
        + weights[right] * coalition_value(model, row, coalition, tree, right)
    ) / weights[node]


def exhaustive(model, row, tree_limit):
    """Definition of Shapley values and half-SII pairwise interactions."""
    m = len(row)
    game = np.array(
        [
            sum((coalition_value(model, row, s, tree) for tree in range(tree_limit)), start=model.base_offset.copy())
            for s in range(1 << m)
        ]
    )
    values = np.zeros((m, model.num_outputs))
    interactions = np.zeros((m, m, model.num_outputs))
    for i in range(m):
        for s in range(1 << m):
            if s & (1 << i):
                continue
            size = s.bit_count()
            weight = math.factorial(size) * math.factorial(m - size - 1) / math.factorial(m)
            values[i] += weight * (game[s | (1 << i)] - game[s])
            for j in range(i + 1, m):
                if s & (1 << j):
                    continue
                weight = math.factorial(size) * math.factorial(m - size - 2) / (2 * math.factorial(m - 1))
                interactions[i, j] += weight * (
                    game[s | (1 << i) | (1 << j)] - game[s | (1 << i)] - game[s | (1 << j)] + game[s]
                )
        for j in range(i):
            interactions[i, j] = interactions[j, i]
    for i in range(m):
        interactions[i, i] = values[i] - interactions[i].sum(axis=0)
    return values, interactions, game[0]


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("categorical", [False, True])
@pytest.mark.parametrize("tree_limit", [1, 2])
@pytest.mark.parametrize("outputs", [2, 3, 8])
def test_quadrature_exhaustive(seed, categorical, tree_limit, outputs):
    model = TreeEnsemble(
        {
            "trees": [random_tree(seed, outputs), random_tree(seed + 20, outputs)],
            "base_offset": np.linspace(0.25, -0.5, outputs),
        },
        model_output="raw",
    )
    if categorical:
        # Exercise repeated categorical constraints without a model-library loader.
        model.threshold_types[:, :15] = 1
        model.thresholds[:, :15] = np.tile([1, 2, 3, 4, 5], 3)
    rows = np.array([[1, 2, 3, 1], [2, np.nan, 1, 3], [np.nan, np.nan, 2, 2]], dtype=float)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(rows, tree_limit=tree_limit)
    if isinstance(values, list):
        values = np.stack(values, axis=-1)
    pairs = explainer.shap_interaction_values(rows, tree_limit=tree_limit)
    if isinstance(pairs, list):
        pairs = np.stack(pairs, axis=-1)
    for row, actual, interactions in zip(rows, values, pairs):
        expected, expected_pairs, baseline = exhaustive(model, row, tree_limit)
        np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)
        np.testing.assert_allclose(interactions, expected_pairs, atol=2e-12, rtol=2e-12)
        np.testing.assert_allclose(explainer.expected_value, baseline, atol=2e-12)
        np.testing.assert_allclose(interactions.sum(axis=0), actual, atol=2e-12)


def chain_model(depth, repeated=False):
    """A conjunction: only the final leaf has value one; every branch has cover 1/2."""
    nodes = 2 * depth + 1
    left = np.full(nodes, -1, dtype=np.int32)
    right = left.copy()
    left[:depth] = np.arange(depth) + depth + 1
    right[:depth] = np.arange(depth) + 1
    cover = np.exp2(-np.arange(nodes, dtype=float))
    cover[depth + 1 :] = cover[1 : depth + 1]
    values = np.zeros((nodes, 1))
    values[depth] = 1
    return {
        "trees": [
            {
                "children_left": left,
                "children_right": right,
                "children_default": left.copy(),
                "features": np.r_[np.zeros(depth, dtype=int) if repeated else np.arange(depth), np.full(depth + 1, -1)],
                "thresholds": np.zeros(nodes),
                "values": values,
                "node_sample_weight": cover,
            }
        ]
    }


@pytest.mark.parametrize("depth", [1, 16, 17, 32, 64, 129])
def test_quadrature_deep_conjunction(depth):
    explainer = shap.TreeExplainer(chain_model(depth))
    row = np.ones((1, depth))
    values = explainer.shap_values(row)
    baseline = 2.0**-depth
    np.testing.assert_allclose(values, (1 - baseline) / depth, atol=2e-13, rtol=2e-12)
    pairs = explainer.shap_interaction_values(row)
    np.testing.assert_allclose(pairs.sum(axis=1), values, atol=2e-13)
    np.testing.assert_allclose(pairs, pairs.swapaxes(1, 2), atol=2e-13)
    if depth > 1:
        # Integral of (1/2 + t/2) ** (depth-2), times (1/2)^2 / 2.
        off_diag = (1 - 2.0 ** -(depth - 1)) / (4 * (depth - 1))
        np.testing.assert_allclose(pairs[0, 0, 1], off_diag, atol=2e-13)


@pytest.mark.parametrize("depth", [20, 100])
def test_quadrature_repeated_feature(depth):
    explainer = shap.TreeExplainer(chain_model(depth, repeated=True))
    rows = np.array([[1.0], [-1.0], [np.nan]])
    values = explainer.shap_values(rows)
    expected = np.array([[1.0], [0.0], [0.0]]) - 2.0**-depth
    np.testing.assert_allclose(values, expected, atol=2e-13)
    np.testing.assert_allclose(explainer.shap_interaction_values(rows)[:, :, 0], expected, atol=2e-13)


def test_quadrature_constant_tree():
    model = {
        "trees": [
            {
                "children_left": np.array([-1]),
                "children_right": np.array([-1]),
                "children_default": np.array([-1]),
                "features": np.array([-1]),
                "thresholds": np.array([0.0]),
                "values": np.array([[2.0, 3.0]]),
                "node_sample_weight": np.array([10.0]),
            }
        ],
        "base_offset": np.array([0.5, 1.0]),
    }
    explainer = shap.TreeExplainer(model)
    rows = np.ones((3, 4))
    np.testing.assert_array_equal(explainer.shap_values(rows), 0)
    np.testing.assert_array_equal(explainer.shap_interaction_values(rows), 0)
    np.testing.assert_allclose(explainer.expected_value, [2.5, 4])


@pytest.mark.parametrize("library", ["xgboost", "lightgbm"])
def test_quadrature_dense_model_dispatch(library, monkeypatch):
    module = pytest.importorskip(library)
    rng = np.random.default_rng(42)
    rows = rng.normal(size=(40, 4))
    target = rows[:, 0] * rows[:, 1] + rows[:, 2]
    if library == "xgboost":
        model = module.XGBRegressor(n_estimators=3, max_depth=3, n_jobs=1).fit(rows, target)
        native = model.get_booster().predict(module.DMatrix(rows), pred_contribs=True)
    elif library == "lightgbm":
        model = module.LGBMRegressor(n_estimators=3, max_depth=3, min_child_samples=2, verbosity=-1, n_jobs=1)
        model.fit(rows, target)
        native = model.booster_.predict(rows, pred_contrib=True)
    explainer = shap.TreeExplainer(model)
    assert explainer._can_use_quadrature_tree_shap(rows)
    calls = []
    original = shap._cext.dense_tree_shap

    def record(*args):
        calls.append(args[-1])
        return original(*args)

    monkeypatch.setattr(shap._cext, "dense_tree_shap", record)
    values = explainer.shap_values(rows)
    pairs = explainer.shap_interaction_values(rows)
    assert calls == [False, True]
    np.testing.assert_allclose(values, native[:, :-1], atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(explainer.expected_value, native[0, -1], atol=2e-6)
    np.testing.assert_allclose(pairs.sum(axis=1), values, atol=2e-12)


def test_quadrature_xgboost_native_fallbacks():
    xgb = pytest.importorskip("xgboost")
    rng = np.random.default_rng(0)
    rows = rng.normal(size=(30, 3))
    rows[::3, 0] = -999
    model = xgb.XGBRegressor(n_estimators=2, max_depth=2, n_jobs=1, missing=-999).fit(rows, rows[:, 1])
    explainer = shap.TreeExplainer(model)
    matrix = xgb.DMatrix(rows, missing=-999)
    assert not explainer._can_use_quadrature_tree_shap(rows)
    assert not explainer._can_use_quadrature_tree_shap(matrix)
    expected = model.get_booster().predict(matrix, pred_contribs=True)
    np.testing.assert_allclose(explainer.shap_values(rows), expected[:, :-1], atol=2e-6)
    np.testing.assert_allclose(explainer.shap_values(matrix), expected[:, :-1], atol=2e-6)


def test_quadrature_tree_limit_updates_baseline():
    explainer = shap.TreeExplainer({"trees": [random_tree(2, outputs=1), random_tree(8, outputs=1)]})
    row = np.ones((1, 4))
    for limit in [1, 2, 1]:
        expected, expected_pairs, baseline = exhaustive(explainer.model, row[0], limit)
        np.testing.assert_allclose(explainer.shap_values(row, tree_limit=limit)[0], expected[:, 0], atol=2e-12)
        np.testing.assert_allclose(explainer.expected_value, baseline[0], atol=2e-12)
        np.testing.assert_allclose(
            explainer.shap_interaction_values(row, tree_limit=limit)[0], expected_pairs[:, :, 0], atol=2e-12
        )
        np.testing.assert_allclose(explainer.expected_value, baseline[0], atol=2e-12)


@pytest.mark.parametrize("probability", [1e-15, 1e-30, 1e-100])
def test_quadrature_rare_branch(probability):
    model = {
        "trees": [
            {
                "children_left": np.array([1, -1, -1]),
                "children_right": np.array([2, -1, -1]),
                "children_default": np.array([1, -1, -1]),
                "features": np.array([0, -1, -1]),
                "thresholds": np.zeros(3),
                "values": np.array([[0], [1 / probability], [0]]),
                "node_sample_weight": np.array([1, probability, 1 - probability]),
            }
        ]
    }
    explainer = shap.TreeExplainer(model)
    # A float32 probability floor changes both of these values dramatically.
    np.testing.assert_allclose(explainer.shap_values(np.ones((1, 1))), -1, atol=2e-13)
    np.testing.assert_allclose(explainer.expected_value, 1, atol=2e-13)


def test_quadrature_dart_native_fallback():
    xgb = pytest.importorskip("xgboost")
    rng = np.random.default_rng(3)
    rows = rng.normal(size=(30, 3))
    model = xgb.XGBRegressor(booster="dart", rate_drop=0.5, n_estimators=6, max_depth=2, n_jobs=1)
    model.fit(rows, rows[:, 0] * rows[:, 1])
    explainer = shap.TreeExplainer(model)
    assert not explainer._can_use_quadrature_tree_shap(rows)
    native = model.get_booster().predict(xgb.DMatrix(rows), pred_contribs=True)
    np.testing.assert_allclose(explainer.shap_values(rows), native[:, :-1], atol=2e-6)


def test_quadrature_lightgbm_zero_missing_fallback():
    lgb = pytest.importorskip("lightgbm")
    rng = np.random.default_rng(5)
    rows = rng.normal(size=(40, 3))
    rows[::2, 0] = 0
    model = lgb.LGBMRegressor(n_estimators=3, min_child_samples=2, zero_as_missing=True, verbosity=-1, n_jobs=1)
    model.fit(rows, rows[:, 0] + rows[:, 1])
    explainer = shap.TreeExplainer(model)
    assert not explainer._can_use_quadrature_tree_shap(rows)
    native = model.booster_.predict(rows, pred_contrib=True)
    np.testing.assert_allclose(explainer.shap_values(rows), native[:, :-1], atol=2e-12)


def test_quadrature_empty_batch():
    explainer = shap.TreeExplainer(chain_model(3))
    baseline = explainer.expected_value
    assert explainer.shap_values(np.empty((0, 3))).shape == (0, 3)
    assert explainer.expected_value == baseline

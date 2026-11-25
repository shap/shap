"""Regression tests for Independent (interventional) TreeSHAP with large trees.

See GH#3486: the independent TreeSHAP C++ implementation (``tree_shap_indep``
in ``shap/cext/tree_shap.h``) stored tree node indices in 16-bit integers,
so any single tree with more than ``2**15 - 1`` nodes caused an integer
overflow and a segmentation fault during ``shap_values``.
"""

import numpy as np

import shap


def _make_wide_tree(n_internal_nodes: int = 40_000, n_features: int = 32) -> dict:
    """Build a wide binary tree with ``n_internal_nodes`` internal nodes.

    The tree is a perfect binary tree using the usual array indexing (children
    of node ``i`` are ``2i + 1`` and ``2i + 2``), so it has
    ``2 * n_internal_nodes + 1`` nodes in total — more than ``2**15`` by
    default, which overflows a signed 16-bit node index.

    The features are assigned so that the all-right path (taken by samples
    above every threshold) and the all-left path (taken by samples below every
    threshold) see pairwise distinct features. This prevents the independent
    TreeSHAP traversal from unwinding early, forcing it to walk all the way
    down to nodes whose indices exceed ``2**15 - 1``.
    """
    n_nodes = 2 * n_internal_nodes + 1
    children_left = np.full(n_nodes, -1, dtype=np.int32)
    children_right = np.full(n_nodes, -1, dtype=np.int32)
    children_default = np.full(n_nodes, -1, dtype=np.int32)
    feature = np.zeros(n_nodes, dtype=np.int32)
    threshold = np.full(n_nodes, 0.5, dtype=np.float64)
    value = np.zeros((n_nodes, 1), dtype=np.float64)
    node_sample_weight = np.ones(n_nodes, dtype=np.float64)

    for i in range(n_internal_nodes):
        children_left[i] = 2 * i + 1
        children_right[i] = 2 * i + 2
        children_default[i] = children_left[i]
    for i in range(n_internal_nodes, n_nodes):
        # deterministic, heterogeneous leaf values
        value[i, 0] = ((i * 2_654_435_761) % 1000) / 1000.0

    # feature 0 at the root (shared by both paths)
    feature[0] = 0
    # all-right path: 2, 6, 14, 30, ... gets features 1, 2, 3, ...
    feat = 1
    node = 2
    while node < n_internal_nodes:
        feature[node] = feat
        feat += 1
        node = 2 * node + 2
    # all-left path: 1, 3, 7, 15, ... gets features 17, 18, 19, ...
    feat = 17
    node = 1
    while node < n_internal_nodes:
        feature[node] = feat
        feat += 1
        node = 2 * node + 1
    assert feat <= n_features

    return {
        "children_left": children_left,
        "children_right": children_right,
        "children_default": children_default,
        "feature": feature,
        "threshold": threshold,
        "value": value,
        "node_sample_weight": node_sample_weight,
    }


def test_independent_treeshap_tree_with_more_than_int16_nodes():
    """Independent TreeSHAP must not segfault when a single tree has >= 2**15 nodes.

    Regression test for https://github.com/shap/shap/issues/3486
    """
    tree = _make_wide_tree()
    assert tree["children_right"].shape[0] > np.iinfo(np.int16).max

    # x goes right at every split, r goes left at every split, so the two
    # paths diverge at the root and the traversal is forced down to nodes
    # whose indices exceed 2**15 - 1 (which previously overflowed the int16
    # fields of the Node struct and corrupted memory).
    background = np.full((2, 32), 0.1)
    X = np.full((2, 32), 0.9)

    explainer = shap.TreeExplainer(
        {"trees": [tree]},
        data=background,
        feature_perturbation="interventional",
    )
    shap_values = np.asarray(explainer.shap_values(X))

    assert shap_values.shape[:2] == X.shape

    # SHAP values must sum to the model output minus the expected value
    model_output = explainer.model.predict(X)
    assert np.allclose(
        shap_values.reshape(X.shape[0], -1).sum(1) + np.asarray(explainer.expected_value).ravel(),
        np.asarray(model_output).ravel(),
        atol=1e-6,
    )

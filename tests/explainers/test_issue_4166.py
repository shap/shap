import numpy as np
from sklearn.tree import DecisionTreeClassifier

import shap


def test_custom_tree_node_sample_weight_data_count():
    """node_sample_weight should be raw data count for SHAP local accuracy.

    Regression test for GitHub issue #4166.

    LightGBM uses data count internally; XGBoost uses hessian sum.
    For custom tree dicts passed to TreeExplainer, the correct value is
    the raw sample count because it preserves the SHAP local accuracy
    property: sum(shap_values) + expected_value == model_output.

    This test confirms that passing weighted_n_node_samples (data count)
    from a sklearn tree satisfies local accuracy to machine precision.
    """
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 4)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    clf = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, y)
    tree_ = clf.tree_

    # Normalize leaf values to probabilities (2 output classes)
    raw_values = tree_.value[:, 0, :]
    norm_values = (raw_values.T / raw_values.sum(1)).T

    tree_dict = {
        "children_left": tree_.children_left,
        "children_right": tree_.children_right,
        "children_default": tree_.children_right.copy(),
        "features": tree_.feature,
        "thresholds": tree_.threshold,
        "values": norm_values,
        "node_sample_weight": tree_.weighted_n_node_samples,  # data count
    }

    explainer = shap.TreeExplainer({"trees": [tree_dict]})
    sv = explainer.shap_values(X)

    # sv shape is (100, 4, 2) for 2-class output — check class 1
    assert sv.ndim == 3, f"Expected 3D shap values, got shape {sv.shape}"
    sv1 = sv[:, :, 1]
    ev1 = explainer.expected_value[1]

    predicted = clf.predict_proba(X)[:, 1]
    diffs = np.abs(sv1.sum(1) + ev1 - predicted)

    assert diffs.max() < 1e-4, (
        f"Local accuracy violated when using data count for node_sample_weight. "
        f"Max diff: {diffs.max():.2e}. "
        f"Ensure node_sample_weight is the raw sample count, not the hessian sum. "
        f"See GitHub issue #4166."
    )


def test_custom_tree_node_sample_weight_hessian_differs():
    """Demonstrate that hessian sum differs from data count for classification.

    This is the numerical evidence behind issue #4166: for classification
    tasks, hessian weights (p*(1-p) per sample) sum to a value far smaller
    than the raw sample count, so they represent fundamentally different
    quantities and should not be used interchangeably as node_sample_weight.
    """
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 4)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    clf = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, y)
    tree_ = clf.tree_

    # Compute per-leaf hessian sums: h_i = p_i * (1 - p_i)
    proba = clf.predict_proba(X)[:, 1]
    hess = proba * (1 - proba)
    node_hess = np.zeros(tree_.node_count)
    for leaf_idx, h in zip(clf.apply(X), hess):
        node_hess[leaf_idx] += h

    data_count_root = tree_.weighted_n_node_samples[0]
    hessian_sum_leaves = node_hess.sum()

    # For classification the ratio should be >> 1 (they are not the same)
    ratio = data_count_root / hessian_sum_leaves
    assert ratio > 2.0, (
        f"Expected data count and hessian sum to differ significantly for "
        f"classification (ratio > 2), but got ratio={ratio:.2f}. "
        f"See GitHub issue #4166."
    )

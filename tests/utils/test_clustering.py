"""Focused tests for ``shap.utils._clustering`` using function-style pytest tests."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from shap.utils._clustering import (
    _mask_delta_score,
    _pt_shuffle_rec,
    _reverse_window,
    _reverse_window_score_gain,
    delta_minimization_order,
    hclust,
    hclust_ordering,
    partition_tree,
    partition_tree_shuffle,
    xgboost_distances_r2,
)
from shap.utils._exceptions import DimensionError


def _example_linkage_tree() -> np.ndarray:
    return np.array([[0, 1, 0.1, 2], [2, 3, 0.2, 3]], dtype=float)


def test_numba_mask_score_reverse_window_and_gain():
    """Numba helpers compute expected score, reversal, and gain for a small mask set."""
    m1 = np.array([1, 0, 1, 0], dtype=bool)
    m2 = np.array([0, 0, 1, 1], dtype=bool)
    assert _mask_delta_score(m1, m2) == 2
    assert _mask_delta_score.py_func(np.array([True]), np.array([False])) == 1

    order = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    _reverse_window(order, start=1, length=3)
    assert_array_equal(order, np.array([0, 3, 2, 1, 4], dtype=np.int64))
    _reverse_window.py_func(order, start=0, length=2)

    masks = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=np.bool_)
    score_gain = _reverse_window_score_gain(masks, np.array([0, 1, 2, 3], dtype=np.int64), start=1, length=2)
    assert score_gain == -2
    assert isinstance(
        _reverse_window_score_gain.py_func(masks, np.array([0, 1, 2, 3], dtype=np.int64), 1, 2), (int, np.integer)
    )


def test_delta_minimization_order_permutation_and_positive_gain_pyfunc():
    """delta_minimization_order returns a permutation and py_func covers the gain>0 reversal branch."""
    all_masks = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.bool_)
    out = delta_minimization_order(all_masks, max_swap_size=4, num_passes=2)
    assert set(out.tolist()) == {0, 1, 2, 3}

    pos_gain_masks = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 1]], dtype=np.bool_)
    py_out = delta_minimization_order.py_func(pos_gain_masks, max_swap_size=4, num_passes=1)
    assert_array_equal(py_out, np.array([0, 2, 1, 3], dtype=np.int64))
    assert delta_minimization_order(np.empty((0, 3), dtype=np.bool_)).shape == (0,)


def test_partition_tree_shuffle_and_pt_shuffle_rec_branches():
    """Recursive shuffle covers include/exclude leaf branches and the public wrapper output contract."""
    tree = _example_linkage_tree()

    excluded_mask = np.array([True, True, False], dtype=bool)
    assert _pt_shuffle_rec.py_func(-1, np.full(2, -1, dtype=np.int64), excluded_mask, tree, M=3, pos=0) == 0

    included_mask = np.array([True, True, True], dtype=bool)
    indexes = np.full(3, -1, dtype=np.int64)
    assert _pt_shuffle_rec.py_func(-1, indexes, included_mask, tree, M=3, pos=0) == 1
    assert indexes[0] == 2

    np.random.seed(42)
    wrapper_mask = np.array([True, True, False], dtype=bool)
    out = np.empty(wrapper_mask.sum(), dtype=np.int64)
    partition_tree_shuffle(out, wrapper_mask, tree)
    assert set(out.tolist()) == {0, 1}


def test_partition_tree_and_hclust_ordering_outputs():
    """partition_tree and hclust_ordering return valid structures and reject single-row ordering input."""
    np.random.seed(42)
    X_df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": [2.0, 3.0, 4.0, 5.0], "c": [5.0, 1.0, 2.0, 3.0]})
    clustering = partition_tree(X_df, metric="correlation")
    assert clustering.shape == (2, 4)

    order = hclust_ordering(np.array([[0.0], [1.0], [3.0], [10.0]]), metric="sqeuclidean")
    assert set(order.tolist()) == {0, 1, 2, 3}

    with pytest.raises(ValueError, match="empty distance matrix"):
        hclust_ordering(np.array([[1.0, 2.0]]))


def test_hclust_validation_and_metric_paths(monkeypatch):
    """hclust validates inputs, warns when y is ignored, and supports metric='auto' with label-based distances."""
    with pytest.raises(DimensionError):
        hclust(np.arange(6), random_state=0)

    with pytest.raises(ValueError, match="Unknown linkage type"):
        hclust(np.array([[1.0, 2.0], [2.0, 3.0]]), linkage="random-string", random_state=0)  # type: ignore[arg-type]

    X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 8.0], [4.0, 16.0]])
    y = np.array([0, 1, 0, 1])
    with pytest.warns(UserWarning, match="Ignoring the y argument"):
        cosine_out = hclust(X, y=y, metric="cosine", linkage="single")
    assert cosine_out.shape == (1, 4)

    dist_full = np.array([[0.0, 0.1, 0.9], [0.6, 0.0, 0.2], [0.4, 0.8, 0.0]])

    def fake_xgb_distances(_x, _y, random_state=0):
        return dist_full

    from shap.utils import _clustering as clustering_mod

    monkeypatch.setattr(clustering_mod, "xgboost_distances_r2", fake_xgb_distances)
    X_small = np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [2.0, 3.0, 4.0]])
    y_small = np.array([0.0, 1.0, 0.0])
    out_single = hclust(X_small, y=y_small, metric="auto", linkage="single")
    out_complete = hclust(X_small, y=y_small, metric="auto", linkage="complete")
    out_average = hclust(X_small, y=y_small, metric="auto", linkage="average")
    assert out_single.shape == out_complete.shape == out_average.shape == (2, 4)
    assert_allclose(out_single[:, 2], np.sort(out_single[:, 2]))


def test_hclust_accepts_dataframe_input():
    """hclust accepts DataFrame input and returns a linkage matrix for non-label metrics."""
    np.random.seed(42)
    X_df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 3.0, 4.0]})
    out = hclust(X_df, y=None, linkage="single", metric="cosine")
    assert out.shape == (1, 4)


def test_xgboost_distances_non_warning_path(monkeypatch):
    """xgboost_distances_r2 returns bounded distances and zeros on diagonal when predictions vary."""
    import sys
    import types

    class FakeModel:
        def __init__(self, **kwargs):
            return

        def fit(self, x, y, eval_set=None, verbose=False):
            return self

        def predict(self, x):
            return x[:, 0].astype(float)

    monkeypatch.setitem(sys.modules, "xgboost", types.SimpleNamespace(XGBRegressor=FakeModel))
    X = np.column_stack((np.arange(20, dtype=float), np.arange(20, dtype=float) ** 2))
    y = (X[:, 0] % 3).astype(float)
    dist = xgboost_distances_r2(X, y, random_state=0, max_estimators=3, early_stopping_rounds=1)
    assert dist.shape == (2, 2)
    assert_allclose(np.diag(dist), np.zeros(2))
    assert np.all((dist >= 0.0) & (dist <= 1.0))


def test_xgboost_distances_low_signal_warning_path(monkeypatch):
    """Low-signal predictor path warns and still returns correctly shaped distances."""
    import sys
    import types

    class FakeModel:
        def __init__(self, **kwargs):
            return

        def fit(self, x, y, eval_set=None, verbose=False):
            return self

        def predict(self, x):
            return np.zeros(x.shape[0], dtype=float)

    monkeypatch.setitem(sys.modules, "xgboost", types.SimpleNamespace(XGBRegressor=FakeModel))
    X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]], dtype=float)
    y = np.array([1.0, 0.0, 1.0, 0.0], dtype=float)
    with pytest.warns(UserWarning, match="No/low signal found from feature"):
        dist = xgboost_distances_r2(X, y, random_state=0, max_estimators=3, early_stopping_rounds=1)
    assert dist.shape == (2, 2)


def test_pt_shuffle_rec_pyfunc_recursive_left_and_right_branches():
    """py_func recursion covers both random branch orderings in _pt_shuffle_rec."""
    tree = _example_linkage_tree()
    index_mask = np.array([True, True, True], dtype=bool)
    indexes = np.empty(3, dtype=np.int64)

    np.random.seed(42)  # first randn >= 0 for this tree depth
    end_pos_pos = _pt_shuffle_rec.py_func(1, indexes, index_mask, tree, M=3, pos=0)
    assert end_pos_pos == 3
    assert set(indexes.tolist()) == {0, 1, 2}

    np.random.seed(2)  # first randn < 0, forcing opposite branch order
    end_pos_neg = _pt_shuffle_rec.py_func(1, indexes, index_mask, tree, M=3, pos=0)
    assert end_pos_neg == 3
    assert set(indexes.tolist()) == {0, 1, 2}


def test_hclust_auto_without_target_uses_cosine_metric():
    """metric='auto' with y=None follows the cosine-distance code path."""
    np.random.seed(42)
    X = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 5.0, 7.0], [4.0, 6.0, 8.0]])
    out = hclust(X, y=None, linkage="single", metric="auto")
    assert out.shape == (2, 4)

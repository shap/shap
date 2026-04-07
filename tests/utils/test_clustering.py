"""Focused tests for ``shap.utils._clustering``.

These tests cover core public behavior and numba helper correctness while
keeping scope compact and deterministic.
"""

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


class TestNumbaHelpers:
    def test_mask_delta_score_and_reverse_window(self):
        """Basic XOR score and in-place window reversal work as expected."""
        m1 = np.array([1, 0, 1, 0], dtype=bool)
        m2 = np.array([0, 0, 1, 1], dtype=bool)
        assert _mask_delta_score(m1, m2) == 2

        order = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        _reverse_window(order, start=1, length=3)
        assert_array_equal(order, np.array([0, 3, 2, 1, 4], dtype=np.int64))
        _reverse_window.py_func(order, start=0, length=2)

    def test_reverse_window_score_gain_and_py_func_paths(self):
        """Score gain and py_func fallbacks execute and return expected values."""
        masks = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=np.bool_)
        order = np.array([0, 1, 2, 3], dtype=np.int64)
        assert _reverse_window_score_gain(masks, order, start=1, length=2) == -2
        assert isinstance(_reverse_window_score_gain.py_func(masks, order, 1, 2), (int, np.integer))
        assert _mask_delta_score.py_func(np.array([True]), np.array([False])) == 1

    def test_delta_minimization_order_core_and_positive_gain(self):
        """Order is a permutation and py_func branch can reverse when gain > 0."""
        all_masks = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.bool_)
        out = delta_minimization_order(all_masks, max_swap_size=4, num_passes=2)
        assert set(out.tolist()) == {0, 1, 2, 3}

        pos_gain_masks = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 1]], dtype=np.bool_)
        py_out = delta_minimization_order.py_func(pos_gain_masks, max_swap_size=4, num_passes=1)
        assert_array_equal(py_out, np.array([0, 2, 1, 3], dtype=np.int64))
        assert delta_minimization_order(np.empty((0, 3), dtype=np.bool_)).shape == (0,)

    def test_pt_shuffle_rec_branches_and_wrapper(self):
        """Recursive shuffle covers leaf include/exclude and wrapper behavior."""
        tree = _example_linkage_tree()

        # Leaf excluded branch.
        excluded_mask = np.array([True, True, False], dtype=bool)
        indexes = np.full(2, -1, dtype=np.int64)
        assert _pt_shuffle_rec.py_func(-1, indexes, excluded_mask, tree, M=3, pos=0) == 0

        # Leaf included branch.
        included_mask = np.array([True, True, True], dtype=bool)
        indexes = np.full(3, -1, dtype=np.int64)
        assert _pt_shuffle_rec.py_func(-1, indexes, included_mask, tree, M=3, pos=0) == 1
        assert indexes[0] == 2

        # Recursive branch (order can vary with randomness).
        np.random.seed(42)
        indexes = np.empty(3, dtype=np.int64)
        end_pos = _pt_shuffle_rec(1, indexes, included_mask, tree, M=3, pos=0)
        assert end_pos == 3
        end_pos_py = _pt_shuffle_rec.py_func(1, indexes, included_mask, tree, M=3, pos=0)
        assert end_pos_py == 3
        np.random.seed(2)  # first randn < 0, covers left-then-right branch
        end_pos_py_neg = _pt_shuffle_rec.py_func(1, indexes, included_mask, tree, M=3, pos=0)
        assert end_pos_py_neg == 3

        # Public wrapper path.
        np.random.seed(42)
        wrapper_mask = np.array([True, True, False], dtype=bool)
        out = np.empty(wrapper_mask.sum(), dtype=np.int64)
        partition_tree_shuffle(out, wrapper_mask, tree)
        assert set(out.tolist()) == {0, 1}


class TestPublicClusteringAPI:
    def test_partition_tree_and_hclust_ordering(self):
        """partition_tree and hclust_ordering return valid clustering outputs."""
        np.random.seed(42)
        X_df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": [2.0, 3.0, 4.0, 5.0], "c": [5.0, 1.0, 2.0, 3.0]})
        clustering = partition_tree(X_df, metric="correlation")
        assert clustering.shape == (2, 4)

        order = hclust_ordering(np.array([[0.0], [1.0], [3.0], [10.0]]), metric="sqeuclidean")
        assert set(order.tolist()) == {0, 1, 2, 3}

        with pytest.raises(ValueError, match="empty distance matrix"):
            hclust_ordering(np.array([[1.0, 2.0]]))

    @pytest.mark.parametrize("linkage", ["single", "complete", "average"])
    def test_hclust_auto_without_target(self, linkage):
        """hclust auto metric without y runs and returns feature linkage."""
        np.random.seed(42)
        X = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 5.0, 7.0], [4.0, 6.0, 8.0]])
        out = hclust(X, y=None, linkage=linkage, metric="auto")
        assert out.shape == (2, 4)

    def test_hclust_validation_and_warning_paths(self):
        """hclust validates shape/linkage and warns when y is ignored."""
        with pytest.raises(DimensionError):
            hclust(np.arange(6), random_state=0)

        with pytest.raises(ValueError, match="Unknown linkage type"):
            hclust(np.array([[1.0, 2.0], [2.0, 3.0]]), linkage="random-string", random_state=0)  # type: ignore[arg-type]

        X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 8.0], [4.0, 16.0]])
        y = np.array([0, 1, 0, 1])
        with pytest.warns(UserWarning, match="Ignoring the y argument"):
            out = hclust(X, y=y, metric="cosine", linkage="single")
        assert out.shape == (1, 4)

    def test_hclust_dataframe_and_stubbed_xgboost_path(self, monkeypatch):
        """DataFrame path and metric='auto' with y use expected branches."""
        np.random.seed(42)
        X_df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 3.0, 4.0]})
        out_df = hclust(X_df, y=None, linkage="single", metric="cosine")
        assert out_df.shape == (1, 4)

        dist_full = np.array([[0.0, 0.1, 0.9], [0.6, 0.0, 0.2], [0.4, 0.8, 0.0]])

        def fake_xgb_distances(_x, _y, random_state=0):
            return dist_full

        from shap.utils import _clustering as clustering_mod

        monkeypatch.setattr(clustering_mod, "xgboost_distances_r2", fake_xgb_distances)
        X = np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [2.0, 3.0, 4.0]])
        y = np.array([0.0, 1.0, 0.0])

        c_single = hclust(X, y=y, metric="auto", linkage="single")
        c_complete = hclust(X, y=y, metric="auto", linkage="complete")
        c_average = hclust(X, y=y, metric="auto", linkage="average")
        assert c_single.shape == c_complete.shape == c_average.shape == (2, 4)
        assert_allclose(c_single[:, 2], np.sort(c_single[:, 2]))


class TestXgboostDistances:
    def test_xgboost_distances_non_warning_path(self, monkeypatch):
        """xgboost_distances_r2 returns bounded square matrix on non-constant predictions."""
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

    def test_xgboost_distances_low_signal_warning_path(self, monkeypatch):
        """Low-variance prediction branch warns and still returns correct shape."""
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

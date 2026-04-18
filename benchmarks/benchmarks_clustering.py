"""Benchmarks for clustering utility functions.

Numba reference implementations are copied from shap/utils/_clustering.py
for migration validation. They will be removed once numba is fully eliminated.

Input sizes are anchored to real SHAP datasets:
  n_features=4  — iris (4 features)
  n_features=8  — california housing (8 features)
  n_features=13 — adult (13 features)

n_masks is derived as 2**n_features to reflect the exact explainer mask matrix size.
"""

import numpy as np
from asv_runner.benchmarks.mark import skip_benchmark_if
from scipy.cluster.hierarchy import ward
from scipy.spatial.distance import pdist

try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

    def njit(fn):
        return fn


try:
    from shap._cutils import mask_delta_score as _cpp_mask_delta_score

    _HAS_MASK_DELTA_SCORE = True
except ImportError:
    _HAS_MASK_DELTA_SCORE = False

try:
    from shap._cutils import reverse_window as _cpp_reverse_window

    _HAS_REVERSE_WINDOW = True
except ImportError:
    _HAS_REVERSE_WINDOW = False

try:
    from shap._cutils import reverse_window_score_gain as _cpp_reverse_window_score_gain

    _HAS_REVERSE_WINDOW_SCORE_GAIN = True
except ImportError:
    _HAS_REVERSE_WINDOW_SCORE_GAIN = False

try:
    from shap._cutils import delta_minimization_order as _cpp_delta_minimization_order

    _HAS_DELTA_MINIMIZATION_ORDER = True
except ImportError:
    _HAS_DELTA_MINIMIZATION_ORDER = False

try:
    from shap._cutils import pt_shuffle_rec as _cpp_pt_shuffle_rec

    _HAS_PT_SHUFFLE_REC = True
except ImportError:
    _HAS_PT_SHUFFLE_REC = False


# --- Numba reference implementations ---


@njit
def _mask_delta_score(m1, m2):
    return (m1 ^ m2).sum()


@njit
def _reverse_window(order, start, length):
    for i in range(length // 2):
        tmp = order[start + i]
        order[start + i] = order[start + length - i - 1]
        order[start + length - i - 1] = tmp


@njit
def _reverse_window_score_gain(masks, order, start, length):
    forward_score = _mask_delta_score(masks[order[start - 1]], masks[order[start]]) + _mask_delta_score(
        masks[order[start + length - 1]], masks[order[start + length]]
    )
    reverse_score = _mask_delta_score(masks[order[start - 1]], masks[order[start + length - 1]]) + _mask_delta_score(
        masks[order[start]], masks[order[start + length]]
    )
    return forward_score - reverse_score


@njit
def _delta_minimization_order(all_masks, max_swap_size=100, num_passes=2):
    order = np.arange(len(all_masks))
    for _ in range(num_passes):
        for length in range(2, max_swap_size):
            for i in range(1, len(order) - length):
                if _reverse_window_score_gain(all_masks, order, i, length) > 0:
                    _reverse_window(order, i, length)
    return order


@njit
def _pt_shuffle_rec(i, indexes, index_mask, partition_tree, M, pos):
    if i < 0:
        if index_mask[i + M]:
            indexes[pos] = i + M
            return pos + 1
        else:
            return pos
    left = int(partition_tree[i, 0] - M)
    right = int(partition_tree[i, 1] - M)
    if np.random.randn() < 0:
        pos = _pt_shuffle_rec(left, indexes, index_mask, partition_tree, M, pos)
        pos = _pt_shuffle_rec(right, indexes, index_mask, partition_tree, M, pos)
    else:
        pos = _pt_shuffle_rec(right, indexes, index_mask, partition_tree, M, pos)
        pos = _pt_shuffle_rec(left, indexes, index_mask, partition_tree, M, pos)
    return pos


# --- Benchmarks ---


class BenchmarkMaskDeltaScore:
    params = [4, 8, 13]
    param_names = ["n_features"]

    def setup(self, n_features):
        rng = np.random.default_rng(0)
        self.m1 = rng.integers(0, 2, n_features).astype(bool)
        self.m2 = rng.integers(0, 2, n_features).astype(bool)
        _mask_delta_score(self.m1, self.m2)  # force JIT compilation

    @skip_benchmark_if(not _HAS_NUMBA)
    def time_numba(self, n_features):
        _mask_delta_score(self.m1, self.m2)

    @skip_benchmark_if(not _HAS_MASK_DELTA_SCORE)
    def time_cpp(self, n_features):
        _cpp_mask_delta_score(self.m1, self.m2)


class BenchmarkReverseWindow:
    params = [4, 8, 13]
    param_names = ["n_features"]

    def setup(self, n_features):
        n_masks = 2**n_features
        rng = np.random.default_rng(0)
        self.order = rng.permutation(n_masks).astype(np.int64)
        self.length = n_masks // 2
        _reverse_window(self.order.copy(), 1, self.length)  # force JIT compilation

    @skip_benchmark_if(not _HAS_NUMBA)
    def time_numba(self, n_features):
        _reverse_window(self.order.copy(), 1, self.length)

    @skip_benchmark_if(not _HAS_REVERSE_WINDOW)
    def time_cpp(self, n_features):
        _cpp_reverse_window(self.order.copy(), 1, self.length)


class BenchmarkReverseWindowScoreGain:
    params = [4, 8, 13]
    param_names = ["n_features"]

    def setup(self, n_features):
        n_masks = 2**n_features
        rng = np.random.default_rng(0)
        self.masks = rng.integers(0, 2, (n_masks, n_features)).astype(bool)
        self.order = rng.permutation(n_masks).astype(np.int64)
        self.length = n_masks // 2
        _reverse_window_score_gain(self.masks, self.order, 1, self.length)  # force JIT

    @skip_benchmark_if(not _HAS_NUMBA)
    def time_numba(self, n_features):
        _reverse_window_score_gain(self.masks, self.order, 1, self.length)

    @skip_benchmark_if(not _HAS_REVERSE_WINDOW_SCORE_GAIN)
    def time_cpp(self, n_features):
        _cpp_reverse_window_score_gain(self.masks, self.order, 1, self.length)


class BenchmarkDeltaMinimizationOrder:
    params = [4, 8, 13]
    param_names = ["n_features"]

    def setup(self, n_features):
        n_masks = 2**n_features
        rng = np.random.default_rng(0)
        self.masks = rng.integers(0, 2, (n_masks, n_features)).astype(bool)
        _delta_minimization_order(self.masks)  # force JIT compilation

    @skip_benchmark_if(not _HAS_NUMBA)
    def time_numba(self, n_features):
        _delta_minimization_order(self.masks)

    @skip_benchmark_if(not _HAS_DELTA_MINIMIZATION_ORDER)
    def time_cpp(self, n_features):
        _cpp_delta_minimization_order(self.masks)


class BenchmarkPtShuffleRec:
    params = [4, 8, 13]
    param_names = ["n_features"]

    def setup(self, n_features):
        rng = np.random.default_rng(0)
        data = rng.random((n_features, n_features))
        self.partition_tree = ward(pdist(data)).astype(np.float64)
        self.M = n_features
        self.index_mask = np.ones(n_features, dtype=bool)
        self.indexes = np.zeros(n_features, dtype=np.int64)
        # force JIT compilation
        _pt_shuffle_rec(
            self.partition_tree.shape[0] - 1,
            self.indexes.copy(),
            self.index_mask,
            self.partition_tree,
            self.M,
            0,
        )

    @skip_benchmark_if(not _HAS_NUMBA)
    def time_numba(self, n_features):
        _pt_shuffle_rec(
            self.partition_tree.shape[0] - 1,
            self.indexes.copy(),
            self.index_mask,
            self.partition_tree,
            self.M,
            0,
        )

    @skip_benchmark_if(not _HAS_PT_SHUFFLE_REC)
    def time_cpp(self, n_features):
        _cpp_pt_shuffle_rec(
            self.partition_tree.shape[0] - 1,
            self.indexes.copy(),
            self.index_mask,
            self.partition_tree,
            self.M,
            0,
        )

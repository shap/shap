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
from numba import njit

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


# --- Benchmarks ---


class BenchmarkMaskDeltaScore:
    params = [4, 8, 13]
    param_names = ["n_features"]

    def setup(self, n_features):
        rng = np.random.default_rng(0)
        self.m1 = rng.integers(0, 2, n_features).astype(bool)
        self.m2 = rng.integers(0, 2, n_features).astype(bool)
        _mask_delta_score(self.m1, self.m2)  # force JIT compilation

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

    def time_numba(self, n_features):
        _reverse_window_score_gain(self.masks, self.order, 1, self.length)

    @skip_benchmark_if(not _HAS_REVERSE_WINDOW_SCORE_GAIN)
    def time_cpp(self, n_features):
        _cpp_reverse_window_score_gain(self.masks, self.order, 1, self.length)

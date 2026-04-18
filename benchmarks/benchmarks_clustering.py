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


# --- Numba reference implementations ---


@njit
def _mask_delta_score(m1, m2):
    return (m1 ^ m2).sum()


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

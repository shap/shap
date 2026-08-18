"""Test script: Compare C++ nanobind output vs original Numba output."""

import time

# Import both implementations
import _clustering_cpp as cpp
import numpy as np

from shap.utils._clustering import (
    _mask_delta_score,
    _reverse_window,
    _reverse_window_score_gain,
    delta_minimization_order,
)

print("=" * 60)
print("  C++ vs Numba Correctness & Speed Test")
print("=" * 60)

passed = 0
failed = 0


def check(name, cpp_val, numba_val):
    global passed, failed
    if np.array_equal(cpp_val, numba_val):
        print(f"  ✅ {name}: PASS  (cpp={cpp_val}, numba={numba_val})")
        passed += 1
    else:
        print(f"  ❌ {name}: FAIL  (cpp={cpp_val}, numba={numba_val})")
        failed += 1


# --- 1. _mask_delta_score ---
print("\n[1] _mask_delta_score")
m1 = np.array([1, 0, 1, 0], dtype=np.int64)
m2 = np.array([1, 1, 0, 0], dtype=np.int64)
check("partial diff", cpp._mask_delta_score(m1, m2), _mask_delta_score(m1, m2))

m_same = np.array([1, 1, 1, 1], dtype=np.int64)
check("identical", cpp._mask_delta_score(m_same, m_same), _mask_delta_score(m_same, m_same))

m_empty = np.array([], dtype=np.int64)
check("empty", cpp._mask_delta_score(m_empty, m_empty), _mask_delta_score(m_empty, m_empty))

# --- 2. _reverse_window ---
print("\n[2] _reverse_window")
o_cpp = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
o_num = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
cpp._reverse_window(o_cpp, 1, 3)
_reverse_window(o_num, 1, 3)
check("mid reverse", list(o_cpp), list(o_num))

o_cpp2 = np.array([10, 20, 30, 40], dtype=np.int64)
o_num2 = np.array([10, 20, 30, 40], dtype=np.int64)
cpp._reverse_window(o_cpp2, 2, 1)
_reverse_window(o_num2, 2, 1)
check("length=1 noop", list(o_cpp2), list(o_num2))

# --- 3. _reverse_window_score_gain ---
print("\n[3] _reverse_window_score_gain")
masks = np.array(
    [
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0],
    ],
    dtype=np.int64,
)
order = np.arange(5, dtype=np.int64)
cpp_gain = cpp._reverse_window_score_gain(masks, order, 1, 2)
numba_gain = _reverse_window_score_gain(masks, order, 1, 2)
check("alternating gain", cpp_gain, numba_gain)

# --- 4. delta_minimization_order ---
print("\n[4] delta_minimization_order")
masks2 = np.array(
    [
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0],
    ],
    dtype=np.int64,
)
cpp_order = cpp.delta_minimization_order(masks2, 4, 2)
numba_order = delta_minimization_order(masks2, 4, 2)
check("same ordering", list(cpp_order), list(numba_order))

# Check it's a valid permutation
check("valid permutation", sorted(cpp_order), [0, 1, 2, 3, 4])

# --- Speed Benchmark ---
print("\n" + "=" * 60)
print("  Speed Benchmark (1000 iterations)")
print("=" * 60)

big_masks = np.random.randint(0, 2, size=(50, 20), dtype=np.int64)

# Numba warmup
_ = delta_minimization_order(big_masks, 10, 1)

t0 = time.perf_counter()
for _ in range(1000):
    delta_minimization_order(big_masks, 10, 1)
numba_time = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(1000):
    cpp.delta_minimization_order(big_masks, 10, 1)
cpp_time = time.perf_counter() - t0

print(f"  Numba: {numba_time:.3f}s")
print(f"  C++:   {cpp_time:.3f}s")
print(f"  Speedup: {numba_time / cpp_time:.1f}x")

# --- Final Report ---
print("\n" + "=" * 60)
total = passed + failed
print(f"  RESULTS: {passed}/{total} passed, {failed}/{total} failed")
if failed == 0:
    print("  🎉 ALL TESTS PASSED! C++ matches Numba perfectly!")
else:
    print("  ⚠️  Some tests failed — review needed.")
print("=" * 60)

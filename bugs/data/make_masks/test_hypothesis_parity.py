"""Hypothesis parity checks for the make_masks numba -> nanobind migration.

Compares ``shap.utils.make_masks`` (the C++ `_init_masks`/`_rec_fill_masks`
bindings) against the pre-migration numba algorithm from master @ f290d210,
ported verbatim to pure Python below. Every generated clustering is checked
exactly: dense mask values plus the CSR ``indptr``/``indices`` internals.

The clustering strategy builds a valid scipy-linkage-style matrix from a
random merge order, so degenerate chains, balanced trees, and everything in
between all come up.

Run with:

    uv run pytest bugs/data/make_masks/test_hypothesis_parity.py -q

``hypothesis`` is not a project dependency (install with
``uv pip install hypothesis``); the module skips cleanly without it.
"""

import sys

import numpy as np
import pytest
import scipy.sparse

pytest.importorskip("hypothesis")
from hypothesis import given, settings
from hypothesis import strategies as st

from shap.utils._masked_model import _MAX_CLUSTERING_DEPTH, make_masks

# --- pure-Python port of the numba baseline (master @ f290d210) -------------


def _init_masks_ref(cluster_matrix, M, indices_row_pos, indptr):
    pos = 0
    for i in range(2 * M - 1):
        if i < M:
            pos += 1
        else:
            pos += int(cluster_matrix[i - M, 3])
        indptr[i + 1] = pos
        indices_row_pos[i] = indptr[i]


def _rec_fill_masks_ref(cluster_matrix, indices_row_pos, indptr, indices, M, ind):
    pos = indices_row_pos[ind]

    if ind < M:
        indices[pos] = ind
        return

    lind = int(cluster_matrix[ind - M, 0])
    rind = int(cluster_matrix[ind - M, 1])
    lind_size = int(cluster_matrix[lind - M, 3]) if lind >= M else 1
    rind_size = int(cluster_matrix[rind - M, 3]) if rind >= M else 1

    lpos = indices_row_pos[lind]
    rpos = indices_row_pos[rind]

    _rec_fill_masks_ref(cluster_matrix, indices_row_pos, indptr, indices, M, lind)
    indices[pos : pos + lind_size] = indices[lpos : lpos + lind_size]

    _rec_fill_masks_ref(cluster_matrix, indices_row_pos, indptr, indices, M, rind)
    indices[pos + lind_size : pos + lind_size + rind_size] = indices[rpos : rpos + rind_size]


def make_masks_ref(cluster_matrix):
    M = cluster_matrix.shape[0] + 1
    indices_row_pos = np.zeros(2 * M - 1, dtype=np.int64)
    indptr = np.zeros(2 * M, dtype=np.int64)
    indices = np.zeros(int(np.sum(cluster_matrix[:, 3])) + M, dtype=np.int64)

    _init_masks_ref(cluster_matrix, M, indices_row_pos, indptr)
    _rec_fill_masks_ref(cluster_matrix, indices_row_pos, indptr, indices, M, cluster_matrix.shape[0] - 1 + M)
    return scipy.sparse.csr_matrix((np.ones(len(indices), dtype=bool), indices, indptr), shape=(2 * M - 1, M))


# --- strategies -------------------------------------------------------------


@st.composite
def cluster_matrices(draw, max_leaves=128):
    """A valid linkage-style clustering built from a random merge order."""
    M = draw(st.integers(min_value=2, max_value=max_leaves))
    active = list(range(M))
    sizes = [1] * (2 * M - 1)
    rows = np.zeros((M - 1, 4))
    for k in range(M - 1):
        left = active.pop(draw(st.integers(0, len(active) - 1)))
        right = active.pop(draw(st.integers(0, len(active) - 1)))
        sizes[M + k] = sizes[left] + sizes[right]
        distance = draw(st.floats(0, 1e6, allow_nan=False))
        rows[k] = [left, right, distance, sizes[M + k]]
        active.append(M + k)
    return rows


# --- properties -------------------------------------------------------------


@settings(max_examples=200, deadline=None)
@given(cluster_matrices())
def test_make_masks_matches_numba_baseline(cluster_matrix):
    got = make_masks(cluster_matrix)
    expected = make_masks_ref(cluster_matrix)
    assert np.array_equal(got.indptr, expected.indptr)
    assert np.array_equal(got.indices, expected.indices)
    assert np.array_equal(got.toarray(), expected.toarray())


@settings(max_examples=50, deadline=None)
@given(cluster_matrices(max_leaves=32), st.data())
def test_make_masks_rejects_out_of_range_children(cluster_matrix, data):
    M = cluster_matrix.shape[0] + 1
    row = data.draw(st.integers(0, M - 2))
    col = data.draw(st.integers(0, 1))
    cluster_matrix[row, col] = data.draw(st.sampled_from([-1, 2 * M - 1, 2 * M + 5]))
    with pytest.raises(IndexError):
        make_masks(cluster_matrix)


# --- larger matrices --------------------------------------------------------
#
# Hypothesis drawing every merge choice is too slow at this scale, so it draws
# a seed and M instead and numpy builds the merge order. The reference needs a
# raised interpreter recursion limit; dense comparison is skipped (a dense
# (2M-1, M) matrix at M=20000 is ~800MB) — CSR internals plus the all-ones
# data array pin the result completely.


def _random_linkage(rng, M):
    active = list(range(M))
    sizes = [1] * (2 * M - 1)
    rows = np.zeros((M - 1, 4))
    for k in range(M - 1):
        merged = []
        for _ in range(2):
            i = int(rng.integers(len(active)))
            active[i], active[-1] = active[-1], active[i]
            merged.append(active.pop())
        sizes[M + k] = sizes[merged[0]] + sizes[merged[1]]
        rows[k] = [merged[0], merged[1], float(rng.random()), sizes[M + k]]
        active.append(M + k)
    return rows


def _chain_linkage(M):
    rows = np.zeros((M - 1, 4))
    rows[0] = [0, 1, 0.0, 2]
    for k in range(1, M - 1):
        rows[k] = [M + k - 1, k + 1, 0.0, k + 2]
    return rows


def _balanced_linkage(M):
    """M must be a power of two: merge consecutive pairs level by level."""
    rows = np.zeros((M - 1, 4))
    level = list(range(M))
    sizes = [1] * (2 * M - 1)
    k = 0
    while len(level) > 1:
        next_level = []
        for j in range(0, len(level), 2):
            left, right = level[j], level[j + 1]
            sizes[M + k] = sizes[left] + sizes[right]
            rows[k] = [left, right, 0.0, sizes[M + k]]
            next_level.append(M + k)
            k += 1
        level = next_level
    return rows


def _assert_csr_parity(cluster_matrix, recursion_limit=120_000):
    got = make_masks(cluster_matrix)
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(recursion_limit)
    try:
        expected = make_masks_ref(cluster_matrix)
    finally:
        sys.setrecursionlimit(old_limit)
    assert np.array_equal(got.indptr, expected.indptr)
    assert np.array_equal(got.indices, expected.indices)
    assert np.array_equal(got.data, expected.data)


@settings(max_examples=40, deadline=None)
@given(st.integers(0, 2**32 - 1), st.integers(20_000, 60_000))
def test_make_masks_matches_numba_baseline_large(seed, M):
    _assert_csr_parity(_random_linkage(np.random.default_rng(seed), M))


def test_make_masks_huge_balanced_tree():
    _assert_csr_parity(_balanced_linkage(2**19))


def test_make_masks_deep_chain_below_depth_limit():
    _assert_csr_parity(_chain_linkage(8_000))


def test_make_masks_rejects_chain_past_depth_limit():
    with pytest.raises(ValueError, match="too deep"):
        make_masks(_chain_linkage(_MAX_CLUSTERING_DEPTH + 1))

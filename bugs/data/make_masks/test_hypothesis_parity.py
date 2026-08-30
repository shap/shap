"""Hypothesis parity checks for the make_masks numba -> nanobind migration.

Compares ``shap.utils.make_masks`` (the C++ `_init_masks`/`_rec_fill_masks`
bindings) against the pre-migration numba algorithm from master @ f290d210,
ported verbatim to pure Python below. Every generated clustering is checked
exactly: dense mask values plus the CSR ``indptr``/``indices`` internals.

The clustering strategy builds a valid scipy-linkage-style matrix from a
random merge order, so degenerate chains, balanced trees, and everything in
between all come up. Per run the properties generate 700+ cases covering:
sizes M = 2..60000 plus a fixed 2^19-leaf tree; shapes from balanced through
caterpillars to chains (both orientations); input dtypes float64/32/16,
int64/32/16, uint32 and object arrays mixing python ints and floats;
fractional child ids (truncation semantics); nan/inf/huge distance columns;
and malformed inputs (out-of-range children, narrow matrices, too-deep
chains) that must raise instead of segfaulting.

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


# --- input dtypes -----------------------------------------------------------
#
# make_masks casts to float64 before calling the bindings, so any dtype whose
# values survive the round trip must give the identical result. Distances are
# zeroed first: they are irrelevant to the result (tested separately) but can
# overflow the narrow dtypes. All node ids and sizes here stay <= 254, exactly
# representable even in float16.

_DTYPES = [np.float64, np.float32, np.float16, np.int64, np.int32, np.int16, np.uint32]


@settings(max_examples=100, deadline=None)
@given(cluster_matrices(), st.sampled_from(_DTYPES))
def test_make_masks_dtype_parity(cluster_matrix, dtype):
    cluster_matrix[:, 2] = 0
    got = make_masks(cluster_matrix.astype(dtype))
    expected = make_masks_ref(cluster_matrix)
    assert np.array_equal(got.indptr, expected.indptr)
    assert np.array_equal(got.indices, expected.indices)
    assert np.array_equal(got.toarray(), expected.toarray())


@settings(max_examples=50, deadline=None)
@given(cluster_matrices())
def test_make_masks_mixed_int_float_object_matrix(cluster_matrix):
    """An object-dtype matrix mixing python ints (ids, sizes) and floats (distances)."""
    mixed = np.empty(cluster_matrix.shape, dtype=object)
    mixed[:, 0] = [int(v) for v in cluster_matrix[:, 0]]
    mixed[:, 1] = [int(v) for v in cluster_matrix[:, 1]]
    mixed[:, 2] = [float(v) for v in cluster_matrix[:, 2]]
    mixed[:, 3] = [int(v) for v in cluster_matrix[:, 3]]
    got = make_masks(mixed)
    expected = make_masks_ref(cluster_matrix)
    assert np.array_equal(got.indptr, expected.indptr)
    assert np.array_equal(got.indices, expected.indices)
    assert np.array_equal(got.toarray(), expected.toarray())


@settings(max_examples=100, deadline=None)
@given(cluster_matrices(), st.integers(0, 2**32 - 1))
def test_make_masks_truncates_fractional_child_ids_like_numba(cluster_matrix, seed):
    """Non-integral child ids truncate (C++ static_cast) exactly like numba's int()."""
    rng = np.random.default_rng(seed)
    cluster_matrix[:, :2] += rng.uniform(0, 0.99, size=(cluster_matrix.shape[0], 2))
    got = make_masks(cluster_matrix)
    expected = make_masks_ref(cluster_matrix)
    assert np.array_equal(got.indptr, expected.indptr)
    assert np.array_equal(got.indices, expected.indices)
    assert np.array_equal(got.toarray(), expected.toarray())


@settings(max_examples=100, deadline=None)
@given(cluster_matrices(), st.integers(0, 2**32 - 1))
def test_make_masks_ignores_distance_column(cluster_matrix, seed):
    """The distance column never influences the masks, even as nan/inf/huge."""
    rng = np.random.default_rng(seed)
    garbage = np.array([np.nan, np.inf, -np.inf, -1e300, 1e300, 0.0, -1.0])
    cluster_matrix[:, 2] = rng.choice(garbage, size=cluster_matrix.shape[0])
    got = make_masks(cluster_matrix)
    expected = make_masks_ref(cluster_matrix)
    assert np.array_equal(got.indptr, expected.indptr)
    assert np.array_equal(got.indices, expected.indices)
    assert np.array_equal(got.toarray(), expected.toarray())


@settings(max_examples=25, deadline=None)
@given(cluster_matrices(max_leaves=16), st.integers(1, 3))
def test_make_masks_rejects_narrow_cluster_matrix(cluster_matrix, ncols):
    """A matrix with fewer than 4 columns errors in Python before the native code."""
    with pytest.raises(IndexError):
        make_masks(cluster_matrix[:, :ncols])


# --- larger matrices --------------------------------------------------------
#
# Hypothesis drawing every merge choice is too slow at this scale, so it draws
# a seed and M instead and numpy builds the merge order. The reference needs a
# raised interpreter recursion limit; dense comparison is skipped (a dense
# (2M-1, M) matrix at M=20000 is ~800MB) — CSR internals plus the all-ones
# data array pin the result completely.


def _random_linkage(rng, M, chain_bias=0.0, swap=False):
    """Random merge order; chain_bias is the probability of re-merging the newest
    cluster (0 = uniform random, 1 = a pure chain), swap flips child order.
    """
    active = list(range(M))
    sizes = [1] * (2 * M - 1)
    rows = np.zeros((M - 1, 4))
    for k in range(M - 1):
        merged = []
        for _ in range(2):
            i = len(active) - 1 if rng.random() < chain_bias else int(rng.integers(len(active)))
            active[i], active[-1] = active[-1], active[i]
            merged.append(active.pop())
        if swap:
            merged.reverse()
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


@settings(max_examples=100, deadline=None)
@given(
    st.integers(0, 2**32 - 1),
    st.integers(100, 3_000),
    st.floats(0.0, 1.0),
    st.booleans(),
)
def test_make_masks_shape_zoo(seed, M, chain_bias, swap):
    """Sweeps tree shape from balanced-ish random to caterpillars to near-chains,
    in both left- and right-leaning orientation.
    """
    _assert_csr_parity(_random_linkage(np.random.default_rng(seed), M, chain_bias, swap))


def test_make_masks_huge_balanced_tree():
    _assert_csr_parity(_balanced_linkage(2**19))


def test_make_masks_deep_chain_below_depth_limit():
    _assert_csr_parity(_chain_linkage(8_000))


def test_make_masks_rejects_chain_past_depth_limit():
    with pytest.raises(ValueError, match="too deep"):
        make_masks(_chain_linkage(_MAX_CLUSTERING_DEPTH + 1))

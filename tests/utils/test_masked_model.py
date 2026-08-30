import multiprocessing
import sys

import numpy as np
import pytest

from shap._cutils import _init_masks  # type: ignore[attr-defined]
from shap.links import identity
from shap.utils._masked_model import _build_fixed_output


def _run_isolated(fn):
    """Run ``fn`` in a forked child process and return its exit code, so a
    segfault (negative exit code) fails the test instead of killing the pytest
    process. 0 means the payload ran to completion, 1 means it raised.

    The fork context is deliberate: spawn re-imports the parent's __main__,
    which under pytest is pytest itself."""
    process = multiprocessing.get_context("fork").Process(target=fn)
    process.start()
    process.join(120)
    if process.is_alive():
        process.kill()
        process.join()
        raise AssertionError("isolated payload timed out")
    return process.exitcode


def test_init_masks():
    """``_init_masks`` declares int64 output arrays but does not mark them
    ``.noconvert()``, so nanobind falls back to its implicit-conversion pass: it
    writes into a temporary cast copy and drops it. The caller gets no exception
    and an untouched array.

    That is the failure mode behind 029be7d8 ("Fix int typing bug"), where the
    binding said int32 while ``make_masks`` passed int64 -- ``indptr`` came back
    all zeros and the mask matrix was silently wrong. Matching the dtypes fixed
    that instance; nothing yet stops the next dtype drift from doing it again.
    """
    # node 4 = {0, 1}, node 5 = {2, 3}, node 6 = {0, 1, 2, 3}
    cluster_matrix = np.array([[0.0, 1.0, 1.0, 2.0], [2.0, 3.0, 1.0, 2.0], [4.0, 5.0, 2.0, 4.0]])
    M = cluster_matrix.shape[0] + 1
    # anything other than the int64 that make_masks happens to allocate today
    indices_row_pos = np.zeros(2 * M - 1, dtype=np.int32)
    indptr = np.zeros(2 * M, dtype=np.int32)

    try:
        _init_masks(cluster_matrix, M, indices_row_pos, indptr)
    except TypeError:
        return  # rejecting the mismatch outright is a fine outcome

    # otherwise the call must have written through to the caller's array
    assert indptr[-1] == int(np.sum(cluster_matrix[:, 3])) + M, (
        "_init_masks accepted int32 arrays, reported success, and wrote nothing"
    )


def test_partition_explainer_with_custom_clustering():
    """Public-interface smoke test for the make_masks path.

    On Windows this currently fails with a TypeError from ``_init_masks``:
    ``make_masks`` allocates its index arrays with ``dtype=int`` (int32 on
    Windows), while the binding demands int64.
    """
    rng = np.random.RandomState(0)
    data = rng.standard_normal((10, 4))
    clustering = np.array([[0.0, 1.0, 1.0, 2.0], [2.0, 3.0, 1.0, 2.0], [4.0, 5.0, 2.0, 4.0]])

    import shap

    masker = shap.maskers.Partition(data, clustering=clustering)
    explainer = shap.PartitionExplainer(lambda x: x.sum(1), masker)
    assert explainer._mask_matrix.shape == (7, 4)
    assert explainer._mask_matrix[6].nnz == 4  # root row selects every feature


def _construct_explainer_with_bad_clustering(explainer_kind):
    import shap

    data = np.zeros((10, 4))
    # the third merge references node 400, which does not exist
    bad = np.array([[0.0, 1.0, 1.0, 2.0], [2.0, 3.0, 1.0, 2.0], [400.0, 5.0, 2.0, 4.0]])
    masker = shap.maskers.Partition(data, clustering=bad)
    try:
        if explainer_kind == "partition":
            shap.PartitionExplainer(lambda x: x.sum(1), masker)
        else:
            shap.CoalitionExplainer(lambda x: x.sum(1), masker, partition_tree={"grp": [0, 1, 2, 3]})
    except (ValueError, IndexError):
        return  # clean rejection is the desired behavior
    raise RuntimeError("malformed clustering was accepted without an error")


def _partition_payload():
    _construct_explainer_with_bad_clustering("partition")


def _coalition_payload():
    _construct_explainer_with_bad_clustering("coalition")


def _deep_chain_payload():
    from shap.utils import make_masks

    M = 100_000
    rows = np.zeros((M - 1, 4))
    prev, size = 0, 1
    for k in range(M - 1):
        rows[k] = [prev, k + 1, k + 1, size + 1]
        prev, size = M + k, size + 1
    mask = make_masks(rows)
    assert mask.shape == (2 * M - 1, M)


@pytest.mark.skipif(sys.platform == "win32", reason="the fork start method is not available on Windows")
def test_partition_explainer_rejects_out_of_range_clustering():
    """A clustering row pointing at a nonexistent node must raise, not crash.

    Reproduces: constructing a PartitionExplainer with a malformed clustering
    currently segfaults inside ``_rec_fill_masks`` (unchecked index read).
    The numba implementation had the same flaw; now that the code is C++ the
    crash should become a Python exception.
    """
    exitcode = _run_isolated(_partition_payload)
    assert exitcode == 0, (
        f"expected a clean Python exception, got exit code {exitcode} (negative = killed by signal, e.g. -11 SIGSEGV)"
    )


@pytest.mark.skipif(sys.platform == "win32", reason="the fork start method is not available on Windows")
def test_coalition_explainer_rejects_out_of_range_clustering():
    """Same as above through CoalitionExplainer, the other public make_masks caller."""
    exitcode = _run_isolated(_coalition_payload)
    assert exitcode == 0, (
        f"expected a clean Python exception, got exit code {exitcode} (negative = killed by signal, e.g. -11 SIGSEGV)"
    )


@pytest.mark.skipif(sys.platform == "win32", reason="the fork start method is not available on Windows")
def test_make_masks_deep_chain_clustering():
    """A valid but fully unbalanced 100k-leaf clustering must not crash.

    ``make_masks``'s docstring says it is "optimized since trees for images can
    be very large"; a degenerate chain of that size currently overflows the C++
    stack (segfault). The numba implementation crashed on the same input, so
    this documents an inherited limit, not a regression.

    ``make_masks`` is exercised directly (it is exported as ``shap.utils.make_masks``)
    because a PartitionExplainer with 100k features is not practical in a test.
    """
    exitcode = _run_isolated(_deep_chain_payload)
    assert exitcode == 0, (
        f"make_masks crashed on a valid deep clustering, exit code {exitcode} (negative = killed by signal)"
    )


def test_make_masks_rejects_narrow_cluster_matrix():
    """The mask-building algorithm reads clustering column 3, so a matrix with
    fewer than 4 columns must be rejected with a clean error through the public
    path (today numpy's own indexing raises before the native code is reached;
    this pins that the error stays a Python exception and never regresses into
    the native layer's unchecked read)."""
    from shap.utils import make_masks

    with pytest.raises((IndexError, ValueError, TypeError)):
        make_masks(np.zeros((3, 3)))


def test__build_fixed_output():
    """GH3651"""
    num_varying_rows = np.array([1])
    varying_rows = np.array([[True]])
    batch_positions = np.array([0, 1])
    averaged_outs = np.zeros((1, 10), dtype=np.float32)
    last_outs = np.zeros((1, 10), dtype=np.float32)
    outputs = np.random.rand(1, 10).astype(np.float16)
    _build_fixed_output(
        averaged_outs, last_outs, outputs, batch_positions, varying_rows, num_varying_rows, identity, None
    )
    assert np.allclose(averaged_outs, outputs, 1e-2)

import numpy as np
from shap._cutils import _init_masks  # type: ignore[attr-defined]

from shap.links import identity
from shap.utils._masked_model import _build_fixed_output


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

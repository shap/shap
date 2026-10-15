import numpy as np
import pytest
from shap._cutils import _init_masks  # type: ignore[attr-defined]

from shap.links import identity, logit
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


def test_empty_first_batch_wraparound_carry():
    """An empty first batch (num_varying_rows[0] == 0) has no previous result to
    carry, so the numba baseline's ``averaged_outs[i - 1]`` wrapped around and
    read the *incoming* last element of ``averaged_outs`` — an arbitrary value,
    not a real result. The native port replicates that wraparound literally, so
    identity links and weighted links are bit-exact with numba even here.

    The one intentional divergence: with an unweighted non-identity link the
    carried value now passes through the trailing vectorized link application,
    so logit turns the carried 0.0 into ``logit(0) = -inf`` where numba left
    the raw 0.0. Both are garbage (the batch never produced an output); this
    pins the flavor of garbage so a change is deliberate. Unreachable from
    MaskedModel, whose first mask always evaluates every row.
    """
    batch_positions = np.array([0, 0, 2], dtype=np.int64)
    varying_rows = np.array([[False, False], [True, True]])
    num_varying_rows = varying_rows.sum(axis=1).astype(np.int64)
    outputs = np.array([0.25, 0.75])

    averaged_outs = np.zeros(2)
    last_outs = np.zeros(2)
    _build_fixed_output(
        averaged_outs, last_outs, outputs, batch_positions, varying_rows, num_varying_rows, identity, None
    )
    assert averaged_outs[0] == 0.0  # numba-compatible wraparound read of the initial last element
    assert averaged_outs[1] == 0.5

    averaged_outs = np.zeros(2)
    last_outs = np.zeros(2)
    with np.errstate(divide="ignore"):
        _build_fixed_output(
            averaged_outs, last_outs, outputs, batch_positions, varying_rows, num_varying_rows, logit, None
        )
    assert averaged_outs[0] == -np.inf  # numba left the raw 0.0 here
    assert np.isfinite(averaged_outs[1])


@pytest.mark.parametrize("output_count", [None, 3], ids=["single-output", "multi-output"])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("weighted", [False, True])
def test_build_fixed_output_matches_numpy(output_count, dtype, weighted):
    """The native implementation carries forward rows and applies links in the original order."""
    sample_count = 3
    batch_positions = np.array([0, 3, 5, 5, 6])
    varying_rows = np.array(
        [
            [True, True, True],
            [True, False, True],
            [False, False, False],
            [False, True, False],
        ]
    )
    num_varying_rows = varying_rows.sum(axis=1)
    output_shape = () if output_count is None else (output_count,)
    outputs = np.linspace(0.15, 0.85, 6 * (output_count or 1), dtype=dtype).reshape((6,) + output_shape)
    averaged_outs = np.zeros((4,) + output_shape, dtype=np.float32 if dtype == np.float16 else dtype)
    last_outs = np.zeros((sample_count,) + output_shape, dtype=averaged_outs.dtype)

    weights = None
    if weighted:
        weights = np.linspace(0.5, 1.5, sample_count * (output_count or 1)).reshape((sample_count,) + output_shape)

    expected = np.zeros_like(averaged_outs)
    expected_last = np.zeros_like(last_outs)
    for i in range(len(expected)):
        if batch_positions[i] == batch_positions[i + 1]:
            expected[i] = expected[i - 1]
            continue
        expected_last[varying_rows[i]] = outputs[batch_positions[i] : batch_positions[i + 1]]
        if weights is None:
            expected[i] = logit(np.mean(expected_last, axis=0))
        else:
            expected[i] = np.mean(weights * logit(expected_last), axis=0)

    _build_fixed_output(
        averaged_outs,
        last_outs,
        outputs,
        batch_positions,
        varying_rows,
        num_varying_rows,
        logit,
        weights,
    )

    np.testing.assert_allclose(averaged_outs, expected, rtol=2e-3, atol=2e-3)

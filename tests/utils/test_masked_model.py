import numpy as np
import pytest

from shap.links import identity, logit
from shap.utils._masked_model import _build_fixed_output


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

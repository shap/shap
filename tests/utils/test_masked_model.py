import numpy as np

from shap.links import identity
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

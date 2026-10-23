"""Validate GPU binding buffers before entering CUDA, without requiring a GPU."""

import numpy as np
import pytest


@pytest.mark.parametrize("invalid", ["dtype", "shape", "readonly", "strided"])
def test_gpu_buffer_validation(invalid):
    module = pytest.importorskip("shap_cuda._cext_gpu")
    out = np.zeros((2, 2, 1), dtype=np.float32 if invalid == "dtype" else np.float64)
    if invalid == "shape":
        out = out[:, :1, :].copy()
    elif invalid == "readonly":
        out.flags.writeable = False
    elif invalid == "strided":
        out = np.zeros((2, 2, 1, 2))[..., 0]
    before = out.copy()
    children_left = np.array([[1, -1, -1]], dtype=np.int32)
    children_right = np.array([[2, -1, -1]], dtype=np.int32)
    X = np.array([[0.0], [1.0]])
    with pytest.raises(ValueError if invalid == "shape" else TypeError):
        module.dense_tree_shap(
            children_left,
            children_right,
            children_left,
            np.array([[0, -1, -1]], dtype=np.int32),
            np.array([[0.5, 0.0, 0.0]]),
            np.zeros((1, 3), dtype=np.int32),
            np.array([[[3.0], [1.0], [5.0]]]),
            np.array([[2.0, 1.0, 1.0]]),
            1,
            X,
            np.zeros_like(X, dtype=bool),
            None,
            None,
            None,
            1,
            np.zeros(1),
            out,
            1,
            0,
            False,
        )
    np.testing.assert_array_equal(out, before)

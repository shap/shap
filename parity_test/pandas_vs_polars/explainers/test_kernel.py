"""KernelExplainer dataframe input parity, based on tests/explainers/test_kernel.py."""

import numpy as np

import shap
from parity_test import _utils


def test_kernel_explainer_dataframe_inputs_match():
    pandas_frame, polars_frame, _ = _utils.make_numeric_pandas_polars_frames(n_rows=12)
    coefficients = np.array([1.0, -0.5, 0.25])

    def model(x):
        return np.asarray(x) @ coefficients

    explanations = []
    shap_values = []

    for frame in (pandas_frame, polars_frame):
        explainer = shap.KernelExplainer(model, frame, seed=0)
        explanations.append(explainer(frame[:2], nsamples=20))
        shap_values.append(explainer.shap_values(frame[:2], nsamples=20))

    _utils.assert_explanations_equal(*explanations)
    np.testing.assert_array_equal(shap_values[0], shap_values[1])

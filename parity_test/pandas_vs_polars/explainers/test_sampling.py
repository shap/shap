"""SamplingExplainer dataframe input parity, based on tests/explainers/test_sampling.py."""

import numpy as np

import shap
from parity_test import _utils


def test_sampling_explainer_dataframe_inputs_match():
    pandas_frame, polars_frame, _ = _utils.make_numeric_pandas_polars_frames(n_rows=12)
    coefficients = np.array([1.0, -0.5, 0.25])

    def model(x):
        return np.asarray(x) @ coefficients

    explanations = []

    for frame in (pandas_frame, polars_frame):
        np.random.seed(0)
        explainer = shap.SamplingExplainer(model, frame)
        explanations.append(explainer(frame[:2], nsamples=20))

    _utils.assert_explanations_equal(*explanations)

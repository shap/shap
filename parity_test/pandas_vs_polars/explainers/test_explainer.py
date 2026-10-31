"""Generic Explainer dataframe input parity, based on tests/explainers/test_explainer.py."""

import numpy as np

import shap
from parity_test import _utils


def test_generic_explainer_dataframe_inputs_match():
    pandas_frame, polars_frame, target = _utils.make_numeric_pandas_polars_frames()
    coefficients = np.array([1.0, -0.5, 0.25])

    def model(x):
        return np.asarray(x) @ coefficients

    explanations = []

    for frame in (pandas_frame, polars_frame):
        explainer = shap.Explainer(model, frame, algorithm="permutation", seed=0)
        explanations.append(explainer(frame, max_evals=20))

    _utils.assert_explanations_equal(*explanations)

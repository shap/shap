"""CoalitionExplainer dataframe input parity, based on tests/explainers/test_coalition.py."""

import numpy as np

import shap
from parity_test import _utils


def test_coalition_explainer_dataframe_inputs_match():
    pandas_frame, polars_frame, _ = _utils.make_numeric_pandas_polars_frames(n_rows=12)
    coefficients = np.array([1.0, -0.5, 0.25])

    def model(x):
        return np.asarray(x) @ coefficients

    explanations = []

    for frame in (pandas_frame, polars_frame):
        masker = shap.maskers.Partition(frame)
        explainer = shap.CoalitionExplainer(model, masker)
        explanations.append(explainer(frame[:1], max_evals=20))

    _utils.assert_explanations_equal(*explanations)

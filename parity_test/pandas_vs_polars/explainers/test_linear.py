"""LinearExplainer dataframe input parity, based on tests/explainers/test_linear.py."""

import numpy as np
from sklearn.linear_model import LinearRegression

import shap
from parity_test import _utils


def test_linear_explainer_dataframe_inputs_match():
    pandas_frame, polars_frame, target = _utils.make_numeric_pandas_polars_frames(n_rows=24)
    model = LinearRegression().fit(pandas_frame, target)
    explanations = []
    shap_values = []

    for background, frame in ((pandas_frame, pandas_frame), (polars_frame, polars_frame)):
        explainer = shap.LinearExplainer(model, background)
        explanations.append(explainer(frame[:5]))
        shap_values.append(explainer.shap_values(frame[:5]))

    _utils.assert_explanations_equal(*explanations)
    np.testing.assert_array_equal(shap_values[0], shap_values[1])

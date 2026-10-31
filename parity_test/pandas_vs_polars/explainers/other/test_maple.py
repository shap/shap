"""MAPLE dataframe input parity, based on tests/explainers/other/test_maple.py."""

import numpy as np

from parity_test import _utils
from shap.explainers.other._maple import Maple


def test_maple_dataframe_inputs_match():
    pandas_frame, polars_frame, target = _utils.make_numeric_pandas_polars_frames(n_rows=20)
    coefficients = np.array([1.0, -0.5, 0.25])

    def model(x):
        return np.asarray(x) @ coefficients

    values = []

    for frame in (pandas_frame, polars_frame):
        explainer = Maple(model, frame)
        values.append(explainer.attributions(frame[:3]))

    np.testing.assert_array_equal(values[0], values[1])

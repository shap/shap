"""LIME wrapper dataframe input parity, based on shap.explainers.other._lime."""

import numpy as np
import pytest

from parity_test import _utils
from shap.explainers.other._lime import LimeTabular


def test_lime_dataframe_inputs_match():
    pytest.importorskip("lime")
    pandas_frame, polars_frame, target = _utils.make_numeric_pandas_polars_frames(n_rows=20)
    coefficients = np.array([1.0, -0.5, 0.25])

    def model(x):
        return np.asarray(x) @ coefficients

    values = []

    for frame in (pandas_frame, polars_frame):
        explainer = LimeTabular(model, frame, mode="regression")
        values.append(explainer.attributions(frame[:2], nsamples=100))

    np.testing.assert_array_equal(values[0], values[1])

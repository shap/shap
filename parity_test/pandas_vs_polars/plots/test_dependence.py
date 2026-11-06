"""DataFrame feature-input parity for legacy dependence plots."""

import numpy as np

import shap
from parity_test import _utils


def test_dependence_plot_dataframe_inputs_match():
    pandas_frame, polars_frame, _ = _utils.make_numeric_pandas_polars_frames(n_rows=24)
    values = pandas_frame.to_numpy()

    def plot(frame):
        np.random.seed(0)
        shap.dependence_plot("a", values, frame, interaction_index=None, show=False)

    _utils.assert_plot_frames_equal(plot, pandas_frame, polars_frame)


def test_partial_dependence_dataframe_inputs_match():
    pandas_frame, polars_frame, _ = _utils.make_numeric_pandas_polars_frames(n_rows=24)
    coefficients = np.array([1.0, -0.5, 0.25])

    def predict(frame):
        return np.asarray(frame) @ coefficients

    def plot(frame):
        shap.partial_dependence_plot("a", predict, frame, ice=False, show=False)

    _utils.assert_plot_frames_equal(plot, pandas_frame, polars_frame)

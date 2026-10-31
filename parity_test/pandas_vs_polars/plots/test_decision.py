"""DataFrame feature-input parity for plots, based on tests/plots/test_decision.py."""

import numpy as np

import shap
from parity_test import _utils


def test_decision_plot_dataframe_inputs_match():
    pandas_frame, polars_frame, _ = _utils.make_numeric_pandas_polars_frames(n_rows=12)
    values = np.arange(36, dtype=float).reshape(12, 3) / 10

    def plot(frame):
        shap.decision_plot(0.0, values, features=frame, show=False)

    _utils.assert_plot_frames_equal(plot, pandas_frame, polars_frame)

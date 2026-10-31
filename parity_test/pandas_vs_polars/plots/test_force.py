"""DataFrame feature-input parity for force plots, based on tests/plots/test_force.py."""

import numpy as np

import shap
from parity_test import _utils


def test_force_plot_dataframe_inputs_match():
    pandas_frame, polars_frame, _ = _utils.make_numeric_pandas_polars_frames(n_rows=4)
    values = np.array([0.2, -0.1, 0.3])

    def plot(frame):
        shap.force_plot(0.5, values, frame[:1], matplotlib=True, show=False)

    _utils.assert_plot_frames_equal(plot, pandas_frame, polars_frame)

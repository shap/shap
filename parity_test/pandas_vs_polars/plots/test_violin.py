"""DataFrame feature-input parity for violin plots, based on tests/plots/test_violin.py."""

import numpy as np

import shap
from parity_test import _utils


def test_violin_plot_dataframe_inputs_match():
    pandas_frame, polars_frame, _ = _utils.make_numeric_pandas_polars_frames(n_rows=24)
    values = np.random.default_rng(0).normal(size=pandas_frame.shape)

    def plot(frame):
        shap.plots.violin(values, features=frame, show=False)

    _utils.assert_plot_frames_equal(plot, pandas_frame, polars_frame)

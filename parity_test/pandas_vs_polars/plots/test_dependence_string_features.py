"""String DataFrame feature parity for legacy dependence plots."""

import numpy as np
import pandas as pd
import polars as pl

import shap
from parity_test import _utils


def test_dependence_plot_string_dataframe_inputs_match():
    pandas_frame = pd.DataFrame(
        {
            "Sex": ["Male", "Female", "Male", "Male", "Female", "Female"],
            "Blood group": ["A", "B", "A", "O", "O", "O"],
            "Age": [10, 15, 28, 3, 84, 56],
            "Height": [130, 170, 185, 40, 150, 164],
        }
    )
    polars_frame = pl.DataFrame(pandas_frame.to_dict(orient="list"))
    values = np.random.default_rng(0).normal(size=pandas_frame.shape)

    def plot(frame):
        shap.dependence_plot("Sex", values, frame, interaction_index="Age", show=False)

    _utils.assert_plot_frames_equal(plot, pandas_frame, polars_frame)

"""Tests for shap.plots._partial_dependence."""

import matplotlib

matplotlib.use("Agg")  # headless backend for CI

import numpy as np
import pandas as pd

import shap


def test_partial_dependence_preserves_categorical_dtypes():
    """Regression test for https://github.com/shap/shap/issues/3670.

    ``shap.plots.partial_dependence`` previously reconstructed temporary
    DataFrames as ``pd.DataFrame(arr, columns=...)``, which silently
    promotes ``category`` columns to ``object`` dtype. That breaks any
    model (notably XGBoost with ``enable_categorical=True``) that
    validates column dtypes on the prediction call.

    This test uses a stub predict function that asserts the DataFrame it
    receives still has categorical dtypes where the original input did.
    """
    rs = np.random.RandomState(0)
    n = 50
    df = pd.DataFrame(
        {
            "numeric": rs.randn(n).astype("float64"),
            "cat_small": pd.Categorical(rs.choice(["a", "b", "c"], size=n)),
            "cat_bool": pd.Categorical(rs.choice([True, False], size=n)),
        }
    )

    expected_dtypes = df.dtypes.to_dict()

    seen_frames: list[pd.DataFrame] = []

    def model(frame):
        # The plot function is expected to hand us a DataFrame (because
        # we passed one in). If it downgrades to ndarray the cast below
        # will raise before we get to the dtype check.
        assert isinstance(frame, pd.DataFrame), (
            f"partial_dependence should keep the DataFrame input type; got {type(frame)}"
        )
        seen_frames.append(frame)
        for col, expected in expected_dtypes.items():
            if col == "numeric":
                continue  # this is the column being varied
            assert frame[col].dtype == expected, (
                f"column {col!r} dtype changed from {expected} to {frame[col].dtype} (GH #3670)"
            )
        return np.full(len(frame), 0.5)

    # hist=False avoids a matplotlib histogram path that isn't relevant here.
    shap.plots.partial_dependence(
        "numeric",
        model,
        df,
        ice=False,
        hist=False,
        show=False,
    )

    # Sanity: the model was actually called (otherwise the test proves nothing).
    assert seen_frames, "model callable was never invoked"

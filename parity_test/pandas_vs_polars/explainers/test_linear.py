"""LinearExplainer dataframe input parity, based on tests/explainers/test_linear.py."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

import shap
from parity_test import _utils


def _mean_cov(frame):
    """Background (mean, cov) pair in the same dataframe backend as ``frame``."""
    if hasattr(frame, "to_pandas"):  # polars
        import polars as pl

        values = frame.to_numpy()
        mean = pl.Series("mean", values.mean(0))
        cov = pl.DataFrame(np.cov(values, rowvar=False), schema=frame.columns)
        return mean, cov
    return frame.mean(), frame.cov()


BACKGROUNDS = {
    "frame": lambda frame: frame,
    "independent": lambda frame: shap.maskers.Independent(frame),
    "impute": lambda frame: shap.maskers.Impute(frame),
    "mean_cov": _mean_cov,
}


@pytest.mark.parametrize("background", sorted(BACKGROUNDS))
def test_linear_explainer_dataframe_inputs_match(background):
    pandas_frame, polars_frame, target = _utils.make_numeric_pandas_polars_frames(n_rows=24)
    model = LinearRegression().fit(pandas_frame, target)
    explanations = []
    shap_values = []

    for frame in (pandas_frame, polars_frame):
        np.random.seed(0)  # correlation_dependent (Impute) samples when estimating its transforms
        explainer = shap.LinearExplainer(model, BACKGROUNDS[background](frame))
        explanations.append(explainer(frame[:5]))
        shap_values.append(explainer.shap_values(frame[:5]))

    _utils.assert_explanations_equal(*explanations)
    np.testing.assert_array_equal(shap_values[0], shap_values[1])
    assert list(explanations[1].feature_names) == list(pandas_frame.columns)


def test_linear_explainer_series_input_matches():
    pandas_frame, polars_frame, target = _utils.make_numeric_pandas_polars_frames(n_rows=24)
    model = LinearRegression().fit(pandas_frame, target)

    pandas_values = shap.LinearExplainer(model, pandas_frame).shap_values(pandas_frame.iloc[0])
    polars_values = shap.LinearExplainer(model, polars_frame).shap_values(pl_row(polars_frame))

    np.testing.assert_array_equal(pandas_values, polars_values)


def pl_row(frame):
    """First row of a polars frame as a Series, mirroring ``pandas.DataFrame.iloc[0]``."""
    import polars as pl

    return pl.Series("row", frame.row(0))


def test_generic_explainer_dispatches_to_linear_for_polars():
    pandas_frame, polars_frame, target = _utils.make_numeric_pandas_polars_frames(n_rows=24)
    model = LinearRegression().fit(pandas_frame, target)
    explanations = []

    for frame in (pandas_frame, polars_frame):
        # pass an explicit masker: wrapping a raw dataframe in Explainer.__init__ is handled in #5289
        explainer = shap.Explainer(model, shap.maskers.Independent(frame))
        assert isinstance(explainer, shap.explainers.LinearExplainer)
        explanations.append(explainer(frame[:5]))

    _utils.assert_explanations_equal(*explanations)

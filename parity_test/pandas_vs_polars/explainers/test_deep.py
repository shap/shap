"""DeepExplainer dataframe input parity, based on tests/explainers/test_deep.py."""

import pytest

import shap
from parity_test import _utils


def test_deep_explainer_dataframe_inputs_match():
    tf = pytest.importorskip("tensorflow")
    pandas_frame, polars_frame, _ = _utils.make_numeric_pandas_polars_frames(n_rows=8)
    tf.keras.utils.set_random_seed(0)
    model = tf.keras.Sequential(
        [tf.keras.layers.Input(shape=(3,)), tf.keras.layers.Dense(1, use_bias=True, activation="linear")]
    )
    explanations = []

    for frame in (pandas_frame, polars_frame):
        explainer = shap.DeepExplainer(model, frame[:4])
        explanations.append(explainer(frame[4:]))

    _utils.assert_explanations_equal(*explanations)

"""Pandas/Polars parity checks for TreeExplainer dataframe inputs."""

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor

import shap
from parity_test import _utils

hypothesis = pytest.importorskip("hypothesis")


@hypothesis.settings(max_examples=1000, deadline=None, derandomize=True)
@hypothesis.given(random_state=hypothesis.strategies.integers(min_value=0, max_value=2**32 - 1))
def test_treeexplainer_dataframe_inputs_match(random_state):
    polars = pytest.importorskip("polars")
    rng = np.random.default_rng(random_state)
    values = rng.normal(size=(48, 5))
    columns = [f"feature_{i}" for i in range(values.shape[1])]
    pandas_frame = pd.DataFrame(values, columns=columns)
    polars_frame = polars.DataFrame({name: values[:, i] for i, name in enumerate(columns)})
    target = values[:, 0] - 0.5 * values[:, 1] + values[:, 2] ** 2
    model = DecisionTreeRegressor(max_depth=4, random_state=random_state).fit(pandas_frame, target)

    explanations = []
    for frame in (pandas_frame, polars_frame):
        explainer = shap.TreeExplainer(model)
        explanation = explainer(frame)
        shap_values = explainer.shap_values(frame)
        np.testing.assert_array_equal(shap_values, explanation.values)
        explanations.append(explanation)

    _utils.assert_explanations_equal(*explanations)

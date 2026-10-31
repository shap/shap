"""TreeExplainer dataframe input parity, based on tests/explainers/test_tree.py."""

import numpy as np
import pytest
from sklearn.tree import DecisionTreeRegressor

import shap
from parity_test import _utils

hypothesis = pytest.importorskip("hypothesis")


@hypothesis.settings(max_examples=1000, deadline=None, derandomize=True)
@hypothesis.given(random_state=hypothesis.strategies.integers(min_value=0, max_value=2**32 - 1))
def test_treeexplainer_dataframe_inputs_match(random_state):
    pandas_frame, polars_frame, target = _utils.make_numeric_pandas_polars_frames(random_state, n_rows=48)
    model = DecisionTreeRegressor(max_depth=4, random_state=random_state).fit(pandas_frame, target)

    explanations = []
    shap_values = []
    for frame in (pandas_frame, polars_frame):
        explainer = shap.TreeExplainer(model)
        explanations.append(explainer(frame))
        shap_values.append(explainer.shap_values(frame))

    _utils.assert_explanations_equal(*explanations)
    np.testing.assert_array_equal(shap_values[0], shap_values[1])
    np.testing.assert_array_equal(shap_values[0], explanations[0].values)
    np.testing.assert_array_equal(shap_values[1], explanations[1].values)

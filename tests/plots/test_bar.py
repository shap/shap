"""This file contains tests for the bar plot.
"""
import matplotlib.pyplot as plt
import numpy as np
import pytest

import shap
from shap.utils._exceptions import DimensionError


@pytest.mark.parametrize(
    "unsupported_inputs",
    [
        [1, 2, 3],
        (1, 2, 3),
        np.array([1, 2, 3]),
        {"a": 1, "b": 2},
    ],
)
def test_input_shap_values_type(unsupported_inputs):
    """Check that a TypeError is raised when shap_values is not a valid input type."""
    emsg = (
        "The shap_values argument must be an Explanation object, Cohorts "
        "object, or dictionary of Explanation objects!"
    )
    with pytest.raises(TypeError, match=emsg):
        shap.plots.bar(unsupported_inputs, show=False)


def test_input_shap_values_type_2():
    """Check that a DimensionError is raised if the cohort Explanation objects have different shape."""
    rs = np.random.RandomState(42)
    emsg = (
        "When passing several Explanation objects, they must all have "
        "the same number of feature columns!"
    )
    with pytest.raises(DimensionError, match=emsg):
        shap.plots.bar(
            {
                "t1": shap.Explanation(
                    values=rs.randn(40, 10),
                    base_values=np.ones(40) * 0.5,
                ),
                "t2": shap.Explanation(
                    values=rs.randn(20, 5),
                    base_values=np.ones(20) * 0.5,
                ),
            },
            show=False,
        )


@pytest.mark.mpl_image_compare
def test_simple_bar(explainer):
    """Check that the bar plot is unchanged."""
    shap_values = explainer(explainer.data)
    fig = plt.figure()
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare
def test_simple_bar_with_cohorts_dict():
    """Ensure that bar plots supports dictionary of Explanations as input."""
    rs = np.random.RandomState(42)
    fig = plt.figure()
    shap.plots.bar(
        {
            "t1": shap.Explanation(
                values=rs.randn(40, 5),
                base_values=np.ones(40) * 0.5,
            ),
            "t2": shap.Explanation(
                values=rs.randn(20, 5),
                base_values=np.ones(20) * 0.5,
            ),
        },
        show=False,
    )
    plt.tight_layout()
    return fig

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import shap
from shap.utils._exceptions import DimensionError


def test_violin_with_invalid_plot_type():
    with pytest.raises(ValueError, match="plot_type: Expected one of "):
        shap.plots.violin(np.random.randn(20, 5), plot_type="nonsense")


def test_violin_wrong_features_shape():
    """Checks that DimensionError is raised if the features data matrix
    has an incompatible shape with the shap_values matrix.
    """
    rs = np.random.RandomState(42)

    emsg = (
        "The shape of the shap_values matrix does not match the shape of "
        "the provided data matrix. Perhaps the extra column"
    )
    with pytest.raises(DimensionError, match=emsg):
        expln = shap.Explanation(
            values=rs.randn(20, 5),
            data=rs.randn(20, 4),
        )
        shap.plots.violin(expln, show=False)
    # legacy API
    with pytest.raises(DimensionError, match=emsg):
        shap.plots.violin(
            shap_values=rs.randn(20, 5),
            features=rs.randn(20, 4),
            show=False,
        )

    emsg = "The shape of the shap_values matrix does not match the shape of the provided data matrix."
    with pytest.raises(DimensionError, match=emsg):
        expln = shap.Explanation(
            values=rs.randn(20, 5),
            data=rs.randn(20, 1),
        )
        shap.plots.violin(expln, show=False)
    # legacy API
    with pytest.raises(DimensionError, match=emsg):
        shap.plots.violin(
            shap_values=rs.randn(20, 5),
            features=rs.randn(20, 1),
            show=False,
        )


@pytest.mark.mpl_image_compare
def test_violin(explainer):
    """Make sure the violin plot is unchanged."""
    fig = plt.figure()
    shap_values = explainer.shap_values(explainer.data)
    shap.plots.violin(shap_values, show=False)
    plt.tight_layout()
    return fig


# FIXME: remove once we migrate violin completely to the Explanation object
# ------ "legacy" violin plots -------
# Currently using the same files as the `test_summary.py` violin tests for comparison
@pytest.mark.mpl_image_compare(
    filename="test_summary_violin_with_data.png",
    tolerance=5,
)
def test_summary_violin_with_data2():
    """Check a violin chart with shap_values as a np.array."""
    rs = np.random.RandomState(0)
    fig = plt.figure()
    shap.plots.violin(
        rs.standard_normal(size=(20, 5)),
        rs.standard_normal(size=(20, 5)),
        plot_type="violin",
        show=False,
    )
    fig.set_layout_engine("tight")
    return fig


# Currently using the same files as the `test_summary.py` violin tests for comparison
@pytest.mark.mpl_image_compare(
    filename="test_summary_layered_violin_with_data.png",
    tolerance=5,
)
def test_summary_layered_violin_with_data2():
    """Check a layered violin chart with shap_values as a np.array."""
    rs = np.random.RandomState(0)
    fig = plt.figure()
    shap_values = rs.randn(200, 5)
    feats = rs.randn(200, 5)
    shap.plots.violin(
        shap_values,
        feats,
        plot_type="layered_violin",
        show=False,
    )
    fig.set_layout_engine("tight")
    return fig


@pytest.mark.mpl_image_compare(filename="test_violin_with_title.png", tolerance=5)
def test_violin_with_title():
    """Checks for warning when title value is passed"""
    fig = plt.figure()

    emsg = "The `title` argument is unused and will be removed in a future release."
    with pytest.warns(DeprecationWarning, match=emsg):
        shap.plots.violin(np.random.randn(20, 5), show=False, title="Violin")

    plt.tight_layout()
    return fig


def test_violin_multi_output_values():
    """Checks for error when multi output values are passed"""
    values = np.random.randn(20, 5)
    shap_values = [values, values]

    emsg = "Violin plots don't support multi-output explanations! Use 'shap.plots.bar` instead."
    with pytest.raises(TypeError, match=emsg):
        shap.plots.violin(shap_values)


@pytest.mark.parametrize(
    "features, expected_feature_names",
    [
        # Feature is of type DataFrame
        (pd.DataFrame({"A": [1, 2], "B": [3, 4]}), ["A", "B"]),
        # Feature is of type list
        (["Feature 1", "Feature 2"], ["Feature 1", "Feature 2"]),
        # Feature is of type Array
        (np.array(["Prop 1", "Prop 2"]), ["Prop 1", "Prop 2"]),
    ],
)
def test_violin_features(features, expected_feature_names):
    """Checks for conditions where features are passed"""
    shap_values = np.random.randn(2, 2)

    shap.plots.violin(shap_values, features=features, show=False)

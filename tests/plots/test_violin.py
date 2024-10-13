import matplotlib.pyplot as plt
import numpy as np
import pytest

import shap
from shap.utils._exceptions import DimensionError


def test_violin_input_is_explanation():
    """Checks an error is raised if a non-Explanation object is passed as input."""
    with pytest.raises(
        TypeError,
        match="The shap_values parameter must be a shap.Explanation object",
    ):
        _ = shap.plots.violin(np.random.randn(20, 5), show=False)  # type: ignore


def test_violin_with_invalid_plot_type():
    rs = np.random.RandomState(42)
    with pytest.raises(ValueError, match="plot_type: Expected one of "):
        expln = shap.Explanation(
            values=rs.randn(20, 5),
            data=rs.randn(20, 4),
        )
        shap.plots.violin(expln, plot_type="nonsense")


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

    emsg = "The shape of the shap_values matrix does not match the shape of the provided data matrix."
    with pytest.raises(DimensionError, match=emsg):
        expln = shap.Explanation(
            values=rs.randn(20, 5),
            data=rs.randn(20, 1),
        )
        shap.plots.violin(expln, show=False)


@pytest.mark.mpl_image_compare
def test_violin(explainer):
    """Make sure legacy violin plot (no feature names, no data) is unchanged."""
    fig = plt.figure()
    explainer.data_feature_names = None  # simulate missing feature names
    shap_values = explainer(explainer.data)
    shap_values.data = None  # plot without feature values
    shap.plots.violin(shap_values, show=False)
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare
def test_violin_with_features(explainer):
    """Make sure violin plot is unchanged."""
    fig = plt.figure()
    shap_values = explainer(explainer.data)
    shap.plots.violin(shap_values, show=False)
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare
def test_violin_layered(explainer):
    """Make sure layered violin plot is unchanged."""
    fig = plt.figure()
    shap_values = explainer(explainer.data)
    shap.plots.violin(shap_values, plot_type="layered_violin", show=False)
    plt.tight_layout()
    return fig

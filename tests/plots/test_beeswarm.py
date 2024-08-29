import matplotlib.pyplot as plt
import numpy as np
import pytest

import shap
from shap.utils._exceptions import DimensionError


def test_beeswarm_input_is_explanation():
    """Checks an error is raised if a non-Explanation object is passed as input."""
    with pytest.raises(
        TypeError,
        match="beeswarm plot requires an `Explanation` object",
    ):
        _ = shap.plots.beeswarm(np.random.randn(20, 5), show=False)  # type: ignore


def test_beeswarm_wrong_features_shape():
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
        shap.plots.beeswarm(expln, show=False)

    emsg = "The shape of the shap_values matrix does not match the shape of the provided data matrix."
    with pytest.raises(DimensionError, match=emsg):
        expln = shap.Explanation(
            values=rs.randn(20, 5),
            data=rs.randn(20, 1),
        )
        shap.plots.beeswarm(expln, show=False)


@pytest.mark.mpl_image_compare
def test_beeswarm(explainer):
    """Check a beeswarm chart renders correctly with shap_values as an Explanation
    object (default settings).
    """
    fig = plt.figure()
    shap_values = explainer(explainer.data)
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare
def test_beeswarm_no_group_remaining(explainer):
    """Beeswarm with group_remaining_features=False."""
    fig = plt.figure()
    shap_values = explainer(explainer.data)
    shap.plots.beeswarm(shap_values, show=False, group_remaining_features=False)
    plt.tight_layout()
    return fig

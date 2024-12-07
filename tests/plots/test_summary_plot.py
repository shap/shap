import matplotlib.pyplot as plt
import numpy as np
import pytest

import shap


def test_summary_plot_wrong_features_shape():
    """Checks that ValueError is raised if the features data matrix
    has an incompatible shape with the shap_values matrix.
    """

    rs = np.random.RandomState(42)

    emsg = (
        r"The shape of the shap_values matrix does not match the shape of the provided data matrix\. "
        r"Perhaps the extra column in the shap_values matrix is the constant offset\? Of so just pass shap_values\[:,:-1\]\."
    )
    with pytest.raises(ValueError, match=emsg):
        shap.summary_plot(rs.randn(20, 5), rs.randn(20, 4), show=False)

    emsg = "The shape of the shap_values matrix does not match the shape of the provided data matrix."
    with pytest.raises(AssertionError, match=emsg):
        shap.summary_plot(rs.randn(20, 5), rs.randn(20, 1), show=False)


@pytest.mark.mpl_image_compare
def test_summary_plot(explainer):
    """Check a beeswarm chart renders correctly with shap_values as an Explanation
    object (default settings).
    """
    fig = plt.figure()
    shap_values = explainer(explainer.data)
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    return fig

import matplotlib.pyplot as plt
import pytest
import shap
from .utils import explainer # (pytest fixture do not remove) pylint: disable=unused-import


@pytest.mark.mpl_image_compare
def test_violin(explainer): # pylint: disable=redefined-outer-name
    """ Make sure the violin plot is unchanged.
    """
    fig = plt.figure()
    shap_values = explainer.shap_values(explainer.data)
    shap.plots.violin(shap_values)
    plt.tight_layout()
    return fig

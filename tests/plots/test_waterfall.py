import matplotlib.pyplot as plt
import pytest
import shap
from .utils import explainer # (pytest fixture do not remove) pylint: disable=unused-import

@pytest.mark.mpl_image_compare
def test_waterfall(explainer): # pylint: disable=redefined-outer-name
    """ Test the new waterfall plot.
    """
    fig = plt.figure()
    shap_values = explainer(explainer.data)
    shap.plots.waterfall(shap_values[0])
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare
def test_waterfall_legacy(explainer): # pylint: disable=redefined-outer-name
    """ Test the old waterfall plot.
    """
    shap_values = explainer.shap_values(explainer.data)
    fig = plt.figure()
    shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0]) # pylint: disable=protected-access
    plt.tight_layout()
    return fig

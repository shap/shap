import matplotlib.pyplot as plt
import numpy as np
import pytest
import shap
from .utils import explainer # (pytest fixture do not remove) pylint: disable=unused-import


@pytest.mark.mpl_image_compare
def test_heatmap(explainer): # pylint: disable=redefined-outer-name
    """ Make sure the heatmap plot is unchanged.
    """
    fig = plt.figure()
    shap_values = explainer(explainer.data)
    shap.plots.heatmap(shap_values)
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare
def test_heatmap_feature_order(explainer): # pylint: disable=redefined-outer-name
    """ Make sure the heatmap plot is unchanged when we apply a feature ordering.
    """
    fig = plt.figure()
    shap_values = explainer(explainer.data)
    shap.plots.heatmap(shap_values, max_display=5,
                       feature_order=np.array(range(shap_values.shape[1]))[::-1])
    plt.tight_layout()
    return fig

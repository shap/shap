import matplotlib.pyplot as plt
import numpy as np
import pytest

import shap


@pytest.mark.mpl_image_compare
def test_monitoring_plot():
    """Basic monitoring plot test."""
    # Create dummy data to simulate a model over time
    np.random.seed(0)
    shap_values = np.random.randn(200, 2)
    features = np.random.randn(200, 2)

    fig = plt.figure()
    shap.plots.monitoring(0, shap_values, features, show=False)
    plt.tight_layout()
    return fig

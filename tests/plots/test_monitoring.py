import matplotlib.pyplot as plt
import numpy as np

import shap


def test_monitoring_short_input_does_not_crash():
    """Ensure monitoring plot does not crash for small input arrays."""

    shap_values = np.ones((75, 2))
    features = np.ones((75, 2))

    shap.plots.monitoring(0, shap_values, features, show=False)

    plt.close("all")

import matplotlib.pyplot as pl
import numpy as np

import shap


def test_random_decision():
    """Make sure the decision plot does not crash on random data."""
    np.random.seed(0)
    shap.decision_plot(0, np.random.randn(20, 5), np.random.randn(20, 5), show=False)
    pl.close()

"""Shared pytest fixtures"""

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def close_matplotlib_plots_after_tests():
    plt.close("all")


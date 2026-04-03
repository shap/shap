import numpy as np
import pytest

import shap
from shap.utils._exceptions import DimensionError


def test_force_plot_dimension_mismatch():
    base_value = 0.5
    shap_values = np.array([1, 2, 3])
    features = np.array([1, 2])  # mismatch

    with pytest.raises(DimensionError):
        shap.force_plot(base_value, shap_values, features, matplotlib=True)

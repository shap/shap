import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')
import shap  # pylint: disable=wrong-import-position


def test_random_dependence():
    """ Make sure a dependence plot does not crash.
    """
    shap.dependence_plot(0, np.random.randn(20, 5), np.random.randn(20, 5), show=False)

def test_random_dependence_no_interaction():
    """ Make sure a dependence plot does not crash when we are not showing interactions.
    """
    shap.dependence_plot(0, np.random.randn(20, 5), np.random.randn(20, 5), show=False, interaction_index=None)

def test_dependence_legacy_deprecation_warning(explainer):
    with pytest.warns(FutureWarning, match="dependence_legacy is being deprecated in Version 0.43.0. This will be removed in Version 0.44"):
        shap.plots._scatter.dependence_legacy(0, np.random.randn(20, 5), np.random.randn(20, 5), show=False, interaction_index=None)

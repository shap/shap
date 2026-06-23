import matplotlib.pyplot as plt
import numpy as np
import pytest

import shap


def test_issue_3553_waterfall_legacy_index_error():
    """
    Regression test for GH #3553.
    Ensures waterfall_legacy handles cases where feature_names is shorter than shap_values
    by padding with default names instead of raising an IndexError.
    """
    expected_value = 0.5
    # 10 SHAP values
    shap_values = np.random.randn(10)
    # Only 5 feature names (this would have caused IndexError before the fix)
    feature_names = ["feat1", "feat2", "feat3", "feat4", "feat5"]
    features = np.random.randn(10)

    try:
        # This should have crashed on line 511/513 of _waterfall.py
        shap.plots._waterfall.waterfall_legacy(
            expected_value, shap_values, features=features, feature_names=feature_names, show=False
        )
    except IndexError as e:
        pytest.fail(f"waterfall_legacy raised IndexError: {e}")
    except Exception as e:
        pytest.fail(f"waterfall_legacy raised an unexpected exception: {e}")
    finally:
        plt.close()


def test_waterfall_legacy_no_feature_names():
    """
    Ensures waterfall_legacy handles feature_names=None.
    """
    expected_value = 0.5
    shap_values = np.random.randn(5)

    try:
        shap.plots._waterfall.waterfall_legacy(expected_value, shap_values, feature_names=None, show=False)
    except Exception as e:
        pytest.fail(f"waterfall_legacy failed with feature_names=None: {e}")
    finally:
        plt.close()

import matplotlib.pyplot as plt
import numpy as np
import pytest

import shap


@pytest.mark.mpl_image_compare
def test_group_difference(explainer):
    """Check that the group_difference plot is unchanged."""
    np.random.seed(0)

    shap_values = explainer(explainer.data)
    group_mask = np.random.randint(2, size=shap_values.shape[0]).astype(bool)
    feature_names = shap_values.feature_names
    fig, ax = plt.subplots()
    ax_out = shap.plots.group_difference(shap_values, group_mask, feature_names=feature_names, show=False, ax=ax)
    assert ax_out is ax
    plt.tight_layout()
    return fig

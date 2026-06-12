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
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
    ax_out = shap.plots.group_difference(
        shap_values,
        group_mask,
        feature_names=feature_names,
        xmin=-0.7,
        xmax=0.9,
        show=False,
        ax=ax,
    )
    assert ax_out is ax
    plt.tight_layout()
    return fig


def test_group_difference_rejects_non_1d_group_mask(explainer):
    """group_mask must be one-dimensional."""
    shap_values = explainer(explainer.data)
    group_mask = np.zeros((shap_values.shape[0], 1), dtype=bool)

    with pytest.raises(ValueError, match="one-dimensional"):
        shap.plots.group_difference(shap_values, group_mask, show=False)


def test_group_difference_rejects_invalid_group_mask_values(explainer):
    """group_mask must be boolean (or contain only 0/1 values)."""
    shap_values = explainer(explainer.data)

    group_mask = np.zeros(shap_values.shape[0], dtype=int)
    group_mask[-1] = 2

    with pytest.raises(ValueError, match="boolean mask"):
        shap.plots.group_difference(shap_values, group_mask, show=False)


def test_group_difference_rejects_group_mask_length_mismatch(explainer):
    """group_mask length must match the number of SHAP rows."""
    shap_values = explainer(explainer.data)
    group_mask = np.zeros(shap_values.shape[0] - 1, dtype=bool)

    with pytest.raises(ValueError, match="same length"):
        shap.plots.group_difference(shap_values, group_mask, show=False)


def test_group_difference_rejects_empty_groups(explainer):
    """Both groups must contain at least one sample."""
    shap_values = explainer(explainer.data)
    group_mask = np.ones(shap_values.shape[0], dtype=bool)

    with pytest.raises(ValueError, match="two non-empty groups"):
        shap.plots.group_difference(shap_values, group_mask, show=False)


def test_group_difference_deprecates_on_invalid_input_type(explainer):
    """Check that a DeprecationWarning is raised when shap_values is not an Explanation."""
    shap_values = explainer(explainer.data)
    group_mask = np.random.randint(2, size=shap_values.shape[0]).astype(bool)
    
    with pytest.warns(DeprecationWarning, match="Passing a numpy array to the group_difference plot is deprecated"):
        shap.plots.group_difference(np.asarray(shap_values.values), group_mask, show=False)

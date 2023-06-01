import matplotlib.pyplot as plt
import numpy as np
import pytest
import shap


def test_violin_with_invalid_plot_type():
    with pytest.raises(ValueError, match="plot_type: Expected one of "):
        shap.plots.violin(np.random.randn(20, 5), plot_type="nonsense")


@pytest.mark.mpl_image_compare
def test_violin(explainer):  # pylint: disable=redefined-outer-name
    """Make sure the violin plot is unchanged."""
    fig = plt.figure()
    shap_values = explainer.shap_values(explainer.data)
    shap.plots.violin(shap_values)
    plt.tight_layout()
    return fig


# FIXME: remove once we migrate violin completely to the Explanation object
# ------ "legacy" violin plots -------
# Currently using the same files as the `test_summary.py` violin tests for comparison
@pytest.mark.mpl_image_compare(
    filename="test_random_summary_violin_with_data.png",
)
def test_random_summary_violin_with_data2():
    """Check a violin chart with shap_values as a np.array."""
    np.random.seed(0)
    fig = plt.figure()
    shap.plots.violin(
        np.random.randn(20, 5),
        np.random.randn(20, 5),
        plot_type="violin",
        show=False,
    )
    plt.tight_layout()
    return fig


# Currently using the same files as the `test_summary.py` violin tests for comparison
@pytest.mark.mpl_image_compare(
    filename="test_random_summary_layered_violin_with_data.png",
)
def test_random_summary_layered_violin_with_data2():
    """Check a layered violin chart with shap_values as a np.array."""
    rs = np.random.RandomState(0)
    fig = plt.figure()
    shap_values = rs.randn(200, 5)
    feats = rs.randn(200, 5)
    shap.plots.violin(
        shap_values,
        feats,
        plot_type="layered_violin",
        show=False,
    )
    plt.tight_layout()
    return fig

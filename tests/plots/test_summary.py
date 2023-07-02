import matplotlib.pyplot as plt
import numpy as np
import pytest

import shap


@pytest.mark.mpl_image_compare
def test_random_summary():
    """ Just make sure the summary_plot function doesn't crash.
    """
    rng = np.random.default_rng(seed=0)
    fig = plt.figure()
    shap.summary_plot(rng.standard_normal(size=(20, 5)), show=False)
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare
def test_random_summary_with_data():
    """ Just make sure the summary_plot function doesn't crash with data.
    """
    rng = np.random.default_rng(seed=0)
    fig = plt.figure()
    shap.summary_plot(rng.standard_normal(size=(20, 5)), rng.standard_normal(size=(20, 5)), show=False)
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare
def test_random_multi_class_summary():
    """ Check a multiclass run.
    """
    rng = np.random.default_rng(seed=0)
    fig = plt.figure()
    shap.summary_plot([rng.standard_normal(size=(20, 5)) for i in range(3)], rng.standard_normal(size=(20, 5)), show=False)
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare
def test_random_summary_bar_with_data():
    """ Check a bar chart.
    """
    rng = np.random.default_rng(seed=0)
    fig = plt.figure()
    shap.summary_plot(rng.standard_normal(size=(20, 5)), rng.standard_normal(size=(20, 5)), plot_type="bar", show=False)
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare
def test_random_summary_dot_with_data():
    """ Check a dot chart.
    """
    rng = np.random.default_rng(seed=0)
    fig = plt.figure()
    shap.summary_plot(rng.standard_normal(size=(20, 5)), rng.standard_normal(size=(20, 5)), plot_type="dot", show=False)
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare
def test_random_summary_violin_with_data():
    """ Check a violin chart.
    """
    rng = np.random.default_rng(seed=0)
    fig = plt.figure()
    shap.summary_plot(rng.standard_normal(size=(20, 5)), rng.standard_normal(size=(20, 5)), plot_type="violin", show=False)
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare
def test_random_summary_layered_violin_with_data():
    """ Check a layered violin chart.
    """
    rs = np.random.RandomState(0)
    fig = plt.figure()
    shap_values = rs.randn(200, 5)
    feats = rs.randn(200, 5)
    shap.summary_plot(
        shap_values,
        feats,
        plot_type="layered_violin",
        show=False,
    )
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare(tolerance=6)
def test_random_summary_with_log_scale():
    """ Check a with a log scale.
    """
    rng = np.random.default_rng(seed=0)
    fig = plt.figure()
    shap.summary_plot(rng.standard_normal(size=(20, 5)), use_log_scale=True, show=False)
    plt.tight_layout()
    return fig

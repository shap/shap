import matplotlib.pyplot as plt
import pytest

import shap


@pytest.mark.mpl_image_compare
def test_scatter_single(explainer):
    explanation = explainer(explainer.data)
    shap.plots.scatter(explanation[:, "Age"], show=False)
    fig = plt.gcf()
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare
def test_scatter_interaction(explainer):
    explanation = explainer(explainer.data)
    shap.plots.scatter(explanation[:, "Age"], color=explanation[:, "Workclass"], show=False)
    fig = plt.gcf()
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare
def test_scatter_dotchain(explainer):
    explanation = explainer(explainer.data)
    shap.plots.scatter(explanation[:, explanation.abs.mean(0).argsort[-2]], show=False)
    fig = plt.gcf()
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare
def test_scatter_custom(explainer):
    # Test with custom x/y limits, alpha and colormap
    explanation = explainer(explainer.data)
    age = explanation[:, "Age"]
    shap.plots.scatter(
        age,
        color=explanation[:, "Workclass"],
        xmin=age.percentile(20),
        xmax=age.percentile(80),
        ymin=age.percentile(10),
        ymax=age.percentile(90),
        alpha=0.5,
        cmap=plt.get_cmap("cool"),
        show=False,
    )
    fig = plt.gcf()
    plt.tight_layout()
    return fig

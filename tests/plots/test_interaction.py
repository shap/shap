import numpy as np
import pytest

import shap


def test_interaction_heatmap(explainer):
    shap_values = explainer(explainer.data)
    ax = shap.plots.interaction_heatmap(shap_values, max_display=8, show=False)
    assert ax is not None
    assert ax.get_title() == "Pairwise interaction strength"


def test_interaction_heatmap_requires_explanation():
    with pytest.raises(TypeError, match="requires an `Explanation` object"):
        shap.plots.interaction_heatmap(np.zeros((10, 3)), show=False)  # type: ignore[arg-type]


def test_interaction_heatmap_requires_data():
    expl = shap.Explanation(values=np.zeros((10, 3)), data=None)
    with pytest.raises(ValueError, match="requires shap_values.data"):
        shap.plots.interaction_heatmap(expl, show=False)


def test_interaction_heatmap_returns_axes_with_show_false():
    rs = np.random.RandomState(0)
    expl = shap.Explanation(
        values=rs.normal(size=(50, 4)),
        data=rs.normal(size=(50, 4)),
        feature_names=["f0", "f1", "f2", "f3"],
    )
    ax = shap.plots.interaction_heatmap(expl, show=False)
    assert ax is not None


def test_interaction_beeswarm(explainer):
    shap_values = explainer(explainer.data)
    ax = shap.plots.interaction_beeswarm(shap_values, max_display=8, max_pairs=10, show=False)
    assert ax is not None
    assert ax.get_title() == "Top pairwise interactions"


def test_interaction_beeswarm_max_pairs_validation():
    rs = np.random.RandomState(0)
    expl = shap.Explanation(
        values=rs.normal(size=(40, 4)),
        data=rs.normal(size=(40, 4)),
        feature_names=["f0", "f1", "f2", "f3"],
    )
    with pytest.raises(ValueError, match="max_pairs"):
        shap.plots.interaction_beeswarm(expl, max_pairs=0, show=False)

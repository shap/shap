from contextlib import nullcontext as does_not_raise

import matplotlib.pyplot as plt
import numpy as np
import pytest
from pytest import param

import shap


@pytest.fixture
def data_explainer_shap_values():
    RandomForestRegressor = pytest.importorskip("sklearn.ensemble").RandomForestRegressor

    # train model
    X, y = shap.datasets.california(n_points=500)
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X, y)

    # explain the model's predictions using SHAP values
    explainer = shap.TreeExplainer(model)
    return X, explainer, explainer.shap_values(X)


@pytest.mark.parametrize(
    "cmap, exp_ctx",
    [
        # Valid cmaps
        param("coolwarm", does_not_raise(), id="valid-str"),
        param(["#000000", "#ffffff"], does_not_raise(), id="valid-list[str]"),
        # Invalid cmaps
        param(
            777,
            pytest.raises(TypeError, match="Plot color map must be string or list!"),
            id="invalid-dtype1",
        ),
        param(
            [],
            pytest.raises(ValueError, match="Color map must be at least two colors"),
            id="invalid-insufficient-colors1",
        ),
        param(
            ["#8834BB"],
            pytest.raises(ValueError, match="Color map must be at least two colors"),
            id="invalid-insufficient-colors2",
        ),
        param(
            ["#883488", "#Gg8888"],
            pytest.raises(ValueError, match=r"Invalid color .+ found in cmap"),
            id="invalid-hexcolor-in-list1",
        ),
        param(
            ["#883488", "#1111119"],
            pytest.raises(ValueError, match=r"Invalid color .+ found in cmap"),
            id="invalid-hexcolor-in-list2",
        ),
    ],
)
def test_verify_valid_cmap(cmap, exp_ctx):
    from shap.plots._force import verify_valid_cmap

    with exp_ctx:
        verify_valid_cmap(cmap)


def test_random_force_plot_mpl_with_data(data_explainer_shap_values):
    """Test if force plot with matplotlib works."""
    X, explainer, shap_values = data_explainer_shap_values

    # visualize the first prediction's explanation
    shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :], matplotlib=True, show=False)
    with pytest.raises(TypeError, match="force plot now requires the base value as the first parameter"):
        shap.force_plot([1, 1], shap_values, X.iloc[0, :], show=False)
    plt.close("all")


def test_random_force_plot_mpl_text_rotation_with_data(data_explainer_shap_values):
    """Test if force plot with matplotlib works when supplied with text_rotation."""
    X, explainer, shap_values = data_explainer_shap_values

    # visualize the first prediction's explanation
    shap.force_plot(
        explainer.expected_value, shap_values[0, :], X.iloc[0, :], matplotlib=True, text_rotation=30, show=False
    )
    plt.close("all")


@pytest.mark.mpl_image_compare(tolerance=3)
def test_force_plot_negative_sign():
    np.random.seed(0)
    base = 100
    contribution = np.r_[-np.random.rand(5)]
    names = [f"minus_{i}" for i in range(5)]

    shap.force_plot(
        base,
        contribution,
        names,
        matplotlib=True,
        show=False,
    )
    return plt.gcf()


@pytest.mark.mpl_image_compare(tolerance=3)
def test_force_plot_positive_sign():
    np.random.seed(0)
    base = 100
    contribution = np.r_[np.random.rand(5)]
    names = [f"plus_{i}" for i in range(5)]

    shap.force_plot(
        base,
        contribution,
        names,
        matplotlib=True,
        show=False,
    )
    return plt.gcf()


def test_force_array_higher_predictions_first():
    """Test that AdditiveForceArrayVisualizer puts higher predictions first (GH #4342).

    np.flipud(clustOrder) must be assigned back; otherwise the reorder is a no-op.
    """
    from shap.plots._force import AdditiveExplanation, AdditiveForceArrayVisualizer, DenseData, Instance, Model
    from shap.utils._legacy import IdentityLink

    link = IdentityLink()
    n_features = 3
    feature_names = [f"f{i}" for i in range(n_features)]
    model = Model(None, ["f(x)"])

    # Build explanations whose total effects span negative to positive.
    # Hierarchical clustering will produce a linear leaf ordering for these
    # well-separated, linearly-spaced effects.
    effect_sums = [-10.0, -5.0, 0.0, 5.0, 10.0]
    explanations = []
    for s in effect_sums:
        effects = np.full(n_features, s / n_features)
        instance = Instance(np.zeros((1, n_features)), np.zeros(n_features))
        data = DenseData(np.zeros((1, n_features)), list(feature_names))
        explanations.append(
            AdditiveExplanation(
                base_value=0.0,
                out_value=float(s),
                effects=effects,
                effects_var=None,
                instance=instance,
                link=link,
                model=model,
                data=data,
            )
        )

    viz = AdditiveForceArrayVisualizer(explanations)

    # Collect (outValue, simIndex) pairs produced by the visualizer.
    out_values = [exp["outValue"] for exp in viz.data["explanations"]]
    sim_indices = [exp["simIndex"] for exp in viz.data["explanations"]]

    # The explanation with simIndex == 1 (displayed first in the stacked plot)
    # must have higher-or-equal total effects than the one displayed last.
    first_idx = sim_indices.index(1)
    last_idx = sim_indices.index(len(explanations))
    assert out_values[first_idx] >= out_values[last_idx]

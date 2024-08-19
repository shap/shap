from contextlib import nullcontext as does_not_raise

import matplotlib.pyplot as plt
import numpy as np
import pytest
from pytest import param

import shap


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


def test_random_force_plot_mpl_with_data():
    """Test if force plot with matplotlib works."""
    RandomForestRegressor = pytest.importorskip("sklearn.ensemble").RandomForestRegressor

    # train model
    X, y = shap.datasets.california(n_points=500)
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    # explain the model's predictions using SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # visualize the first prediction's explanation
    shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :], matplotlib=True, show=False)
    with pytest.raises(TypeError, match="force plot now requires the base value as the first parameter"):
        shap.force_plot([1, 1], shap_values, X.iloc[0, :], show=False)


def test_random_force_plot_mpl_text_rotation_with_data():
    """Test if force plot with matplotlib works when supplied with text_rotation."""
    RandomForestRegressor = pytest.importorskip("sklearn.ensemble").RandomForestRegressor

    # train model
    X, y = shap.datasets.california(n_points=500)
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    # explain the model's predictions using SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # visualize the first prediction's explanation
    shap.force_plot(
        explainer.expected_value, shap_values[0, :], X.iloc[0, :], matplotlib=True, text_rotation=30, show=False
    )


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

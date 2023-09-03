import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

matplotlib.use('Agg')
import shap  # noqa: E402


@pytest.mark.parametrize(
    "cmap, emsg",
    [
        ("coolwarm", None),
        (["#000000", "#ffffff"], None),
        (777, "Plot color map must be string or list!"),
        ([], "Color map must be at least two colors"),
        (["#8834BB"], "Color map must be at least two colors"),
        (["#883488", "#Gg8888"], r"Invalid color .+ found in cmap"),
        (["#883488", "#1111119"], r"Invalid color .+ found in cmap"),
    ],
    ids=[
        "valid-str",
        "valid-list[str]",
        "invalid-dtype1",
        "invalid-insufficient-colors1",
        "invalid-insufficient-colors2",
        "invalid-hexcolor-in-list1",
        "invalid-hexcolor-in-list2",
    ],
)
def test_verify_valid_cmap(cmap, emsg):
    from shap.plots._force import verify_valid_cmap

    if emsg is None:
        # Valid cmaps
        _ = verify_valid_cmap(cmap)
    else:
        # Invalid cmaps
        with pytest.raises(ValueError, match=emsg):
            verify_valid_cmap(cmap)

def test_random_force_plot_mpl_with_data():
    """ Test if force plot with matplotlib works.
    """

    RandomForestRegressor = pytest.importorskip('sklearn.ensemble').RandomForestRegressor

    # train model
    X, y = shap.datasets.california(n_points=500)
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    # explain the model's predictions using SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # visualize the first prediction's explanation
    shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :], matplotlib=True, show=False)

def test_random_force_plot_mpl_text_rotation_with_data():
    """ Test if force plot with matplotlib works when supplied with text_rotation.
    """

    RandomForestRegressor = pytest.importorskip('sklearn.ensemble').RandomForestRegressor

    # train model
    X, y = shap.datasets.california(n_points=500)
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    # explain the model's predictions using SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # visualize the first prediction's explanation
    shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :], matplotlib=True, text_rotation=30, show=False)

@pytest.mark.mpl_image_compare(tolerance=3)
def test_random_force_plot_negative_sign():
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
def test_random_force_plot_positive_sign():
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

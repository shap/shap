import sys
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
    model = RandomForestRegressor(n_estimators=100)
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


@pytest.mark.skipif(
    sys.platform == "darwin", reason="Since this test is flaky on MacOS, we skip it for now. See GH #4102."
)
def test_random_force_plot_mpl_with_data(data_explainer_shap_values):
    """Test if force plot with matplotlib works."""
    X, explainer, shap_values = data_explainer_shap_values

    # visualize the first prediction's explanation
    shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :], matplotlib=True, show=False)
    with pytest.raises(TypeError, match="force plot now requires the base value as the first parameter"):
        shap.force_plot([1, 1], shap_values, X.iloc[0, :], show=False)


@pytest.mark.skipif(
    sys.platform == "darwin", reason="Since this test is flaky on MacOS, we skip it for now. See GH #4102."
)
def test_random_force_plot_mpl_text_rotation_with_data(data_explainer_shap_values):
    """Test if force plot with matplotlib works when supplied with text_rotation."""
    X, explainer, shap_values = data_explainer_shap_values

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


def test_force_plot_with_explanation_object():
    """Test force plot by passing an Explanation object as base_value."""
    np.random.seed(42)
    shap_values = np.random.randn(10)
    features = np.random.randn(10)
    feature_names = [f"Feature {i}" for i in range(10)]

    explanation = shap.Explanation(values=shap_values, base_values=0.5, data=features, feature_names=feature_names)

    # Pass Explanation object as first parameter
    shap.force_plot(explanation, matplotlib=True, show=False)
    plt.close()


def test_force_plot_with_dataframe_features():
    """Test force plot with pandas DataFrame features."""
    pd = pytest.importorskip("pandas")
    np.random.seed(42)

    shap_values = np.random.randn(3)
    features_df = pd.DataFrame([[1, 2, 3]], columns=["A", "B", "C"])

    shap.force_plot(0.0, shap_values, features_df.iloc[0], matplotlib=True, show=False)
    plt.close()


def test_force_plot_with_series_features():
    """Test force plot with pandas Series features."""
    pd = pytest.importorskip("pandas")
    np.random.seed(42)

    shap_values = np.random.randn(3)
    features_series = pd.Series([1, 2, 3], index=["A", "B", "C"])

    shap.force_plot(0.0, shap_values, features_series, matplotlib=True, show=False)
    plt.close()


def test_force_plot_with_feature_names_only():
    """Test force plot with feature names instead of feature values."""
    np.random.seed(42)
    shap_values = np.random.randn(5)
    feature_names = ["Feat1", "Feat2", "Feat3", "Feat4", "Feat5"]

    shap.force_plot(0.0, shap_values, feature_names, matplotlib=True, show=False)
    plt.close()


def test_force_plot_no_features():
    """Test force plot without providing features."""
    np.random.seed(42)
    shap_values = np.random.randn(5)

    shap.force_plot(0.0, shap_values, matplotlib=True, show=False)
    plt.close()


def test_force_plot_with_out_names():
    """Test force plot with custom output names."""
    np.random.seed(42)
    shap_values = np.random.randn(5)
    features = np.random.randn(5)

    shap.force_plot(0.0, shap_values, features, out_names="Prediction", matplotlib=True, show=False)
    plt.close()


def test_force_plot_with_logit_link():
    """Test force plot with logit link."""
    np.random.seed(42)
    shap_values = np.random.randn(5) * 0.1
    features = np.random.randn(5)

    shap.force_plot(0.0, shap_values, features, link="logit", matplotlib=True, show=False)
    plt.close()


def test_force_plot_with_figsize():
    """Test force plot with custom figsize."""
    np.random.seed(42)
    shap_values = np.random.randn(5)
    features = np.random.randn(5)

    shap.force_plot(0.0, shap_values, features, figsize=(10, 5), matplotlib=True, show=False)
    plt.close()


def test_force_plot_with_contribution_threshold():
    """Test force plot with custom contribution_threshold."""
    np.random.seed(42)
    shap_values = np.random.randn(10) * 0.5
    features = np.random.randn(10)

    shap.force_plot(0.0, shap_values, features, contribution_threshold=0.1, matplotlib=True, show=False)
    plt.close()


def test_force_plot_dimension_error():
    """Test that force plot raises DimensionError for mismatched features."""
    np.random.seed(42)
    shap_values = np.random.randn(5)
    features = np.random.randn(3)  # Wrong size

    with pytest.raises(shap.utils._exceptions.DimensionError, match="Length of features is not equal"):
        shap.force_plot(0.0, shap_values, features, matplotlib=True, show=False)


def test_force_plot_list_shap_values_error():
    """Test that force plot raises TypeError for list shap_values (multi-output)."""
    np.random.seed(42)
    shap_values = [np.random.randn(5), np.random.randn(5)]

    with pytest.raises(TypeError, match="shap_values arg looks multi output"):
        shap.force_plot(0.0, shap_values, matplotlib=True, show=False)


def test_force_plot_base_value_unwrap():
    """Test force plot with base_value as single-element array."""
    np.random.seed(42)
    base_value = np.array([0.5])  # Single element array
    shap_values = np.random.randn(5)
    features = np.random.randn(5)

    shap.force_plot(base_value, shap_values, features, matplotlib=True, show=False)
    plt.close()


def test_force_plot_base_value_all_same():
    """Test force plot with base_value array where all values are same."""
    np.random.seed(42)
    base_value = np.array([0.5, 0.5, 0.5])  # All same
    shap_values = np.random.randn(3, 5)

    # Multi-sample force plots don't support matplotlib=True
    shap.force_plot(base_value, shap_values, matplotlib=False, show=False)


def test_force_plot_not_implemented_multi_sample_matplotlib():
    """Test that multi-sample matplotlib force plot raises NotImplementedError."""
    np.random.seed(42)
    base_value = 0.5
    shap_values = np.random.randn(3, 5)  # Multi-sample

    with pytest.raises(
        NotImplementedError, match="matplotlib = True is not yet supported for force plots with multiple samples"
    ):
        shap.force_plot(base_value, shap_values, matplotlib=True, show=False)

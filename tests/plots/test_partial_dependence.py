import matplotlib.pyplot as plt
import pytest

import shap


@pytest.fixture()
def simple_model_data():
    """Create a simple model and dataset for partial dependence testing."""
    xgboost = pytest.importorskip("xgboost")
    X, y = shap.datasets.adult()
    X = X.iloc[:100]
    y = y[:100]

    model = xgboost.XGBClassifier(random_state=0, tree_method="exact", base_score=0.5).fit(X, y)

    # Create a simple prediction function
    def predict_fn(data):
        return model.predict_proba(data)[:, 1]

    return predict_fn, X


@pytest.mark.mpl_image_compare(tolerance=3)
def test_partial_dependence_basic(simple_model_data):
    """Test basic 1D partial dependence plot."""
    model, X = simple_model_data
    fig, ax = shap.plots.partial_dependence("Age", model, X, ice=False, show=False)
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare(tolerance=3)
def test_partial_dependence_with_ice(simple_model_data):
    """Test partial dependence plot with ICE curves."""
    model, X = simple_model_data
    fig, ax = shap.plots.partial_dependence("Age", model, X, ice=True, show=False)
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare(tolerance=3)
def test_partial_dependence_no_hist(simple_model_data):
    """Test partial dependence plot without histogram."""
    model, X = simple_model_data
    fig, ax = shap.plots.partial_dependence("Age", model, X, hist=False, ice=False, show=False)
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare(tolerance=3)
def test_partial_dependence_percentile(simple_model_data):
    """Test partial dependence plot with percentile bounds."""
    model, X = simple_model_data
    fig, ax = shap.plots.partial_dependence(
        "Age", model, X, xmin="percentile(5)", xmax="percentile(95)", ice=False, show=False
    )
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare(tolerance=3)
def test_partial_dependence_custom_opacity(simple_model_data):
    """Test partial dependence plot with custom opacity settings."""
    model, X = simple_model_data
    fig, ax = shap.plots.partial_dependence("Age", model, X, ice=True, ace_opacity=0.3, pd_opacity=1.0, show=False)
    plt.tight_layout()
    return fig


def test_partial_dependence_with_dataframe(simple_model_data):
    """Test partial dependence plot with DataFrame input."""
    model, X = simple_model_data

    # Test that it works with DataFrame input
    fig, ax = shap.plots.partial_dependence("Age", model, X, ice=False, show=False)
    plt.close(fig)


@pytest.mark.mpl_image_compare(tolerance=5)
def test_partial_dependence_2d(simple_model_data):
    """Test 2D partial dependence plot."""
    model, X = simple_model_data
    fig, ax = shap.plots.partial_dependence(
        ("Age", "Education-Num"),
        model,
        X,
        npoints=10,  # Use fewer points for faster test
        show=False,
    )
    plt.tight_layout()
    return fig


def test_partial_dependence_custom_ylabel(simple_model_data):
    """Test partial dependence plot with custom y-axis label."""
    model, X = simple_model_data
    fig, ax = shap.plots.partial_dependence("Age", model, X, ylabel="Custom Y Label", ice=False, show=False)
    assert ax.get_ylabel() == "Custom Y Label"
    plt.close(fig)


def test_partial_dependence_feature_expected_value(simple_model_data):
    """Test partial dependence plot with feature expected value marker."""
    model, X = simple_model_data
    fig, ax = shap.plots.partial_dependence("Age", model, X, feature_expected_value=True, ice=False, show=False)
    plt.close(fig)


def test_partial_dependence_model_expected_value(simple_model_data):
    """Test partial dependence plot with model expected value marker."""
    model, X = simple_model_data
    fig, ax = shap.plots.partial_dependence("Age", model, X, model_expected_value=True, ice=False, show=False)
    plt.close(fig)


def test_partial_dependence_with_ax(simple_model_data):
    """Test partial dependence plot with custom axes."""
    model, X = simple_model_data
    fig, ax = plt.subplots()
    returned_fig, returned_ax = shap.plots.partial_dependence("Age", model, X, ax=ax, ice=False, show=False)
    assert returned_ax == ax
    plt.close(fig)


def test_partial_dependence_numpy_array(simple_model_data):
    """Test partial dependence plot with numpy array (not DataFrame)."""
    model, X = simple_model_data
    X_numpy = X.values

    # Create model that accepts numpy arrays
    def numpy_model(x):
        return model(x)

    fig, ax = shap.plots.partial_dependence(
        0,  # Use index since no feature names
        numpy_model,
        X_numpy,
        ice=True,
        show=False,
    )
    plt.close(fig)


def test_partial_dependence_no_feature_names(simple_model_data):
    """Test partial dependence with numpy array and no feature names."""
    model, X = simple_model_data
    X_numpy = X.values

    def numpy_model(x):
        return model(x)

    fig, ax = shap.plots.partial_dependence(0, numpy_model, X_numpy, ice=False, show=False)
    plt.close(fig)


def test_partial_dependence_show_true(simple_model_data, monkeypatch):
    """Test partial dependence with show=True."""
    model, X = simple_model_data
    show_called = []
    monkeypatch.setattr(plt, "show", lambda: show_called.append(True))
    shap.plots.partial_dependence("Age", model, X, ice=False, show=True)
    assert len(show_called) == 1
    plt.close()

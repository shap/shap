import matplotlib.pyplot as plt
import numpy as np
import pytest
import sklearn
import sklearn.ensemble

import shap


@pytest.mark.mpl_image_compare
def test_summary():
    """Just make sure the summary_plot function doesn't crash."""
    np.random.seed(0)
    fig = plt.figure()
    shap.summary_plot(np.random.randn(20, 5), show=False)
    fig.set_layout_engine("tight")
    return fig


@pytest.mark.mpl_image_compare
def test_summary_with_data():
    """Just make sure the summary_plot function doesn't crash with data."""
    np.random.seed(0)
    fig = plt.figure()
    shap.summary_plot(np.random.randn(20, 5), np.random.randn(20, 5), show=False)
    fig.set_layout_engine("tight")
    return fig


@pytest.mark.mpl_image_compare
def test_summary_multi_class():
    """Check a multiclass run."""
    np.random.seed(0)
    fig = plt.figure()
    shap.summary_plot([np.random.randn(20, 5) for i in range(3)], np.random.randn(20, 5), show=False)
    fig.set_layout_engine("tight")
    return fig


@pytest.mark.mpl_image_compare
def test_summary_multi_class_legend_decimals():
    """Check the functionality of printing the legend in the plot of a multiclass run when
    all the SHAP values are smaller than 1.
    """
    np.random.seed(0)
    fig = plt.figure()
    shap.summary_plot(
        [np.random.randn(20, 5) for i in range(3)], np.random.randn(20, 5), show=False, show_values_in_legend=True
    )
    fig.set_layout_engine("tight")
    return fig


@pytest.mark.mpl_image_compare
def test_summary_multi_class_legend():
    """Check the functionality of printing the legend in the plot of a multiclass run when
    SHAP values are bigger than 1.
    """
    np.random.seed(0)
    fig = plt.figure()
    shap.summary_plot(
        [(2 + np.random.randn(20, 5)) for i in range(3)],
        2 + np.random.randn(20, 5),
        show=False,
        show_values_in_legend=True,
    )
    fig.set_layout_engine("tight")
    return fig


@pytest.mark.mpl_image_compare
def test_summary_bar_with_data():
    """Check a bar chart."""
    np.random.seed(0)
    fig = plt.figure()
    shap.summary_plot(np.random.randn(20, 5), np.random.randn(20, 5), plot_type="bar", show=False)
    fig.set_layout_engine("tight")
    return fig


@pytest.mark.mpl_image_compare
def test_summary_dot_with_data():
    """Check a dot chart."""
    np.random.seed(0)
    fig = plt.figure()
    shap.summary_plot(np.random.randn(20, 5), np.random.randn(20, 5), plot_type="dot", show=False)
    fig.set_layout_engine("tight")
    return fig


@pytest.mark.mpl_image_compare
def test_summary_violin_with_data():
    """Check a violin chart."""
    np.random.seed(0)
    fig = plt.figure()
    shap.summary_plot(np.random.randn(20, 5), np.random.randn(20, 5), plot_type="violin", show=False)
    fig.set_layout_engine("tight")
    return fig


@pytest.mark.mpl_image_compare
def test_summary_layered_violin_with_data():
    """Check a layered violin chart."""
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
    fig.set_layout_engine("tight")
    return fig


@pytest.mark.mpl_image_compare(tolerance=6)
def test_summary_with_log_scale():
    """Check a with a log scale."""
    np.random.seed(0)
    fig = plt.figure()
    shap.summary_plot(np.random.randn(20, 5), use_log_scale=True, show=False)
    fig.set_layout_engine("tight")
    return fig


@pytest.mark.parametrize("background", [True, False])
def test_summary_binary_multiclass(background):
    # See GH #2893
    lightgbm = pytest.importorskip("lightgbm")
    num_examples, num_features = 100, 3
    rs = np.random.RandomState(0)
    X = rs.normal(size=[num_examples, num_features])
    y = ((2 * X[:, 0] + X[:, 1]) > 0).astype(int)

    train_data = lightgbm.Dataset(X, label=y)
    model = lightgbm.train(dict(objective="multiclass", num_classes=2), train_data)
    data = X if background else None
    explainer = shap.TreeExplainer(model, data=data)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, feature_names=["foo", "bar", "baz"], show=False)


@pytest.mark.mpl_image_compare
def test_summary_multiclass_explanation():
    """Check summary plot with multiclass model with explanation as input."""
    xgboost = pytest.importorskip("xgboost")
    n_samples = 100
    n_features = 5
    n_classes = 3
    np.random.seed(0)  # for reproducibility
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    feature_names = [f"Feature {i + 1}" for i in range(n_features)]
    model = xgboost.XGBClassifier(n_estimators=10, random_state=0, tree_method="exact", base_score=0.5).fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    fig = plt.figure()
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    fig.set_layout_engine("tight")
    return fig


@pytest.mark.mpl_image_compare
def test_summary_bar_multiclass():
    # GH 3984
    X, y = shap.datasets.iris()
    model = sklearn.ensemble.RandomForestClassifier(max_depth=2, random_state=0)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(
        shap_values, X, plot_type="bar", class_names=[0, 1, 2], feature_names=np.array(X.columns), show=False
    )
    fig = plt.gcf()
    fig.set_layout_engine("tight")
    return fig


@pytest.mark.mpl_image_compare
def test_summary_violin_regression():
    # GH 4030
    X, y = sklearn.datasets.make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)

    regr = sklearn.ensemble.RandomForestRegressor(max_depth=2, random_state=0)
    _ = regr.fit(X, y)

    explainer = shap.TreeExplainer(regr)
    shap_values = explainer.shap_values(X, y=y)
    shap.summary_plot(shap_values, features=X, plot_type="violin", show=False)
    fig = plt.gcf()
    fig.set_layout_engine("tight")
    return fig

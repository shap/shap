import platform

import matplotlib.pyplot as plt
import numpy as np
import pytest
import sklearn
import sklearn.ensemble
from numpy.testing import assert_array_equal

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


@pytest.mark.skipif(platform.system() in ["Windows", "Darwin"], reason="Images not matching on MacOS and Windows.")
@pytest.mark.mpl_image_compare
def test_summary_compact_dot_with_data():
    """Check a bar chart."""
    n_samples = 100
    n_features = 5
    np.random.seed(0)  # for reproducibility
    X = np.random.randn(n_samples, n_features)
    feature_names = [f"Feature {i + 1}" for i in range(n_features)]
    shap_values = np.random.randn(n_samples, n_features, n_features)
    fig = plt.figure()

    shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="compact_dot", show=False)
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


@pytest.mark.mpl_image_compare
def test_summary_plot_interaction():
    """Checks the summary plot with interaction effects (GH #4081)."""
    xgboost = pytest.importorskip("xgboost")

    X, y = shap.datasets.nhanesi()

    xgb_full = xgboost.DMatrix(X, label=y)

    # train final model on the full data set
    params = {"eta": 0.002, "max_depth": 3, "objective": "survival:cox", "subsample": 0.5}
    model = xgboost.train(params, xgb_full, 100)

    number_patients = 300
    shap_interaction_values = shap.TreeExplainer(model).shap_interaction_values(X.iloc[:number_patients, :])

    shap.summary_plot(shap_interaction_values, X.iloc[:number_patients, :])
    fig = plt.gcf()
    fig.set_layout_engine("tight")
    return fig


@pytest.mark.xfail(
    reason="Currently not supported since this needs an overhaul of the summary plot code. See #3920 for more information."
)
@pytest.mark.mpl_image_compare
def test_summary_plot_twice():
    # GH 3920
    xgboost = pytest.importorskip("xgboost")

    X, y = shap.datasets.california()
    model = xgboost.XGBRegressor().fit(X, y)

    explainer = shap.TreeExplainer(model)
    shapValues = explainer.shap_values(X)

    shap.summary_plot(shapValues, X, show=False)
    shap.summary_plot(shapValues, X, show=False)
    fig = plt.gcf()
    fig.set_layout_engine("tight")
    return fig


def test_summary_plot_wrong_features_shape():
    """Checks that ValueError is raised if the features data matrix
    has an incompatible shape with the shap_values matrix.
    """

    rs = np.random.RandomState(42)

    emsg = (
        r"The shape of the shap_values matrix does not match the shape of the provided data matrix\. "
        r"Perhaps the extra column in the shap_values matrix is the constant offset\? Of so just pass shap_values\[:,:-1\]\."
    )
    with pytest.raises(ValueError, match=emsg):
        shap.summary_plot(rs.randn(20, 5), rs.randn(20, 4), show=False)

    emsg = "The shape of the shap_values matrix does not match the shape of the provided data matrix."
    with pytest.raises(AssertionError, match=emsg):
        shap.summary_plot(rs.randn(20, 5), rs.randn(20, 1), show=False)


@pytest.mark.mpl_image_compare
def test_summary_plot(explainer):
    """Check a beeswarm chart renders correctly with shap_values as an Explanation
    object (default settings).
    """
    fig = plt.figure()
    shap_values = explainer(explainer.data)
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    return fig


@pytest.mark.parametrize(
    "rng",
    [
        np.random.default_rng(167089660),
        17,
        np.random.SeedSequence(entropy=60767),
    ],
)
def test_summary_plot_seed_insulated(explainer, rng):
    # ensure that it is possible for downstream
    # projects to avoid mutating global NumPy
    # random state
    # see i.e., https://scientific-python.org/specs/spec-0007/
    shap_values = explainer(explainer.data)
    state_before = np.random.get_state()[1]  # type: ignore[index]
    shap.summary_plot(shap_values, show=False, rng=rng)
    state_after = np.random.get_state()[1]  # type: ignore[index]
    assert_array_equal(state_after, state_before)


def test_summary_plot_warning(explainer):
    # enforce FutureWarning for usage of global random
    # state as we prepare for SPEC 7 adoption
    shap_values = explainer(explainer.data)
    with pytest.warns(FutureWarning, match="NumPy global RNG"):
        shap.summary_plot(shap_values, show=False)

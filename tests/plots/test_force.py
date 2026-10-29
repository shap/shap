from contextlib import nullcontext as does_not_raise

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from pytest import param

import shap

matplotlib.use("Agg")


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


@pytest.fixture
def simple_shap_values():
    """Lightweight fixture that does not require sklearn."""
    np.random.seed(42)
    base = 0.5
    shap_vals = np.array([0.1, -0.2, 0.05, 0.3, -0.15])
    feature_names = [f"f{i}" for i in range(len(shap_vals))]
    return base, shap_vals, feature_names


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


def test_flipud_reverses_clust_order():
    """Regression test for GH-4342: np.flipud(clustOrder) was a no-op."""
    from shap.plots._force import AdditiveExplanation, AdditiveForceArrayVisualizer
    from shap.utils._legacy import DenseData, IdentityLink, Instance, Model

    feature_names = ["f0", "f1"]
    base_value = 0.0
    link = IdentityLink()
    data = DenseData(np.zeros((1, 2)), feature_names)
    model = Model(lambda x: x, ["f(x)"])

    def _make_exp(effects):
        effects = np.array(effects, dtype=float)
        out_value = base_value + effects.sum()
        instance = Instance(np.ones((1, len(feature_names))), np.zeros(len(feature_names)))
        return AdditiveExplanation(base_value, out_value, effects, None, instance, link, model, data)

    # Sample 0: low total  (sum = 1.0)
    # Sample 1: high total (sum = 10.0)
    exp_low = _make_exp([0.5, 0.5])
    exp_high = _make_exp([5.0, 5.0])

    viz = AdditiveForceArrayVisualizer([exp_low, exp_high])

    sim_low = viz.data["explanations"][0]["simIndex"]
    sim_high = viz.data["explanations"][1]["simIndex"]

    assert sim_high < sim_low, (
        f"Higher-prediction sample should come first (lower simIndex), "
        f"got simIndex_high={sim_high}, simIndex_low={sim_low}"
    )


class TestShowFalseReturnsUsableObject:
    """show=False must return a usable matplotlib Axes (matplotlib=True)."""

    def test_returns_axes_instance(self, simple_shap_values):
        base, shap_vals, names = simple_shap_values
        result = shap.plots.force(base, shap_vals, names, matplotlib=True, show=False)
        assert isinstance(result, plt.Axes), f"Expected Axes, got {type(result)}"
        plt.close("all")

    def test_returned_axes_has_correct_figure(self, simple_shap_values):
        base, shap_vals, names = simple_shap_values
        ax = shap.plots.force(base, shap_vals, names, matplotlib=True, show=False)
        assert ax.get_figure() is not None
        plt.close("all")

    def test_js_path_returns_visualizer(self, simple_shap_values):
        """Non-matplotlib path must return a BaseVisualizer (HTML-embeddable)."""
        from shap.plots._force import BaseVisualizer

        base, shap_vals, names = simple_shap_values
        result = shap.plots.force(base, shap_vals, names, matplotlib=False, show=True)
        assert isinstance(result, BaseVisualizer)


class TestNoGlobalStateContamination:
    """force(show=False) must not silently alter shared pyplot state."""

    def test_current_figure_unchanged(self, simple_shap_values):
        base, shap_vals, names = simple_shap_values
        fig_before, ax_before = plt.subplots()
        ax_before.plot([1, 2], [3, 4])
        lines_before = len(ax_before.lines)
        _ = shap.plots.force(base, shap_vals, names, matplotlib=True, show=False)
        assert ax_before.get_figure() is fig_before, "Existing axes must still reference their original figure"
        assert len(ax_before.lines) == lines_before, "Existing axes data must not be modified by force plot call"
        plt.close("all")

    def test_sibling_axes_data_intact(self, simple_shap_values):
        """Data plotted on a sibling axes must survive a force plot call."""
        base, shap_vals, names = simple_shap_values
        fig, (ax1, ax2) = plt.subplots(1, 2)
        x = np.linspace(0, 1, 10)
        ax1.plot(x, x**2, label="sentinel")
        sentinel_lines_before = len(ax1.lines)

        shap.plots.force(base, shap_vals, names, matplotlib=True, show=False, ax=ax2)

        assert len(ax1.lines) == sentinel_lines_before, "Sibling axes lines were modified"
        plt.close("all")


class TestAxParameter:
    """ax parameter must be respected (injected axes are used, not created)."""

    def test_explicit_ax_is_used(self, simple_shap_values):
        base, shap_vals, names = simple_shap_values
        fig, ax = plt.subplots(figsize=(20, 3))
        result = shap.plots.force(base, shap_vals, names, matplotlib=True, show=False, ax=ax)
        assert result is ax, "Returned axes must be the same object passed as ax"
        plt.close("all")

    def test_explicit_ax_figure_unchanged(self, simple_shap_values):
        base, shap_vals, names = simple_shap_values
        fig, ax = plt.subplots(figsize=(20, 3))
        shap.plots.force(base, shap_vals, names, matplotlib=True, show=False, ax=ax)
        assert ax.get_figure() is fig, "Figure reference of supplied ax must not change"
        plt.close("all")

    def test_no_ax_creates_new_figure(self, simple_shap_values):
        base, shap_vals, names = simple_shap_values
        figs_before = set(plt.get_fignums())
        shap.plots.force(base, shap_vals, names, matplotlib=True, show=False)
        figs_after = set(plt.get_fignums())
        assert len(figs_after - figs_before) >= 1, "A new figure should have been created when ax=None"
        plt.close("all")

    def test_figsize_ignored_when_ax_provided(self, simple_shap_values):
        """When ax is supplied, force plot must not resize the caller's figure."""
        base, shap_vals, names = simple_shap_values
        fig, ax = plt.subplots(figsize=(5, 5))
        original_size = fig.get_size_inches().tolist()
        shap.plots.force(base, shap_vals, names, matplotlib=True, show=False, ax=ax)
        assert fig.get_size_inches().tolist() == original_size, "Figure size must remain unchanged when ax is provided"
        plt.close("all")


class TestChainedUsage:
    """Chained calls and subplot grids must not crash."""

    def test_two_force_plots_on_subplots(self, simple_shap_values):
        base, shap_vals, names = simple_shap_values
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(40, 3))
        ax_out1 = shap.plots.force(base, shap_vals, names, matplotlib=True, show=False, ax=ax1)
        ax_out2 = shap.plots.force(base, -shap_vals, names, matplotlib=True, show=False, ax=ax2)
        assert ax_out1 is ax1
        assert ax_out2 is ax2
        plt.close("all")

    def test_force_then_annotate(self, simple_shap_values):
        base, shap_vals, names = simple_shap_values
        ax = shap.plots.force(base, shap_vals, names, matplotlib=True, show=False)
        # Must be able to annotate without error
        ax.set_title("annotated")
        assert ax.get_title() == "annotated"
        plt.close("all")


class TestBackwardCompatibility:
    """Public API surface must be preserved."""

    def test_force_plot_alias(self, simple_shap_values):
        base, shap_vals, names = simple_shap_values
        result = shap.force_plot(base, shap_vals, names, matplotlib=True, show=False)
        assert isinstance(result, plt.Axes)
        plt.close("all")

    def test_contribution_threshold_param(self, simple_shap_values):
        base, shap_vals, names = simple_shap_values
        result = shap.plots.force(base, shap_vals, names, matplotlib=True, show=False, contribution_threshold=0.1)
        assert isinstance(result, plt.Axes)
        plt.close("all")

    def test_text_rotation_param(self, simple_shap_values):
        base, shap_vals, names = simple_shap_values
        result = shap.plots.force(base, shap_vals, names, matplotlib=True, show=False, text_rotation=45)
        assert isinstance(result, plt.Axes)
        plt.close("all")

    def test_plot_cmap_str(self, simple_shap_values):
        base, shap_vals, names = simple_shap_values
        result = shap.plots.force(base, shap_vals, names, matplotlib=True, show=False, plot_cmap="coolwarm")
        assert isinstance(result, plt.Axes)
        plt.close("all")


class TestEmbeddingDoesNotAlterSiblingPlots:
    """JS/HTML path: obtaining HTML representation must not touch pyplot."""

    def test_html_repr_no_pyplot_side_effect(self, simple_shap_values):
        from shap.plots._force import BaseVisualizer

        base, shap_vals, names = simple_shap_values

        # Create a sentinel matplotlib figure before obtaining the HTML
        sentinel_fig, sentinel_ax = plt.subplots()
        sentinel_ax.plot([1, 2], [3, 4])
        n_lines_before = len(sentinel_ax.lines)

        vis = shap.plots.force(base, shap_vals, names, matplotlib=False, show=True)
        assert isinstance(vis, BaseVisualizer)
        html = vis.html()
        assert isinstance(html, str)
        assert len(html) > 0

        # Sentinel must be untouched
        assert len(sentinel_ax.lines) == n_lines_before
        plt.close("all")

    def test_html_contains_script_tag(self, simple_shap_values):
        base, shap_vals, names = simple_shap_values
        vis = shap.plots.force(base, shap_vals, names, matplotlib=False, show=True)
        html = vis.html()
        assert "<script" in html

import matplotlib.pyplot as plt
import numpy as np
import pytest

import shap
from shap.utils._exceptions import DimensionError


def test_violin_with_invalid_plot_type():
    with pytest.raises(ValueError, match="plot_type: Expected one of "):
        shap.plots.violin(np.random.randn(20, 5), plot_type="nonsense")


def test_violin_wrong_features_shape():
    """Checks that DimensionError is raised if the features data matrix
    has an incompatible shape with the shap_values matrix.
    """
    rs = np.random.RandomState(42)

    emsg = (
        "The shape of the shap_values matrix does not match the shape of "
        "the provided data matrix. Perhaps the extra column"
    )
    with pytest.raises(DimensionError, match=emsg):
        expln = shap.Explanation(
            values=rs.randn(20, 5),
            data=rs.randn(20, 4),
        )
        shap.plots.violin(expln, show=False)

    with pytest.raises(DimensionError, match=emsg):
        shap.plots.violin(
            shap_values=rs.randn(20, 5),
            features=rs.randn(20, 4),
            show=False,
        )

    emsg = "The shape of the shap_values matrix does not match the shape of the provided data matrix."
    with pytest.raises(DimensionError, match=emsg):
        expln = shap.Explanation(
            values=rs.randn(20, 5),
            data=rs.randn(20, 1),
        )
        shap.plots.violin(expln, show=False)

    with pytest.raises(DimensionError, match=emsg):
        shap.plots.violin(
            shap_values=rs.randn(20, 5),
            features=rs.randn(20, 1),
            show=False,
        )


@pytest.mark.mpl_image_compare
def test_violin(explainer):
    """Make sure the violin plot is unchanged."""
    fig = plt.figure()
    shap_values = explainer.shap_values(explainer.data)
    shap.plots.violin(shap_values, show=False)
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare(
    filename="test_summary_violin_with_data.png",
    tolerance=5,
)
def test_summary_violin_with_data2():
    """Check a violin chart with shap_values as a np.array."""
    rs = np.random.RandomState(0)
    fig = plt.figure()
    shap.plots.violin(
        rs.standard_normal(size=(20, 5)),
        rs.standard_normal(size=(20, 5)),
        plot_type="violin",
        show=False,
    )
    fig.set_layout_engine("tight")
    return fig


# Currently using the same files as the `test_summary.py` violin tests for comparison
@pytest.mark.mpl_image_compare(
    filename="test_summary_layered_violin_with_data.png",
    tolerance=5,
)
def test_summary_layered_violin_with_data2():
    """Check a layered violin chart with shap_values as a np.array."""
    rs = np.random.RandomState(0)
    fig = plt.figure()
    shap_values = rs.randn(200, 5)
    feats = rs.randn(200, 5)
    shap.plots.violin(
        shap_values,
        feats,
        plot_type="layered_violin",
        show=False,
    )
    fig.set_layout_engine("tight")
    return fig


class TestViolinAxParameter:
    """Verify that the optional ``ax`` parameter behaves correctly."""

    @staticmethod
    def _make_shap_values(n_samples=30, n_features=4, seed=0):
        rs = np.random.RandomState(seed)
        return rs.randn(n_samples, n_features)

    def test_show_false_returns_ax_by_default(self):
        """When ax is not supplied, show=False must return a matplotlib Axes."""
        shap_values = self._make_shap_values()
        fig, _ = plt.subplots()
        result = shap.plots.violin(shap_values, show=False)
        assert isinstance(result, plt.Axes), "violin(show=False) should return a matplotlib Axes object"
        plt.close(fig)

    def test_ax_parameter_draws_into_supplied_axes(self):
        """When ax is supplied, all drawing must go into that specific axes."""
        shap_values = self._make_shap_values()
        fig, axes = plt.subplots(1, 2)
        target_ax = axes[0]
        other_ax = axes[1]

        returned = shap.plots.violin(shap_values, ax=target_ax, show=False)

        assert returned is target_ax, "violin() must return the ax that was supplied"

        assert len(target_ax.get_yticks()) > 0, "Target axes should have y-tick labels after plotting"

        assert len(other_ax.lines) == 0, "Sibling axes must not be modified"
        assert len(other_ax.collections) == 0, "Sibling axes must not be modified"

        plt.close(fig)

    def test_ax_parameter_returns_the_same_axes(self):
        """The returned axes object must be identical (not a copy) to the one passed."""
        shap_values = self._make_shap_values()
        fig, ax = plt.subplots()
        returned = shap.plots.violin(shap_values, ax=ax, show=False)
        assert returned is ax
        plt.close(fig)

    def test_ax_parameter_does_not_resize_figure(self):
        """Supplying ax must not resize the caller's figure."""
        shap_values = self._make_shap_values(n_features=10)
        fig, ax = plt.subplots(figsize=(6, 4))
        original_size = fig.get_size_inches().tolist()

        shap.plots.violin(shap_values, ax=ax, show=False)

        assert fig.get_size_inches().tolist() == original_size, "violin() must not resize a figure when ax is provided"
        plt.close(fig)

    def test_no_ax_resizes_figure_by_default(self):
        """Without ax, figure resizing must still happen (backward-compat guard)."""
        shap_values = self._make_shap_values(n_features=15)
        fig = plt.figure(figsize=(1, 1))  # intentionally tiny
        shap.plots.violin(shap_values, plot_size="auto", show=False)
        final_size = fig.get_size_inches()
        # auto-sizing should produce something taller than 1 inch
        assert final_size[1] > 1, "violin() should auto-resize the figure when ax is not supplied"
        plt.close(fig)

    def test_subplot_grid_independence(self):
        """Violin drawn in one cell of a subplot grid must not affect other cells."""
        shap_values = self._make_shap_values()
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        target = axes[1, 2]

        shap.plots.violin(shap_values, ax=target, show=False)

        for r in range(2):
            for c in range(3):
                if r == 1 and c == 2:
                    continue  # skip the target cell
                sibling = axes[r, c]
                assert len(sibling.lines) == 0, f"axes[{r},{c}] should be untouched"
                assert len(sibling.collections) == 0, f"axes[{r},{c}] should be untouched"

        plt.close(fig)

    def test_multiple_calls_on_same_ax(self):
        """Calling violin twice on the same axes must not raise an exception."""
        rs = np.random.RandomState(1)
        shap_values1 = rs.randn(20, 3)
        shap_values2 = rs.randn(20, 3)
        fig, ax = plt.subplots()
        shap.plots.violin(shap_values1, ax=ax, show=False)
        # Second call should not crash; it overlays on the same axes
        shap.plots.violin(shap_values2, ax=ax, show=False)
        plt.close(fig)

    def test_ax_with_log_scale(self):
        """ax parameter must work correctly in combination with use_log_scale."""
        shap_values = self._make_shap_values()
        fig, ax = plt.subplots()
        returned = shap.plots.violin(shap_values, ax=ax, use_log_scale=True, show=False)
        assert returned is ax
        assert ax.get_xscale() == "symlog"
        plt.close(fig)

    def test_ax_with_color_bar_and_features(self):
        """Color bar must render without error when ax is supplied with feature data."""
        rs = np.random.RandomState(2)
        shap_values = rs.randn(30, 4)
        features = rs.randn(30, 4)
        fig, ax = plt.subplots()

        returned = shap.plots.violin(
            shap_values,
            features=features,
            color_bar=True,
            ax=ax,
            show=False,
        )
        assert returned is ax
        plt.close(fig)

    def test_backward_compat_no_ax_no_show(self):
        """Legacy call pattern (no ax, show=False) must return an Axes, not None.

        This was a pre-existing silent bug that this PR also fixes: previously
        show=False returned None because there was no return statement in that
        branch.  Callers relying on the (broken) None return would see a change
        in behaviour — but returning an Axes is strictly correct and consistent
        with shap.plots.bar.
        """
        shap_values = self._make_shap_values()
        fig = plt.figure()
        result = shap.plots.violin(shap_values, show=False)
        assert result is not None, "violin(show=False) must return an Axes (was returning None before this PR)"
        assert isinstance(result, plt.Axes)
        plt.close(fig)

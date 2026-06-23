"""Tests for the partial_dependence plot function."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

import shap

matplotlib.use("Agg")


@pytest.fixture()
def trained_model_and_data():
    """Create a simple synthetic dataset and trained LinearRegression model."""
    rng = np.random.RandomState(42)
    X = rng.randn(50, 4)
    # y is a linear combination so the model fits perfectly
    y = X @ np.array([1.0, -2.0, 0.5, 3.0]) + 0.5
    model = LinearRegression().fit(X, y)
    feature_names = ["feat_0", "feat_1", "feat_2", "feat_3"]
    return model, X, feature_names


class TestPartialDependenceCustomAx:
    """Regression tests for passing a custom ``ax`` to partial_dependence_plot."""

    def test_each_subplot_gets_correct_feature_label(self, trained_model_and_data):
        """When looping over a 2×2 subplot grid, each axes should display its
        own feature name on the x-axis rather than all features ending up on
        the same (last-active) axes.
        """
        model, X, feature_names = trained_model_and_data
        fig, axes = plt.subplots(2, 2)
        flat_axes = axes.flatten()

        for i, ax in enumerate(flat_axes):
            shap.partial_dependence_plot(
                ind=i,
                model=model.predict,
                data=X,
                feature_names=feature_names,
                ax=ax,
                ice=False,
                show=False,
            )

        # Each subplot should have the corresponding feature as its xlabel
        for i, ax in enumerate(flat_axes):
            assert ax.get_xlabel() == feature_names[i], (
                f"Subplot {i} xlabel is '{ax.get_xlabel()}', expected '{feature_names[i]}'"
            )

        plt.close(fig)

    def test_returns_correct_figure_and_axes(self, trained_model_and_data):
        """When ``show=False`` and a custom ``ax`` is provided, the returned
        figure and axes should correspond to the supplied axes object.
        """
        model, X, feature_names = trained_model_and_data
        fig, axes = plt.subplots(1, 2)

        returned_fig, returned_ax = shap.partial_dependence_plot(
            ind=0,
            model=model.predict,
            data=X,
            feature_names=feature_names,
            ax=axes[0],
            ice=False,
            show=False,
        )

        assert returned_fig is fig
        assert returned_ax is axes[0]

        plt.close(fig)

    def test_default_ax_creates_new_figure(self, trained_model_and_data):
        """When no ``ax`` is given, the function should create its own figure."""
        model, X, feature_names = trained_model_and_data
        plt.close("all")

        returned_fig, returned_ax = shap.partial_dependence_plot(
            ind=0,
            model=model.predict,
            data=X,
            feature_names=feature_names,
            ice=False,
            show=False,
        )

        assert returned_fig is not None
        assert returned_ax is not None
        assert returned_ax.get_xlabel() == feature_names[0]

        plt.close(returned_fig)

"""Tests for shap/plots/_violin.py"""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for testing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import shap
from shap.plots._violin import _trim_crange, shorten_text

# ── helpers ──────────────────────────────────────────────────────────────────


def make_shap_values(n_samples=50, n_features=5, seed=42):
    rng = np.random.RandomState(seed)
    return rng.randn(n_samples, n_features)


def make_features(n_samples=50, n_features=5, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randn(n_samples, n_features)


def make_explanation(n_samples=50, n_features=5):
    shap_values = make_shap_values(n_samples, n_features)
    features = make_features(n_samples, n_features)
    feature_names = [f"feature_{i}" for i in range(n_features)]
    return shap.Explanation(
        values=shap_values,
        data=features,
        feature_names=feature_names,
    )


# ── _trim_crange ──────────────────────────────────────────────────────────────


class TestTrimCrange:
    def test_basic(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 20)
        nan_mask = np.zeros(len(values), dtype=bool)
        vmin, vmax, cvals = _trim_crange(values, nan_mask)
        assert vmin <= vmax

    def test_all_same_values_fallback_to_minmax(self):
        # all equal → vmin == vmax after percentiles → falls to min/max
        values = np.ones(50)
        nan_mask = np.zeros(50, dtype=bool)
        vmin, vmax, cvals = _trim_crange(values, nan_mask)
        assert vmin <= vmax

    def test_returns_correct_shapes(self):
        values = np.arange(100, dtype=float)
        nan_mask = np.zeros(100, dtype=bool)
        nan_mask[0] = True  # one NaN position
        vmin, vmax, cvals = _trim_crange(values, nan_mask)
        # cvals only contains non-nan entries
        assert cvals.shape[0] == 99

    def test_cvals_clipped_to_vmin_vmax(self):
        values = np.linspace(0, 100, 200)
        nan_mask = np.zeros(200, dtype=bool)
        vmin, vmax, cvals = _trim_crange(values, nan_mask)
        assert np.all(cvals >= vmin)
        assert np.all(cvals <= vmax)

    def test_vmin_never_greater_than_vmax(self):
        # rare precision edge-case guard
        rng = np.random.RandomState(99)
        values = rng.randn(30)
        nan_mask = np.zeros(30, dtype=bool)
        vmin, vmax, _ = _trim_crange(values, nan_mask)
        assert vmin <= vmax


# ── shorten_text ──────────────────────────────────────────────────────────────


class TestShortenText:
    def test_short_text_unchanged(self):
        assert shorten_text("hi", 10) == "hi"

    def test_long_text_truncated(self):
        result = shorten_text("hello world", 8)
        assert len(result) == 8
        assert result.endswith("...")

    def test_exact_length_unchanged(self):
        text = "exact"
        assert shorten_text(text, 5) == text

    def test_ellipsis_appended(self):
        result = shorten_text("abcdefgh", 6)
        assert result == "abc..."


# ── violin() — basic smoke tests ──────────────────────────────────────────────


class TestViolinPlot:
    def setup_method(self):
        plt.close("all")

    def teardown_method(self):
        plt.close("all")

    def test_basic_violin_no_features(self):
        shap_values = make_shap_values()
        shap.plots.violin(shap_values, show=False)

    def test_basic_violin_with_numpy_features(self):
        shap_values = make_shap_values()
        features = make_features()
        shap.plots.violin(shap_values, features=features, show=False)

    def test_violin_with_dataframe_features(self):
        shap_values = make_shap_values()
        df = pd.DataFrame(make_features(), columns=[f"f{i}" for i in range(5)])
        shap.plots.violin(shap_values, features=df, show=False)

    def test_violin_with_explanation_object(self):
        exp = make_explanation()
        shap.plots.violin(exp, show=False)

    def test_layered_violin_plot_type(self):
        shap_values = make_shap_values()
        features = make_features()
        shap.plots.violin(shap_values, features=features, plot_type="layered_violin", show=False)

    def test_sort_false(self):
        shap_values = make_shap_values()
        shap.plots.violin(shap_values, sort=False, show=False)

    def test_max_display(self):
        shap_values = make_shap_values()
        shap.plots.violin(shap_values, max_display=3, show=False)

    def test_color_bar_false(self):
        shap_values = make_shap_values()
        features = make_features()
        shap.plots.violin(shap_values, features=features, color_bar=False, show=False)

    def test_use_log_scale(self):
        shap_values = make_shap_values()
        shap.plots.violin(shap_values, use_log_scale=True, show=False)

    def test_custom_feature_names(self):
        shap_values = make_shap_values()
        names = ["alpha", "beta", "gamma", "delta", "epsilon"]
        shap.plots.violin(shap_values, feature_names=names, show=False)

    def test_plot_size_tuple(self):
        shap_values = make_shap_values()
        shap.plots.violin(shap_values, plot_size=(10, 6), show=False)

    def test_plot_size_float(self):
        shap_values = make_shap_values()
        shap.plots.violin(shap_values, plot_size=0.5, show=False)

    def test_plot_size_none(self):
        shap_values = make_shap_values()
        shap.plots.violin(shap_values, plot_size=None, show=False)

    def test_alpha_parameter(self):
        shap_values = make_shap_values()
        shap.plots.violin(shap_values, alpha=0.5, show=False)


# ── violin() — error/edge cases ───────────────────────────────────────────────


class TestViolinErrors:
    def setup_method(self):
        plt.close("all")

    def teardown_method(self):
        plt.close("all")

    def test_raises_on_list_shap_values(self):
        """Multi-output (list) input should raise TypeError."""
        shap_values = [make_shap_values(), make_shap_values()]
        with pytest.raises(TypeError, match="multi-output"):
            shap.plots.violin(shap_values, show=False)

    def test_raises_on_invalid_plot_type(self):
        shap_values = make_shap_values()
        with pytest.raises(ValueError, match="plot_type"):
            shap.plots.violin(shap_values, plot_type="invalid_type", show=False)

    def test_raises_on_1d_shap_values(self):
        shap_values = np.array([1.0, 2.0, 3.0])
        with pytest.raises(AssertionError):
            shap.plots.violin(shap_values, show=False)

    def test_raises_on_feature_shape_mismatch(self):
        shap_values = make_shap_values(n_features=5)
        features = make_features(n_features=3)  # wrong number of features
        with pytest.raises(Exception):
            shap.plots.violin(shap_values, features=features, show=False)

    def test_title_deprecation_warning(self):
        shap_values = make_shap_values()
        with pytest.warns(DeprecationWarning, match="title"):
            shap.plots.violin(shap_values, title="Test Title", show=False)

    def test_features_as_list_sets_feature_names(self):
        """Passing a list as features should treat it as feature_names."""
        shap_values = make_shap_values()
        feature_list = [f"feat_{i}" for i in range(5)]
        # should not raise
        shap.plots.violin(shap_values, features=feature_list, show=False)

    def test_features_with_nan_values(self):
        """NaN values in features should be handled gracefully."""
        shap_values = make_shap_values()
        features = make_features()
        features[0, 0] = np.nan
        shap.plots.violin(shap_values, features=features, show=False)

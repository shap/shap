import types

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pytest

import shap.plots._decision as decision_plot


class _HugeShapValues(np.ndarray):
    """Large-shape view that keeps downstream computations lightweight for warning-path tests."""

    def __new__(cls):
        base = np.lib.stride_tricks.as_strided(np.array([0.0]), shape=(2000, 50001), strides=(0, 0))
        return base.view(cls)

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2 and isinstance(key[1], np.ndarray):
            return np.zeros((2000, 20), dtype=float)
        return super().__getitem__(key)


@pytest.fixture
def suppress_plot(monkeypatch):
    monkeypatch.setattr(decision_plot, "__decision_plot_matplotlib", lambda *args, **kwargs: None)


def test_change_shap_base_value_interaction_cube_branch():
    shap_values = np.zeros((1, 2, 2), dtype=float)
    shifted = decision_plot.__change_shap_base_value(3.0, 1.0, shap_values)

    np.testing.assert_allclose(shifted[0, 0, 0], 2.0 / 3.0)
    np.testing.assert_allclose(shifted[0, 1, 1], 2.0 / 3.0)
    np.testing.assert_allclose(shifted[0, 0, 1], 1.0 / 3.0)
    np.testing.assert_allclose(shifted[0, 1, 0], 1.0 / 3.0)


def test_decision_plot_matplotlib_highlight_text_overflow_legend_and_show(monkeypatch):
    fig, ax = plt.subplots()
    shown = []

    monkeypatch.setattr(plt, "show", lambda: shown.append(True))

    decision_plot.__decision_plot_matplotlib(
        base_value=0.0,
        cumsum=np.array([[0.0, 0.05]]),
        ascending=True,
        feature_display_count=1,
        features=np.array([["very_long_feature_value_" * 15]], dtype=object),
        feature_names=["f0"],
        highlight=np.array([0]),
        plot_color=cm.get_cmap("viridis"),
        axis_color="#333333",
        y_demarc_color="#333333",
        xlim=(0.0, 0.1),
        alpha=1.0,
        color_bar=False,
        auto_size_plot=False,
        title="title",
        show=True,
        legend_labels=["obs0"],
        legend_location="best",
    )

    assert shown == [True]
    assert ax.get_legend() is not None
    assert ax.yaxis_inverted()
    assert ax.texts
    assert ax.texts[0].get_ha() == "left"
    assert np.isclose(ax.texts[0].get_position()[0], 0.0)

    plt.close(fig)


def test_decision_base_value_array_and_feature_list_branch(suppress_plot):
    result = decision_plot.decision(
        np.array([0.0]),
        np.array([[0.1, -0.2]]),
        features=["f0", "f1"],
        show=False,
        return_objects=True,
    )

    assert result is not None
    assert result.base_value == 0.0


def test_decision_features_from_1d_array_branch(suppress_plot):
    result = decision_plot.decision(
        0.0,
        np.array([[0.1, 0.2]]),
        features=np.array(["f0", "f1"], dtype=object),
        show=False,
        return_objects=True,
    )

    assert result is not None
    assert result.feature_names == ["f0", "f1"]


def test_decision_raises_for_invalid_base_or_shap_type(suppress_plot):
    with pytest.raises(TypeError, match="Looks like multi output"):
        decision_plot.decision([0.0], np.array([[0.1, 0.2]]), show=False)

    with pytest.raises(TypeError, match="wrong type"):
        decision_plot.decision(0.0, "bad", show=False)


def test_decision_feature_and_name_validation_errors(suppress_plot):
    class _UnsupportedFeatures:
        ndim = 2

    with pytest.raises(TypeError, match="unsupported type"):
        decision_plot.decision(0.0, np.array([[0.1, 0.2]]), features=_UnsupportedFeatures(), show=False)

    with pytest.raises(ValueError, match="must include all features"):
        decision_plot.decision(0.0, np.array([[0.1, 0.2]]), feature_names=["f0"], show=False)

    with pytest.raises(TypeError, match="requires a list or numpy array"):
        decision_plot.decision(0.0, np.array([[0.1, 0.2]]), feature_names=("f0", "f1"), show=False)


def test_decision_feature_order_branches(suppress_plot, monkeypatch):
    r_list = decision_plot.decision(
        0.0,
        np.array([[0.1, 0.2]]),
        feature_order=[1, 0],
        show=False,
        return_objects=True,
    )
    assert r_list is not None
    assert r_list.feature_idx.tolist() == [1, 0]

    r_array = decision_plot.decision(
        0.0,
        np.array([[0.1, 0.2]]),
        feature_order=np.array([0, 1]),
        show=False,
        return_objects=True,
    )
    assert r_array is not None
    assert r_array.feature_idx.tolist() == [0, 1]

    r_none = decision_plot.decision(
        0.0,
        np.array([[0.1, 0.2]]),
        feature_order=None,
        show=False,
        return_objects=True,
    )
    assert r_none is not None
    assert r_none.feature_idx.tolist() == [0, 1]

    monkeypatch.setattr(decision_plot, "hclust_ordering", lambda x: [1, 0])
    r_hclust = decision_plot.decision(
        0.0,
        np.array([[0.1, 0.2]]),
        feature_order="hclust",
        show=False,
        return_objects=True,
    )
    assert r_hclust is not None
    assert r_hclust.feature_idx.tolist() == [1, 0]

    with pytest.raises(ValueError, match="feature_order"):
        decision_plot.decision(0.0, np.array([[0.1, 0.2]]), feature_order="bad", show=False)

    with pytest.raises(ValueError, match="data type must be integer"):
        decision_plot.decision(0.0, np.array([[0.1, 0.2]]), feature_order=[0.0, 1.0], show=False)


def test_decision_feature_display_range_validation_and_range_conversion(suppress_plot, monkeypatch):
    with pytest.raises(TypeError, match="requires a slice or a range"):
        decision_plot.decision(0.0, np.array([[0.1, 0.2]]), feature_display_range=5, show=False)

    with pytest.raises(ValueError, match="supports a step"):
        decision_plot.decision(0.0, np.array([[0.1, 0.2]]), feature_display_range=slice(None, None, 2), show=False)

    monkeypatch.setattr(decision_plot.np, "iinfo", lambda _dtype: types.SimpleNamespace(min=-999999999))

    r = decision_plot.decision(
        0.0,
        np.array([[0.1, 0.2, 0.3]]),
        feature_order=None,
        feature_display_range=range(2, -1, -1),
        show=False,
        return_objects=True,
    )
    assert r is not None


def test_decision_large_data_runtime_guards(suppress_plot):
    with pytest.raises(RuntimeError, match="Plotting 2001 observations"):
        decision_plot.decision(0.0, np.zeros((2001, 1)), show=False)

    with pytest.raises(RuntimeError, match="Plotting 201 features"):
        decision_plot.decision(
            0.0,
            np.zeros((1, 201)),
            feature_order=None,
            feature_display_range=slice(None, None, 1),
            show=False,
        )

    with pytest.raises(RuntimeError, match="Processing SHAP values"):
        decision_plot.decision(0.0, _HugeShapValues(), feature_order=None, show=False)


def test_decision_logit_link_and_identity_xlim_branches(suppress_plot):
    r_logit = decision_plot.decision(
        0.0,
        np.array([[0.1, -0.2]]),
        link="logit",
        show=False,
        return_objects=True,
    )
    assert r_logit is not None
    assert r_logit.xlim == (-0.02, 1.02)

    r_identity = decision_plot.decision(
        0.0,
        np.array([[-5.0, 1.0]]),
        feature_order=None,
        show=False,
        return_objects=True,
    )
    assert r_identity is not None
    assert r_identity.xlim[0] < -5.0
    assert r_identity.xlim[1] < 1.0


def test_multioutput_decision_validation_errors():
    with pytest.raises(ValueError, match="list of scalars"):
        decision_plot.multioutput_decision([["a"], ["b"]], [np.zeros((2, 2)), np.ones((2, 2))], row_index=0)

    with pytest.raises(ValueError, match="output length is different"):
        decision_plot.multioutput_decision(
            [0.0, 1.0],
            [np.zeros((2, 2)), np.ones((2, 2)), np.ones((2, 2))],
            row_index=0,
        )


def test_multioutput_decision_slices_numpy_features_row(monkeypatch):
    captured = {}

    def fake_decision(base_value, shap_values, **kwargs):
        captured["base_value"] = base_value
        captured["shap_values"] = shap_values
        captured["features"] = kwargs.get("features")
        return kwargs.get("features")

    monkeypatch.setattr(decision_plot, "decision", fake_decision)

    base_values = [0.0, 2.0]
    shap_values = [
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        np.array([[2.0, 1.0], [4.0, 3.0], [6.0, 5.0]]),
    ]
    features = np.array([[10, 11], [20, 21], [30, 31]])

    out = decision_plot.multioutput_decision(base_values, shap_values, row_index=1, features=features)

    assert out.shape == (1, 2)
    np.testing.assert_array_equal(captured["features"], np.array([[20, 21]]))
    assert captured["shap_values"].shape == (2, 2)

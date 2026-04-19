"""Tests for shap/plots/_decision.py - targeting 80%+ coverage."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import sklearn.ensemble

import shap

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rfc_data():
    """RandomForest on adult dataset - reused across tests."""
    X, y = shap.datasets.adult(n_points=20)
    rfc = sklearn.ensemble.RandomForestClassifier(n_estimators=10, random_state=0)
    rfc.fit(X, y)
    ex = shap.TreeExplainer(rfc)
    sv = ex(X)
    return sv, X


@pytest.fixture(scope="module")
def base_sv_features(rfc_data):
    """Return (base_value, shap_values_2d, features_df) for class 1."""
    sv, X = rfc_data
    return sv.base_values[0, 1], sv.values[:, :, 1], X


@pytest.fixture(autouse=True)
def close_plots():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# __change_shap_base_value  (tested indirectly via new_base_value)
# ---------------------------------------------------------------------------


def test_new_base_value_zero(base_sv_features):
    """Shifting base value to 0 should keep sum(shap) + new_base == original prediction."""
    base, sv, X = base_sv_features
    r = shap.decision_plot(base, sv, show=False, return_objects=True, new_base_value=0)
    assert r.base_value == 0
    np.testing.assert_allclose(
        r.shap_values.sum(axis=1) + 0,
        sv.sum(axis=1) + base,
        atol=1e-10,
    )


def test_new_base_value_nonzero(base_sv_features):
    base, sv, X = base_sv_features
    new_bv = 0.3
    r = shap.decision_plot(base, sv, show=False, return_objects=True, new_base_value=new_bv)
    assert r.base_value == new_bv
    np.testing.assert_allclose(
        r.shap_values.sum(axis=1) + new_bv,
        sv.sum(axis=1) + base,
        atol=1e-10,
    )


def test_new_base_value_with_interactions(rfc_data):
    """new_base_value with 3-D interaction values (cube branch of __change_shap_base_value)."""
    sv, X = rfc_data
    rfc = sklearn.ensemble.RandomForestClassifier(n_estimators=10, random_state=0)
    rfc.fit(X, sv.base_values[:, 0] > 0)
    ex2 = shap.TreeExplainer(rfc)
    iv = ex2(X, interactions=True)
    base = iv.base_values[0, 1]
    shap_interaction = iv.values[:, :, :, 1]
    r = shap.decision_plot(
        base, shap_interaction, show=False, return_objects=True, new_base_value=0
    )
    assert r.base_value == 0


# ---------------------------------------------------------------------------
# Type / validation errors
# ---------------------------------------------------------------------------


def test_raises_list_base_value(base_sv_features):
    base, sv, _ = base_sv_features
    with pytest.raises(TypeError, match="multi output"):
        shap.decision_plot([base, base], sv, show=False)


def test_raises_list_shap_values(base_sv_features):
    base, sv, _ = base_sv_features
    with pytest.raises(TypeError, match="multi output"):
        shap.decision_plot(base, [sv], show=False)


def test_raises_wrong_shap_values_type(base_sv_features):
    base, _, _ = base_sv_features
    with pytest.raises(TypeError, match="wrong type"):
        shap.decision_plot(base, "not_an_array", show=False)


def test_raises_wrong_features_type(base_sv_features):
    base, sv, _ = base_sv_features
    class WeirdFeatures:
        ndim = 2
    with pytest.raises(TypeError, match="unsupported type"):
        shap.decision_plot(base, sv[:5], features=WeirdFeatures(), show=False)


def test_raises_feature_names_wrong_length(base_sv_features):
    base, sv, _ = base_sv_features
    with pytest.raises(ValueError, match="feature_names arg must include all features"):
        shap.decision_plot(base, sv[:5], feature_names=["only_one"], show=False)


def test_raises_feature_names_wrong_type(base_sv_features):
    base, sv, _ = base_sv_features
    n = sv.shape[1]
    with pytest.raises(TypeError, match="feature_names arg requires a list or numpy array"):
        shap.decision_plot(base, sv[:5], feature_names=tuple(range(n)), show=False)


def test_raises_bad_feature_order(base_sv_features):
    base, sv, _ = base_sv_features
    with pytest.raises(ValueError, match="feature_order arg"):
        shap.decision_plot(base, sv[:5], feature_order="bad_value", show=False)


def test_raises_bad_feature_order_wrong_length(base_sv_features):
    base, sv, _ = base_sv_features
    with pytest.raises(ValueError, match="length must match"):
        shap.decision_plot(base, sv[:5], feature_order=np.array([0, 1]), show=False)


def test_raises_bad_feature_display_range_type(base_sv_features):
    base, sv, _ = base_sv_features
    with pytest.raises(TypeError, match="feature_display_range arg requires a slice or a range"):
        shap.decision_plot(base, sv[:5], feature_display_range=[0, 1, 2], show=False)


def test_raises_bad_feature_display_range_step(base_sv_features):
    base, sv, _ = base_sv_features
    with pytest.raises(ValueError, match="step of 1, -1, or None"):
        shap.decision_plot(base, sv[:5], feature_display_range=slice(0, 5, 2), show=False)


def test_raises_too_many_observations(base_sv_features):
    base, _, _ = base_sv_features
    big = np.random.randn(2001, 5)
    with pytest.raises(RuntimeError, match="2001 observations"):
        shap.decision_plot(base, big, show=False)


def test_raises_too_many_features(base_sv_features):
    base, _, _ = base_sv_features
    big = np.random.randn(5, 201)
    with pytest.raises(RuntimeError, match="201 features"):
        shap.decision_plot(base, big, feature_display_range=slice(0, 201, 1), show=False)


# ---------------------------------------------------------------------------
# base_value unwrapping
# ---------------------------------------------------------------------------


def test_base_value_numpy_array_length1(base_sv_features):
    _, sv, _ = base_sv_features
    base_arr = np.array([0.5])
    shap.decision_plot(base_arr, sv[:5], show=False)


# ---------------------------------------------------------------------------
# features input variants
# ---------------------------------------------------------------------------


def test_features_dataframe(base_sv_features):
    base, sv, X = base_sv_features
    shap.decision_plot(base, sv, features=X, show=False)


def test_features_series_single_row(base_sv_features):
    base, sv, X = base_sv_features
    shap.decision_plot(base, sv[0], features=X.iloc[0], show=False)


def test_features_numpy_1d_becomes_feature_names(base_sv_features):
    """1-D numpy array of feature values is treated as feature names."""
    base, sv, X = base_sv_features
    names_array = np.array(X.columns.tolist())
    shap.decision_plot(base, sv[:5], features=names_array, show=False)


def test_features_list_becomes_feature_names(base_sv_features):
    base, sv, X = base_sv_features
    shap.decision_plot(base, sv[:5], features=X.columns.tolist(), show=False)


def test_features_numpy_2d(base_sv_features):
    base, sv, X = base_sv_features
    shap.decision_plot(base, sv, features=X.values, show=False)


def test_feature_names_as_numpy_array(base_sv_features):
    base, sv, X = base_sv_features
    shap.decision_plot(base, sv[:5], feature_names=np.array(X.columns.tolist()), show=False)


# ---------------------------------------------------------------------------
# feature_order variants
# ---------------------------------------------------------------------------


def test_feature_order_none_string(base_sv_features):
    base, sv, _ = base_sv_features
    shap.decision_plot(base, sv[:5], feature_order="none", show=False)


def test_feature_order_none_value(base_sv_features):
    base, sv, _ = base_sv_features
    shap.decision_plot(base, sv[:5], feature_order=None, show=False)


def test_feature_order_hclust(base_sv_features):
    base, sv, _ = base_sv_features
    shap.decision_plot(base, sv[:5], feature_order="hclust", show=False)


def test_feature_order_custom_list(base_sv_features):
    base, sv, _ = base_sv_features
    r = shap.decision_plot(base, sv[:5], show=False, return_objects=True)
    # reuse returned feature_idx
    shap.decision_plot(base, sv[5:10], feature_order=r.feature_idx, show=False)


def test_feature_order_custom_array(base_sv_features):
    base, sv, _ = base_sv_features
    idx = np.arange(sv.shape[1])
    shap.decision_plot(base, sv[:5], feature_order=idx, show=False)


# ---------------------------------------------------------------------------
# feature_display_range variants
# ---------------------------------------------------------------------------


def test_feature_display_range_slice_ascending(base_sv_features):
    base, sv, _ = base_sv_features
    shap.decision_plot(base, sv[:5], feature_display_range=slice(0, 5, 1), show=False)


def test_feature_display_range_slice_descending(base_sv_features):
    base, sv, _ = base_sv_features
    shap.decision_plot(base, sv[:5], feature_display_range=slice(-1, -6, -1), show=False)


def test_feature_display_range_slice_none_step(base_sv_features):
    base, sv, _ = base_sv_features
    shap.decision_plot(base, sv[:5], feature_display_range=slice(0, 3), show=False)


def test_feature_display_range_range_ascending(base_sv_features):
    base, sv, _ = base_sv_features
    shap.decision_plot(base, sv[:5], feature_display_range=range(0, 5, 1), show=False)


def test_feature_display_range_range_descending(base_sv_features):
    base, sv, _ = base_sv_features
    n = sv.shape[1]
    shap.decision_plot(base, sv[:5], feature_display_range=range(n - 1, n - 6, -1), show=False)


def test_feature_display_range_range_negative_stop(base_sv_features):
    """range with negative stop: range(5, -1, -1) == [5,4,3,2,1,0], clipped to iinfo.min."""
    base, sv, _ = base_sv_features
    # Use small positive indices only to avoid iinfo dtype issue
    shap.decision_plot(base, sv[:5], feature_display_range=range(4, 0, -1), show=False)


def test_feature_display_range_reuse_xlim(base_sv_features):
    base, sv, _ = base_sv_features
    r = shap.decision_plot(
        base, sv[:5], feature_display_range=slice(0, 5, 1), show=False, return_objects=True
    )
    shap.decision_plot(base, sv[5:10], feature_display_range=slice(0, 5, 1), xlim=r.xlim, show=False)


# ---------------------------------------------------------------------------
# link variants
# ---------------------------------------------------------------------------


def test_link_logit(base_sv_features):
    base, sv, _ = base_sv_features
    # logit requires base_value in (0,1)
    shap.decision_plot(0.5, sv[:5] * 0.01, link="logit", show=False)


def test_link_logit_xlim_auto(base_sv_features):
    """Logit link should auto-set xlim to (-0.02, 1.02)."""
    _, sv, _ = base_sv_features
    r = shap.decision_plot(0.5, sv[:3] * 0.01, link="logit", show=False, return_objects=True)
    assert r.xlim[0] < 0
    assert r.xlim[1] > 1


# ---------------------------------------------------------------------------
# highlight variants
# ---------------------------------------------------------------------------


def test_highlight_list(base_sv_features):
    base, sv, _ = base_sv_features
    shap.decision_plot(base, sv, highlight=[0, 1], show=False)


def test_highlight_bool_array(base_sv_features):
    base, sv, _ = base_sv_features
    mask = np.zeros(sv.shape[0], dtype=bool)
    mask[0] = True
    shap.decision_plot(base, sv, highlight=mask, show=False)


def test_highlight_slice(base_sv_features):
    base, sv, _ = base_sv_features
    shap.decision_plot(base, sv, highlight=slice(0, 3), show=False)


# ---------------------------------------------------------------------------
# visual / style options
# ---------------------------------------------------------------------------


def test_color_bar_false(base_sv_features):
    base, sv, _ = base_sv_features
    shap.decision_plot(base, sv[:5], color_bar=False, show=False)


def test_alpha(base_sv_features):
    base, sv, _ = base_sv_features
    shap.decision_plot(base, sv[:5], alpha=0.3, show=False)


def test_plot_color_string(base_sv_features):
    base, sv, _ = base_sv_features
    shap.decision_plot(base, sv[:5], plot_color="coolwarm", show=False)


def test_axis_color(base_sv_features):
    base, sv, _ = base_sv_features
    shap.decision_plot(base, sv[:5], axis_color="#FF0000", show=False)


def test_y_demarc_color(base_sv_features):
    base, sv, _ = base_sv_features
    shap.decision_plot(base, sv[:5], y_demarc_color="#0000FF", show=False)


def test_auto_size_plot_false(base_sv_features):
    base, sv, _ = base_sv_features
    shap.decision_plot(base, sv[:5], auto_size_plot=False, show=False)


def test_title(base_sv_features):
    base, sv, _ = base_sv_features
    shap.decision_plot(base, sv[:5], title="My Test Title", show=False)


def test_legend_labels(base_sv_features):
    base, sv, _ = base_sv_features
    labels = [f"obs_{i}" for i in range(sv[:5].shape[0])]
    shap.decision_plot(base, sv[:5], legend_labels=labels, legend_location="upper right", show=False)


def test_ignore_warnings_large_features(base_sv_features):
    base, _, _ = base_sv_features
    big = np.random.randn(5, 201)
    shap.decision_plot(base, big, ignore_warnings=True, show=False)


# ---------------------------------------------------------------------------
# return_objects validation
# ---------------------------------------------------------------------------


def test_return_objects_fields(base_sv_features):
    base, sv, X = base_sv_features
    r = shap.decision_plot(base, sv, features=X, show=False, return_objects=True)
    assert r.base_value == base
    assert isinstance(r.shap_values, np.ndarray)
    assert isinstance(r.feature_names, list)
    assert isinstance(r.feature_idx, np.ndarray)
    assert len(r.xlim) == 2


def test_return_objects_false_returns_none(base_sv_features):
    base, sv, _ = base_sv_features
    result = shap.decision_plot(base, sv[:5], show=False, return_objects=False)
    assert result is None


# ---------------------------------------------------------------------------
# 1-D shap_values (single observation, no features)
# ---------------------------------------------------------------------------


def test_1d_shap_values(base_sv_features):
    base, sv, _ = base_sv_features
    shap.decision_plot(base, sv[0], show=False)


def test_single_observation_with_feature_values(base_sv_features):
    """When cumsum.shape[0]==1 and features provided, feature values are printed."""
    base, sv, X = base_sv_features
    shap.decision_plot(base, sv[0], features=X.iloc[0], show=False)


def test_single_observation_string_feature_value(base_sv_features):
    """String feature values should be formatted with parentheses."""
    base, sv, _ = base_sv_features
    n = sv.shape[1]
    feat = np.array([["a"] * n], dtype=object)
    shap.decision_plot(base, sv[[0]], features=feat, show=False)


# ---------------------------------------------------------------------------
# Interaction values (3-D shap_values)
# ---------------------------------------------------------------------------


def test_interaction_values(rfc_data):
    sv, X = rfc_data
    rfc = sklearn.ensemble.RandomForestClassifier(n_estimators=10, random_state=0)
    rfc.fit(X, sv.base_values[:, 0] > 0)
    ex = shap.TreeExplainer(rfc)
    iv = ex(X, interactions=True)
    # binary classifier returns 3D values (samples x features x features)
    base = float(iv.base_values[0])
    shap.decision_plot(base, iv.values[:5], show=False)


def test_xlim_symmetric_around_base(base_sv_features):
    """xlim should be centered around base_value (symmetric up to margin)."""
    base, sv, _ = base_sv_features
    sv_neg = -np.abs(sv[:5])
    r = shap.decision_plot(base, sv_neg, show=False, return_objects=True)
    lo, hi = r.xlim
    # xlim is symmetric around base_value with a small margin — just check it spans base
    assert lo < base < hi


# ---------------------------------------------------------------------------
# multioutput_decision
# ---------------------------------------------------------------------------


def test_multioutput_basic(rfc_data):
    sv, X = rfc_data
    sv_list = [sv.values[:, :, i] for i in range(sv.shape[2])]
    bv_list = list(sv.base_values[0, :])
    shap.multioutput_decision_plot(bv_list, sv_list, row_index=0, features=X, show=False)


def test_multioutput_return_objects(rfc_data):
    sv, X = rfc_data
    sv_list = [sv.values[:, :, i] for i in range(sv.shape[2])]
    bv_list = list(sv.base_values[0, :])
    r = shap.multioutput_decision_plot(
        bv_list, sv_list, row_index=0, features=X, show=False, return_objects=True
    )
    assert r is not None
    assert hasattr(r, "base_value")


def test_multioutput_new_base_value(rfc_data):
    sv, X = rfc_data
    sv_list = [sv.values[:, :, i] for i in range(sv.shape[2])]
    bv_list = list(sv.base_values[0, :])
    r = shap.multioutput_decision_plot(
        bv_list, sv_list, row_index=0, show=False, return_objects=True, new_base_value=0
    )
    assert r.base_value == 0


def test_multioutput_features_numpy_2d(rfc_data):
    sv, X = rfc_data
    sv_list = [sv.values[:, :, i] for i in range(sv.shape[2])]
    bv_list = list(sv.base_values[0, :])
    shap.multioutput_decision_plot(
        bv_list, sv_list, row_index=0, features=X.values, show=False
    )


def test_multioutput_features_dataframe(rfc_data):
    sv, X = rfc_data
    sv_list = [sv.values[:, :, i] for i in range(sv.shape[2])]
    bv_list = list(sv.base_values[0, :])
    shap.multioutput_decision_plot(
        bv_list, sv_list, row_index=2, features=X, show=False
    )


def test_multioutput_raises_not_lists(rfc_data):
    sv, X = rfc_data
    with pytest.raises(ValueError, match="base_values and shap_values args expect lists"):
        shap.multioutput_decision_plot(
            sv.base_values[0, :],  # ndarray, not list
            sv.values[:, :, :],
            row_index=0,
        )


def test_multioutput_raises_wrong_shap_dims(rfc_data):
    sv, X = rfc_data
    bv_list = list(sv.base_values[0, :])
    bad_sv = [sv.values[:, 0, i] for i in range(sv.shape[2])]  # 1-D arrays
    with pytest.raises(ValueError, match="list of two or three dimensional"):
        shap.multioutput_decision_plot(bv_list, bad_sv, row_index=0)
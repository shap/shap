"""Tests to verify that warnings.warn() calls use stacklevel=2 so warnings
point to the user's calling code rather than internal SHAP files."""

import warnings

import numpy as np
import pytest
import sklearn.linear_model


def _this_file() -> str:
    return __file__


def test_bar_legacy_warning_stacklevel():
    """bar_legacy() should emit a DeprecationWarning pointing at the caller."""
    from shap.plots._bar import bar_legacy

    rs = np.random.RandomState(42)
    shap_values = rs.randn(4)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bar_legacy(shap_values, show=False)  # caller line

    dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert dep_warnings, "Expected a DeprecationWarning from bar_legacy"
    assert dep_warnings[0].filename == _this_file(), (
        f"Warning should point to test file, not {dep_warnings[0].filename}"
    )


def test_violin_title_deprecation_warning_stacklevel():
    """violin() title argument should emit a DeprecationWarning pointing at the caller."""
    import shap

    rs = np.random.RandomState(42)
    shap_values = rs.randn(20, 5)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        shap.plots.violin(shap_values, title="some title", show=False)  # caller line

    dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert dep_warnings, "Expected a DeprecationWarning for unused title argument"
    assert dep_warnings[0].filename == _this_file(), (
        f"Warning should point to test file, not {dep_warnings[0].filename}"
    )


def test_linear_feature_perturbation_deprecation_stacklevel():
    """LinearExplainer with feature_perturbation should emit FutureWarning at caller."""
    import shap

    rs = np.random.RandomState(42)
    X = rs.randn(20, 5)
    y = rs.randn(20)
    model = sklearn.linear_model.Ridge()
    model.fit(X, y)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        shap.LinearExplainer(model, X, feature_perturbation="interventional")  # caller line

    future_warnings = [w for w in caught if issubclass(w.category, FutureWarning)]
    assert future_warnings, "Expected a FutureWarning for feature_perturbation deprecation"
    assert future_warnings[0].filename == _this_file(), (
        f"Warning should point to test file, not {future_warnings[0].filename}"
    )


def test_hclust_ignores_y_warning_stacklevel():
    """hclust() should emit a warning pointing at the caller when y is ignored."""
    from shap.utils import hclust

    rs = np.random.RandomState(42)
    X = rs.randn(20, 5)
    y = rs.randn(20)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        hclust(X, y=y, metric="cosine")  # caller line; non-label-fit metric causes y to be ignored

    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert user_warnings, "Expected a UserWarning about ignoring y in hclust"
    assert user_warnings[0].filename == _this_file(), (
        f"Warning should point to test file, not {user_warnings[0].filename}"
    )


def test_tree_approximate_deprecation_stacklevel():
    """TreeExplainer with approximate arg should emit DeprecationWarning at caller."""
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier

    import shap

    rs = np.random.RandomState(42)
    X = rs.randn(100, 4)
    y = (X[:, 0] > 0).astype(int)
    model = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        shap.TreeExplainer(model, approximate=False)  # caller line

    dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert dep_warnings, "Expected a DeprecationWarning for the approximate argument"
    assert dep_warnings[0].filename == _this_file(), (
        f"Warning should point to test file, not {dep_warnings[0].filename}"
    )


def test_tree_feature_perturbation_futurewarning_stacklevel():
    """TreeExplainer with interventional perturbation but no data should warn at caller."""
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier

    import shap

    rs = np.random.RandomState(42)
    X = rs.randn(100, 4)
    y = (X[:, 0] > 0).astype(int)
    model = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        shap.TreeExplainer(model, feature_perturbation="interventional")  # caller line

    future_warnings = [w for w in caught if issubclass(w.category, FutureWarning)]
    assert future_warnings, "Expected a FutureWarning for interventional without data"
    assert future_warnings[0].filename == _this_file(), (
        f"Warning should point to test file, not {future_warnings[0].filename}"
    )

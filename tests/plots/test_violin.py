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
    # legacy API
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
    # legacy API
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


# FIXME: remove once we migrate violin completely to the Explanation object
# ------ "legacy" violin plots -------
# Currently using the same files as the `test_summary.py` violin tests for comparison
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


def test_violin_with_title_warning():
    """Test that title parameter triggers deprecation warning."""
    rs = np.random.RandomState(42)
    with pytest.warns(DeprecationWarning, match="The `title` argument is unused"):
        shap.plots.violin(rs.randn(20, 5), title="Test Title", show=False)
    plt.close()


def test_violin_multioutput_raises_error():
    """Test that multi-output explanations raise TypeError."""
    rs = np.random.RandomState(42)
    with pytest.raises(TypeError, match="Violin plots don't support multi-output explanations"):
        shap.plots.violin([rs.randn(20, 5), rs.randn(20, 5)], show=False)


def test_violin_plot_type_none():
    """Test violin plot with plot_type=None defaults to violin."""
    rs = np.random.RandomState(42)
    shap.plots.violin(rs.randn(20, 5), plot_type=None, show=False)
    plt.close()


@pytest.mark.mpl_image_compare
def test_violin_features_as_dataframe():
    """Test violin plot with features as DataFrame."""
    pytest.importorskip("pandas")
    import pandas as pd

    rs = np.random.RandomState(42)
    shap_values = rs.randn(50, 3)
    features_df = pd.DataFrame(rs.randn(50, 3), columns=["A", "B", "C"])
    fig = plt.figure()
    shap.plots.violin(shap_values, features=features_df, show=False)
    plt.tight_layout()
    return fig


def test_violin_features_as_list():
    """Test violin plot with features as list of feature names."""
    rs = np.random.RandomState(42)
    shap_values = rs.randn(20, 3)
    feature_names_list = ["Feature1", "Feature2", "Feature3"]
    shap.plots.violin(shap_values, features=feature_names_list, show=False)
    plt.close()


def test_violin_features_as_1d_array():
    """Test violin plot with features as 1D array (feature names)."""
    rs = np.random.RandomState(42)
    shap_values = rs.randn(20, 3)
    feature_names_array = np.array(["F1", "F2", "F3"])
    shap.plots.violin(shap_values, features=feature_names_array, show=False)
    plt.close()


def test_violin_show_true(monkeypatch):
    """Test violin plot with show=True."""
    rs = np.random.RandomState(42)
    show_called = []
    monkeypatch.setattr(plt, "show", lambda: show_called.append(True))
    shap.plots.violin(rs.randn(20, 5), show=True)
    assert len(show_called) == 1
    plt.close()

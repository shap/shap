import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

import shap
from shap.utils import hclust
from shap.utils._clustering import (
    partition_tree,
)
from shap.utils._exceptions import DimensionError


@pytest.mark.parametrize("linkage", ["single", "complete", "average"])
def test_hclust_runs(linkage):
    # GH #3290
    pytest.importorskip("xgboost")
    X = np.column_stack((np.arange(1, 10), np.arange(100, 1000, step=100)))
    y = np.where(X[:, 0] > 5, 1, 0)

    # just check if clustered ran successfully (using xgboost_distances_r2)
    clustered = hclust(X, y, linkage=linkage, random_state=0)
    assert isinstance(clustered, np.ndarray)
    assert clustered.shape == (1, 4)

    # Check clustering runs if y=None (using scipy metrics)
    clustered = hclust(X, linkage=linkage, random_state=0)
    assert isinstance(clustered, np.ndarray)
    assert clustered.shape == (1, 4)


@pytest.mark.parametrize(
    "X",
    [
        np.arange(1, 10),
        list(range(1, 10)),
    ],
)
def test_hclust_errors_on_input_shapes(X):
    # hclust only accepts 2-d arrays for X
    with pytest.raises(DimensionError):
        hclust(X, random_state=0)


def test_hclust_errors_on_unknown_linkages():
    X = np.column_stack((np.arange(1, 10), np.arange(100, 1000, step=100)))
    with pytest.raises(ValueError, match=r"Unknown linkage type:"):
        hclust(X, linkage="random-string", random_state=0)  # type: ignore


def test_hclust_with_dataframe_input():
    """hclust should accept a pandas DataFrame as input."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "c": [7.0, 8.0, 9.0]})
    result = hclust(df, random_state=0)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 4)


def test_hclust_with_nan_values():
    """hclust should handle NaN values in input by replacing them with column means."""
    X = np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]])
    result = hclust(X, random_state=0)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 4)


def test_hclust_warns_when_y_passed_with_scipy_metric():
    """hclust should warn when y is provided but metric is not xgboost-based."""
    X = np.column_stack((np.arange(1, 10), np.arange(100, 1000, step=100)))
    y = np.ones(9)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        hclust(X, y=y, metric="cosine", random_state=0)
        assert any("Ignoring the y argument" in str(warning.message) for warning in w)


def test_partition_masker_calls_hclust():
    """Partition masker with a string metric should internally call hclust and produce valid clustering."""
    rs = np.random.RandomState(42)
    X = pd.DataFrame(rs.randn(50, 4), columns=["a", "b", "c", "d"])
    masker = shap.maskers.Partition(X, clustering="correlation")
    assert masker.clustering is not None
    assert isinstance(masker.clustering, np.ndarray)
    # linkage matrix: (n_features - 1, 4)
    assert masker.clustering.shape == (3, 4)


def test_permutation_explainer_with_clustered_masker():
    """PermutationExplainer with a Partition masker exercises partition_tree_shuffle internally."""
    rs = np.random.RandomState(0)
    X = pd.DataFrame(rs.randn(50, 4), columns=["a", "b", "c", "d"])
    y = X["a"] + 2 * X["b"] + rs.randn(50) * 0.1
    model = LinearRegression().fit(X, y)

    masker = shap.maskers.Partition(X, clustering="correlation")
    explainer = shap.PermutationExplainer(model.predict, masker)
    shap_values = explainer(X.iloc[:5])

    assert isinstance(shap_values, shap.Explanation)
    assert shap_values.shape == (5, 4)
    # additivity check: sum of shap values + base value ≈ prediction
    preds = model.predict(X.iloc[:5])
    reconstructed = shap_values.values.sum(axis=1) + shap_values.base_values
    np.testing.assert_allclose(reconstructed, preds, atol=0.1)


def test_explanation_hclust_ordering():
    """Explanation.hclust() exercises hclust_ordering to produce a valid sample ordering."""
    rs = np.random.RandomState(0)
    X = pd.DataFrame(rs.randn(30, 4), columns=["a", "b", "c", "d"])
    y = X["a"] + X["b"] + rs.randn(30) * 0.1
    model = GradientBoostingRegressor(n_estimators=10, max_depth=2, random_state=0).fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    order = shap_values.hclust()
    assert isinstance(order, np.ndarray)
    assert len(order) == 30
    assert set(order) == set(range(30))


def test_explanation_hclust_axis1():
    """Explanation.hclust(axis=1) clusters along the feature axis."""
    rs = np.random.RandomState(0)
    X = pd.DataFrame(rs.randn(30, 4), columns=["a", "b", "c", "d"])
    y = X["a"] + X["b"] + rs.randn(30) * 0.1
    model = GradientBoostingRegressor(n_estimators=10, max_depth=2, random_state=0).fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    order = shap_values.hclust(axis=1)
    assert isinstance(order, np.ndarray)
    assert len(order) == 4
    assert set(order) == set(range(4))


def test_partition_tree_via_explainer():
    """partition_tree used to build a clustering for a Partition masker and explainer."""
    rs = np.random.RandomState(0)
    X = pd.DataFrame(rs.randn(50, 4), columns=["a", "b", "c", "d"])
    y = X["a"] + 2 * X["b"] + rs.randn(50) * 0.1
    model = LinearRegression().fit(X, y)

    pt = partition_tree(X)
    assert isinstance(pt, np.ndarray)
    assert pt.shape == (3, 4)

    # use the partition tree as clustering in a masker
    masker = shap.maskers.Partition(X, clustering=pt)
    explainer = shap.PermutationExplainer(model.predict, masker)
    shap_values = explainer(X.iloc[:3])
    assert shap_values.shape == (3, 4)

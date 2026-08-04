import numpy as np
import pandas as pd
import pytest
import scipy.sparse as ssp
import sklearn

import shap


@pytest.mark.parametrize(
    "arr",
    [
        np.arange(100),
        ["zz"] * 100,
        pd.Series(range(100), name="test"),
        pd.DataFrame(np.random.RandomState(0).randn(100, 2), columns=["a", "b"]),
    ],
)
def test_sample_basic(arr):
    """Tests the basic functionality of `sample()` on a variety of array-like objects."""
    new_arr = shap.utils.sample(arr, 30, random_state=42)
    assert len(new_arr) == 30


def test_sample_basic_sparse():
    """Tests the basic functionality of `sample()` on sparse objects."""
    arr = ssp.csr_matrix((100, 3), dtype=np.int8)
    new_arr = shap.utils.sample(arr, 30, random_state=42)
    assert new_arr.shape[0] == 30


def test_sample_no_op():
    """Ensures that `sample()` is a no-op when numsamples is larger
    than the size of X.
    """
    arr = np.arange(50)
    new_arr = shap.utils.sample(arr, 100, random_state=42)

    assert len(arr) == len(new_arr)


def test_sample_sampling_without_replacement():
    """Ensures that `sample()` is performing sampling without replacement.

    See GH dsgibbons#36.
    """
    arr = np.arange(100)
    new_arr = shap.utils.sample(arr, 99, random_state=0)

    assert len(new_arr) == 99
    assert len(np.unique(new_arr)) == 99


def test_sample_can_be_zipped():
    """Ensures that the sampling is done via indexing.

    That is, sampling X and y separately would give the same result as sampling
    concat(X, y), up to a random state. Our `datasets` module relies on
    this behaviour.
    """
    arr1 = pd.Series(np.arange(100))
    arr2 = pd.Series(np.repeat(np.arange(25), 4))
    combined = pd.DataFrame(
        {
            "arr1": arr1,
            "arr2": arr2,
        }
    )

    new_arr1 = shap.utils.sample(arr1, 75, random_state=42)
    new_arr2 = shap.utils.sample(arr2, 75, random_state=42)
    new_combined = shap.utils.sample(combined, 75, random_state=42)

    assert (new_arr1 == new_combined["arr1"]).all()
    assert (new_arr2 == new_combined["arr2"]).all()


def test_opchain_repr():
    """Ensures OpChain repr is working properly"""
    opchain = (
        shap.utils.OpChain("shap.DummyExplanation")
        .foo.foo(0, "big_blue_bear")
        .foo(0, v1=10)
        .foo(k1="alpha", k2="beta")
        .baz
    )
    expected_repr = "shap.DummyExplanation.foo.foo(0, 'big_blue_bear').foo(0, v1=10).foo(k1='alpha', k2='beta').baz"

    assert repr(opchain) == expected_repr


def test_format_value_empty_string():
    """Tests that format_value() handles empty strings without raising IndexError."""
    # Test with empty string
    result = shap.utils._general.format_value("", "%0.03f")
    assert result == ""


def test_format_value_negative_number():
    """Tests that format_value() correctly formats negative numbers with unicode minus sign."""
    result = shap.utils._general.format_value(-1.5, "%0.03f")
    assert result == "\u2212" + "1.5"


def test_format_value_positive_number():
    """Tests that format_value() correctly formats positive numbers."""
    result = shap.utils._general.format_value(1.5, "%0.03f")
    assert result == "1.5"


def test_format_value_trailing_zeros():
    """Tests that format_value() removes trailing zeros."""
    result = shap.utils._general.format_value(1.5000, "%0.03f")
    assert result == "1.5"


def test_format_value_string_input():
    """Tests that format_value() handles string inputs correctly."""
    # Test with non-empty string
    result = shap.utils._general.format_value("test_string", "%0.03f")
    assert result == "test_string"

    # Test with string that starts with minus
    result = shap.utils._general.format_value("-123", "%0.03f")
    assert result == "\u2212" + "123"


def test_rank_interactions_detects_strong_pair():
    rs = np.random.RandomState(0)
    n = 300
    x0 = rs.normal(size=n)
    x1 = rs.normal(size=n)
    x2 = rs.normal(size=n)
    X = np.c_[x0, x1, x2]

    # shap_values[:, 0] depends strongly on x1, creating a clear (0, 1) signal.
    shap_values = np.c_[x0 * x1, x1, x2 * 0.01]

    pairs = shap.utils.rank_interactions(shap_values=shap_values, X=X)
    assert pairs[0][:2] == (0, 1)
    assert pairs[0][2] >= pairs[-1][2]


def test_rank_interactions_returns_symmetric_matrix():
    rs = np.random.RandomState(1)
    X = rs.normal(size=(100, 4))
    shap_values = rs.normal(size=(100, 4))

    pairs, matrix = shap.utils.rank_interactions(shap_values=shap_values, X=X, return_matrix=True)

    assert matrix.shape == (4, 4)
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(matrix), np.zeros(4), atol=1e-12)
    assert len(pairs) == 6


def test_rank_interactions_with_explanation_and_max_pairs():
    rs = np.random.RandomState(2)
    X = rs.normal(size=(80, 5))
    expl = shap.Explanation(
        values=rs.normal(size=(80, 5)),
        data=X,
        feature_names=["a", "b", "c", "d", "e"],
    )

    pairs = shap.utils.rank_interactions(expl, max_pairs=3)
    assert len(pairs) == 3


def test_rank_interactions_rejects_bad_shapes():
    with pytest.raises(ValueError, match="same shape"):
        shap.utils.rank_interactions(shap_values=np.zeros((10, 2)), X=np.zeros((10, 3)))


def _top_pair_from_interaction_matrix(matrix):
    matrix = np.asarray(matrix, dtype=float)
    matrix = matrix.copy()
    np.fill_diagonal(matrix, -np.inf)
    i, j = np.unravel_index(np.argmax(matrix), matrix.shape)
    if i > j:
        i, j = j, i
    return int(i), int(j)


def test_rank_interactions_calibrates_with_tree_interactions_random_forest():
    rs = np.random.RandomState(11)
    X = rs.normal(size=(300, 4))
    y = (X[:, 0] * X[:, 1]) + 0.1 * X[:, 2] + rs.normal(scale=0.03, size=300)

    model = sklearn.ensemble.RandomForestRegressor(n_estimators=80, random_state=0).fit(X, y)
    explainer = shap.TreeExplainer(model)

    shap_values = explainer(X)
    interaction_values = explainer(X, interactions=True).values

    ranked_pairs = shap.utils.rank_interactions(shap_values, max_pairs=3)
    ranked_top3 = {(int(i), int(j)) for i, j, _ in ranked_pairs}

    true_strength = np.mean(np.abs(interaction_values), axis=0)
    true_top = _top_pair_from_interaction_matrix(true_strength)

    assert true_top in ranked_top3


def test_rank_interactions_calibrates_with_tree_interactions_xgboost():
    xgboost = pytest.importorskip("xgboost")
    rs = np.random.RandomState(13)
    X = rs.normal(size=(250, 4))
    y = (1.5 * X[:, 0] * X[:, 1]) + 0.05 * X[:, 2] + rs.normal(scale=0.02, size=250)

    model = xgboost.XGBRegressor(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.08,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=0,
    ).fit(X, y)
    explainer = shap.TreeExplainer(model)

    shap_values = explainer(X)
    interaction_values = explainer(X, interactions=True).values

    ranked_pairs = shap.utils.rank_interactions(shap_values, max_pairs=3)
    ranked_top3 = {(int(i), int(j)) for i, j, _ in ranked_pairs}

    true_strength = np.mean(np.abs(interaction_values), axis=0)
    true_top = _top_pair_from_interaction_matrix(true_strength)

    assert true_top in ranked_top3

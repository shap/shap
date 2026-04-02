import numpy as np
import pandas as pd
import pytest
import scipy.sparse as ssp

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
    result = shap.utils.format_value("", "%0.03f")
    assert result == ""


def test_format_value_negative_number():
    result = shap.utils.format_value(-1.5, "%0.03f")
    assert result == "\u2212" + "1.5"


def test_format_value_positive_number():
    result = shap.utils.format_value(1.5, "%0.03f")
    assert result == "1.5"


def test_format_value_trailing_zeros():
    result = shap.utils.format_value(1.5000, "%0.03f")
    assert result == "1.5"


def test_format_value_string_input():
    result = shap.utils.format_value("test_string", "%0.03f")
    assert result == "test_string"

    result = shap.utils.format_value("-123", "%0.03f")
    assert result == "\u2212" + "123"


def test_approximate_interactions_with_string_feature_name():
    rng = np.random.RandomState(0)
    X = rng.randn(100, 3)
    shap_values = rng.randn(100, 3)
    feature_names = ["age", "income", "score"]

    result = shap.utils.approximate_interactions("income", shap_values, X, feature_names)
    assert result.shape == (3,)
    assert set(result.tolist()) == {0, 1, 2}


def test_approximate_interactions_with_integer_index():
    rng = np.random.RandomState(0)
    X = rng.randn(100, 3)
    shap_values = rng.randn(100, 3)

    result = shap.utils.approximate_interactions(0, shap_values, X)
    assert result.shape == (3,)
    assert set(result.tolist()) == {0, 1, 2}


def test_approximate_interactions_with_rank_index():
    rng = np.random.RandomState(0)
    X = rng.randn(100, 3)
    shap_values = rng.randn(100, 3)
    feature_names = ["a", "b", "c"]

    result = shap.utils.approximate_interactions("rank(0)", shap_values, X, feature_names)
    assert result.shape == (3,)


def test_approximate_interactions_with_dataframe():
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(100, 3), columns=["a", "b", "c"])
    shap_values = rng.randn(100, 3)

    result = shap.utils.approximate_interactions("b", shap_values, X)
    assert result.shape == (3,)


def test_approximate_interactions_with_string_column():
    rng = np.random.RandomState(0)
    X = pd.DataFrame(
        {
            "color": rng.choice(["red", "blue", "green"], 100),
            "size": rng.randn(100),
            "weight": rng.randn(100),
        }
    )
    shap_values = rng.randn(100, 3)

    result = shap.utils.approximate_interactions("size", shap_values, X)
    assert result.shape == (3,)


def test_record_and_assert_import(monkeypatch):
    err = ImportError("fake error")
    shap.utils.record_import_error("_test_fake_pkg", "Could not import _test_fake_pkg", err)
    with pytest.raises(ImportError, match="fake error"):
        shap.utils.assert_import("_test_fake_pkg")

    from shap.utils._general import import_errors

    monkeypatch.delitem(import_errors, "_test_fake_pkg")


def test_assert_import_no_error():
    shap.utils.assert_import("definitely_not_recorded")


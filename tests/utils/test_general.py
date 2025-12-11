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

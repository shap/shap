import sys

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as ssp
from numpy.testing import assert_allclose

import shap
from shap.utils._general import (
    OpChain,
    assert_import,
    approximate_interactions,
    convert_name,
    encode_array_if_needed,
    import_errors,
    ordinal_str,
    record_import_error,
    safe_isinstance,
    shapley_coefficients,
    suppress_stderr,
)


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# OpChain
# ---------------------------------------------------------------------------


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


def test_opchain_apply_attribute():
    """OpChain.apply retrieves an attribute from the target object."""
    chain = OpChain().shape
    result = chain.apply(np.array([1, 2, 3]))
    assert result == (3,)


def test_opchain_apply_method():
    """OpChain.apply calls a method on the target object."""
    chain = OpChain().sum()
    result = chain.apply(np.array([1, 2, 3]))
    assert result == 6


def test_opchain_getitem():
    """OpChain.__getitem__ indexes into the target object."""
    chain = OpChain()[0]
    result = chain.apply(np.array([10, 20, 30]))
    assert result == 10


def test_opchain_chained():
    """OpChain supports chaining multiple operations."""
    chain = OpChain().reshape(2, 3).sum()
    result = chain.apply(np.array([1, 2, 3, 4, 5, 6]))
    assert result == 21


def test_opchain_empty_apply():
    """Empty OpChain returns the object unchanged."""
    chain = OpChain()
    obj = [1, 2, 3]
    assert chain.apply(obj) is obj


# ---------------------------------------------------------------------------
# format_value
# ---------------------------------------------------------------------------


def test_format_value_empty_string():
    """Tests that format_value() handles empty strings without raising IndexError."""
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
    result = shap.utils._general.format_value("test_string", "%0.03f")
    assert result == "test_string"

    result = shap.utils._general.format_value("-123", "%0.03f")
    assert result == "\u2212" + "123"


# ---------------------------------------------------------------------------
# shapley_coefficients
# ---------------------------------------------------------------------------


def test_shapley_coefficients_length():
    """Returns array of correct length."""
    result = shapley_coefficients(5)
    assert isinstance(result, np.ndarray)
    assert len(result) == 5


def test_shapley_coefficients_n1():
    """Single feature case."""
    assert_allclose(shapley_coefficients(1), [1.0])


def test_shapley_coefficients_n2():
    """Two feature case: both coefficients equal 0.5."""
    assert_allclose(shapley_coefficients(2), [0.5, 0.5])


def test_shapley_coefficients_values():
    """Coefficients match the formula 1/(n * C(n-1, i))."""
    import scipy.special

    result = shapley_coefficients(4)
    expected = np.array([1 / (4 * scipy.special.comb(3, i)) for i in range(4)])
    assert_allclose(result, expected)


def test_shapley_coefficients_positive():
    """All coefficients are positive."""
    assert np.all(shapley_coefficients(10) > 0)


# ---------------------------------------------------------------------------
# convert_name
# ---------------------------------------------------------------------------


def test_convert_name_none():
    """None input returns None."""
    assert convert_name(None, None, ["a", "b"]) is None


def test_convert_name_int():
    """Integer input passes through unchanged."""
    assert convert_name(2, None, ["a", "b", "c"]) == 2


def test_convert_name_string_match():
    """String input returns matching index."""
    assert convert_name("b", None, ["a", "b", "c"]) == 1


def test_convert_name_not_found():
    """Unrecognized string raises ValueError."""
    with pytest.raises(ValueError, match="Could not find feature named"):
        convert_name("z", None, ["a", "b", "c"])


def test_convert_name_sum():
    """'sum()' string passes through unchanged."""
    assert convert_name("sum()", None, ["a", "b"]) == "sum()"


def test_convert_name_rank():
    """Rank-based indexing returns the feature with highest mean |SHAP|."""
    shap_values = np.array([[1.0, 5.0, 3.0], [2.0, 4.0, 6.0]])
    result = convert_name("rank(0)", shap_values, ["a", "b", "c"])
    mean_abs = np.abs(shap_values).mean(0)
    expected = np.argsort(-mean_abs)[0]
    assert result == expected


def test_convert_name_rank_no_shap():
    """Rank-based indexing without shap_values raises ValueError."""
    with pytest.raises(ValueError, match="shap_values must be provided"):
        convert_name("rank(0)", None, ["a", "b"])


# ---------------------------------------------------------------------------
# encode_array_if_needed
# ---------------------------------------------------------------------------


def test_encode_numeric_passthrough():
    """Numeric arrays pass through with dtype conversion."""
    arr = np.array([1.0, 2.0, 3.0])
    result = encode_array_if_needed(arr)
    assert_allclose(result, arr)
    assert result.dtype == np.float64


def test_encode_int_to_float():
    """Integer arrays are cast to float."""
    arr = np.array([1, 2, 3])
    result = encode_array_if_needed(arr, dtype=np.float64)
    assert result.dtype == np.float64


def test_encode_string_array():
    """String arrays are encoded to numeric values."""
    arr = np.array(["cat", "dog", "cat", "bird"])
    result = encode_array_if_needed(arr, dtype=float)
    assert result.dtype == float
    assert len(result) == 4
    assert result[0] == result[2]  # "cat" encoded the same
    assert len(np.unique(result)) == 3


# ---------------------------------------------------------------------------
# safe_isinstance
# ---------------------------------------------------------------------------


def test_safe_isinstance_match():
    """Matches numpy array against its class path."""
    assert safe_isinstance(np.array([1]), "numpy.ndarray") is True


def test_safe_isinstance_no_match():
    """Non-matching type returns False."""
    assert safe_isinstance(42, "numpy.ndarray") is False


def test_safe_isinstance_list_of_paths():
    """Accepts a list of class paths, matches if any hit."""
    assert safe_isinstance(np.array([1]), ["pandas.DataFrame", "numpy.ndarray"]) is True


def test_safe_isinstance_unimported_module():
    """Unimported module returns False without error."""
    assert safe_isinstance(42, "nonexistent_module_xyz.SomeClass") is False


def test_safe_isinstance_no_dot_raises():
    """Class path without a dot raises ValueError."""
    with pytest.raises(ValueError, match="full"):
        safe_isinstance(42, "nodot")


def test_safe_isinstance_nonexistent_class():
    """Nonexistent class in an existing module returns False."""
    assert safe_isinstance(42, "numpy.FakeClassName123") is False


def test_safe_isinstance_non_str_non_list():
    """Non-string, non-list class_path_str defaults to [''] which raises."""
    with pytest.raises(ValueError, match="full"):
        safe_isinstance(42, 123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ordinal_str
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (1, "1st"),
        (2, "2nd"),
        (3, "3rd"),
        (4, "4th"),
        (11, "11th"),
        (12, "12th"),
        (13, "13th"),
        (21, "21st"),
        (100, "100th"),
        (111, "111th"),
    ],
)
def test_ordinal_str(n, expected):
    """Ordinal suffix is correct for various edge cases."""
    assert ordinal_str(n) == expected


# ---------------------------------------------------------------------------
# assert_import / record_import_error
# ---------------------------------------------------------------------------


def test_record_and_assert_import():
    """Recorded import error is raised by assert_import."""
    error = ImportError("test error")
    record_import_error("fake_package_xyz", "fake_package not found", error)
    with pytest.raises(ImportError):
        assert_import("fake_package_xyz")
    import_errors.pop("fake_package_xyz", None)


def test_assert_import_no_error():
    """assert_import does nothing for unregistered packages."""
    assert assert_import("not_registered_abc") is None


# ---------------------------------------------------------------------------
# suppress_stderr
# ---------------------------------------------------------------------------


def test_suppress_stderr_redirected():
    """stderr is redirected inside the context manager."""
    with suppress_stderr():
        assert sys.stderr is not sys.__stderr__


def test_suppress_stderr_restored():
    """stderr is restored after the context manager exits."""
    original = sys.stderr
    with suppress_stderr():
        pass
    assert sys.stderr is original


def test_suppress_stderr_restored_on_exception():
    """stderr is restored even if an exception occurs."""
    original = sys.stderr
    with pytest.raises(RuntimeError):
        with suppress_stderr():
            raise RuntimeError("test")
    assert sys.stderr is original


# ---------------------------------------------------------------------------
# approximate_interactions
# ---------------------------------------------------------------------------


def test_approximate_interactions_shape():
    """Output shape matches number of features."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 5))
    shap_values = rng.standard_normal((100, 5))
    result = approximate_interactions(0, shap_values, X)
    assert result.shape == (5,)


def test_approximate_interactions_returns_indices():
    """Output is a permutation of feature indices."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 4))
    shap_values = rng.standard_normal((100, 4))
    result = approximate_interactions(0, shap_values, X)
    assert set(result) == {0, 1, 2, 3}


def test_approximate_interactions_dataframe():
    """Works with DataFrame input and string index."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((100, 3)), columns=["a", "b", "c"])
    shap_values = rng.standard_normal((100, 3))
    result = approximate_interactions("a", shap_values, X)
    assert result.shape == (3,)


def test_approximate_interactions_string_index():
    """Works with string index and explicit feature_names."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 3))
    shap_values = rng.standard_normal((50, 3))
    result = approximate_interactions("feat_b", shap_values, X, feature_names=["feat_a", "feat_b", "feat_c"])
    assert result.shape == (3,)


def test_approximate_interactions_correlated():
    """Feature with strongest correlation is ranked first."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 3))
    shap_values = np.zeros((200, 3))
    shap_values[:, 0] = X[:, 1] * 2 + rng.standard_normal(200) * 0.1
    shap_values[:, 1] = rng.standard_normal(200)
    shap_values[:, 2] = rng.standard_normal(200)
    result = approximate_interactions(0, shap_values, X)
    assert result[0] == 1


# ---------------------------------------------------------------------------
# potential_interactions
# ---------------------------------------------------------------------------


def test_potential_interactions_shape():
    """Output shape matches number of features."""
    from shap.utils._general import potential_interactions

    rng = np.random.default_rng(42)

    class MockExplanation:
        def __init__(self, values, data):
            self.values = values
            self.data = data

    n_samples, n_features = 100, 4
    values = rng.standard_normal((n_samples, n_features))
    data = rng.standard_normal((n_samples, n_features))

    column = MockExplanation(values[:, 0], data[:, 0])
    matrix = MockExplanation(values, data)

    result = potential_interactions(column, matrix)
    assert result.shape == (n_features,)
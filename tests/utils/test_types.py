from typing import TypeVar

import numpy as np
import pandas as pd
import scipy.sparse

from shap.utils._types import _ArrayT


def test_array_t_is_typevar():
    assert isinstance(_ArrayT, TypeVar)


def test_array_t_constraints():
    constraints = _ArrayT.__constraints__
    assert np.ndarray in constraints
    assert pd.DataFrame in constraints
    assert pd.Series in constraints
    assert scipy.sparse.spmatrix in constraints
    assert list in constraints


def test_array_like_includes_numpy():
    arr = np.array([1, 2, 3])
    assert isinstance(arr, np.ndarray)


def test_array_like_includes_dataframe():
    df = pd.DataFrame({"a": [1, 2]})
    assert isinstance(df, pd.DataFrame)


def test_array_like_includes_series():
    s = pd.Series([1, 2, 3])
    assert isinstance(s, pd.Series)


def test_array_like_includes_list():
    assert isinstance([1, 2, 3], list)


def test_array_like_includes_sparse():
    m = scipy.sparse.csr_matrix([[1, 0], [0, 1]])
    assert isinstance(m, scipy.sparse.spmatrix)

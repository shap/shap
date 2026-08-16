from typing import get_args

import numpy as np
import pandas as pd
import scipy.sparse as ssp

from shap.utils import _types


def test_arraylike_union_contains_supported_types():
    expected = (np.ndarray, pd.DataFrame, pd.Series, list, ssp.spmatrix)
    alias_value = _types._ArrayLike.__value__
    assert get_args(alias_value) == expected


def test_arrayt_typevar_constraints_match_arraylike():
    expected = (np.ndarray, pd.DataFrame, pd.Series, ssp.spmatrix, list)
    assert _types._ArrayT.__constraints__ == expected


def test_arraylike_runtime_examples_match_declared_types():
    array_like_types = get_args(_types._ArrayLike.__value__)
    examples = [
        np.array([1, 2, 3]),
        pd.DataFrame({"a": [1, 2]}),
        pd.Series([1, 2]),
        [1, 2, 3],
        ssp.csr_matrix(np.eye(2)),
    ]

    assert all(isinstance(example, array_like_types) for example in examples)

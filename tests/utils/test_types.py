"""Tests for shap/utils/_types.py.

The module only exports type-checking constructs (a TypeAliasType and a
TypeVar). There is no callable logic, so these tests verify the structural
contract: the right names are exported, _ArrayT has the expected TypeVar
constraints, and _ArrayLike covers the expected member types.
"""

from __future__ import annotations

import typing
from typing import TypeVar

import numpy as np
import pandas as pd
import pytest
import scipy.sparse

from shap.utils._types import _ArrayLike, _ArrayT

# The canonical set of array-like types used throughout shap.
_EXPECTED_TYPES = {np.ndarray, pd.DataFrame, pd.Series, list, scipy.sparse.spmatrix}


class TestArrayT:
    """Tests for the _ArrayT TypeVar."""

    def test_is_typevar(self):
        assert isinstance(_ArrayT, TypeVar)

    def test_name(self):
        assert _ArrayT.__name__ == "_ArrayT"

    def test_constraints_are_not_empty(self):
        assert len(_ArrayT.__constraints__) > 0

    def test_constraints_match_expected_types(self):
        assert set(_ArrayT.__constraints__) == _EXPECTED_TYPES

    @pytest.mark.parametrize(
        "t",
        [np.ndarray, pd.DataFrame, pd.Series, list, scipy.sparse.spmatrix],
    )
    def test_each_expected_type_is_constrained(self, t):
        assert t in _ArrayT.__constraints__


class TestArrayLike:
    """Tests for the _ArrayLike TypeAliasType."""

    def test_is_type_alias(self):
        assert isinstance(_ArrayLike, typing.TypeAliasType)

    def test_name(self):
        assert _ArrayLike.__name__ == "_ArrayLike"

    def test_value_is_union(self):
        # __value__ should be the raw union, not None
        assert _ArrayLike.__value__ is not None

    def test_member_types_match_expected(self):
        member_types = set(typing.get_args(_ArrayLike.__value__))
        assert member_types == _EXPECTED_TYPES

    @pytest.mark.parametrize(
        "t",
        [np.ndarray, pd.DataFrame, pd.Series, list, scipy.sparse.spmatrix],
    )
    def test_each_expected_type_is_member(self, t):
        member_types = typing.get_args(_ArrayLike.__value__)
        assert t in member_types


class TestConsistency:
    """_ArrayLike and _ArrayT should cover the same set of types."""

    def test_arraylike_and_arrayt_cover_same_types(self):
        alias_types = set(typing.get_args(_ArrayLike.__value__))
        typevar_types = set(_ArrayT.__constraints__)
        assert alias_types == typevar_types

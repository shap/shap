import numpy as np
import pytest

from shap.utils._slicer import Alias, AtomicSlicer, Slicer, _handle_newaxis_ellipses, unify_slice


class TestSlicerNormalizations:
    """Tests for core index and slice normalization logic."""

    def test_handle_newaxis_ellipses(self):
        # Test filling max_dim with slice(None)
        res = _handle_newaxis_ellipses((...,), max_dim=3)
        assert res == (slice(None), slice(None), slice(None))

        # Test concrete indices with ellipsis
        res = _handle_newaxis_ellipses((1, ..., 2), max_dim=4)
        assert res == (1, slice(None), slice(None), 2)

        # Test multiple ellipses raise error
        with pytest.raises(IndexError, match="an index can only have a single ellipsis"):
            _handle_newaxis_ellipses((..., 1, ...), max_dim=3)

        # Test too many indices raise error
        with pytest.raises(IndexError, match="too many indices for array"):
            _handle_newaxis_ellipses((1, 2, 3), max_dim=2)

    def test_unify_slice(self):
        # Unify single int
        assert unify_slice(1, max_dim=2) == (1, slice(None))

        # Unify tuple
        assert unify_slice((1, 2), max_dim=2) == (1, 2)


class TestAtomicSlicer:
    """Tests for the base AtomicSlicer logic across data types."""

    def test_list_tuple_handler(self):
        data = [[1, 2, 3], [4, 5, 6]]
        s = AtomicSlicer(data)

        assert s[0] == [1, 2, 3]
        assert s[:, 1] == [2, 5]
        assert s[1, 1:] == [5, 6]

    def test_dict_handler(self):
        data = {"a": [1, 2, 3], "b": [4, 5, 6]}
        s = AtomicSlicer(data)

        # Slicing by key (DictHandler treats keys as dimension 0)
        assert s["a"] == [1, 2, 3]

        # Slicing the elements inside the dictionary values (dimension 1)
        # slice(None) for keys, 0 for the first element in each array
        assert s[:, 0] == {"a": 1, "b": 4}
        assert s[:, 1:] == {"a": [2, 3], "b": [5, 6]}

    def test_numpy_array_handler(self):
        data = np.array([[1, 2], [3, 4], [5, 6]])
        s = AtomicSlicer(data)

        np.testing.assert_array_equal(s[1:], np.array([[3, 4], [5, 6]]))
        np.testing.assert_array_equal(s[:, 1], np.array([2, 4, 6]))


class TestSlicerAPI:
    """Tests for the main Slicer public API."""

    def test_slicer_initialization(self):
        # Test anonymous (positional) initialization
        s = Slicer([1, 2, 3])
        assert s.o == [1, 2, 3]

        # Test kwargs initialization
        s_kwargs = Slicer(x=[1, 2, 3], y=["a", "b", "c"])
        assert s_kwargs.x == [1, 2, 3]
        assert s_kwargs.y == ["a", "b", "c"]

    def test_slicer_alias_lookup(self):
        data = [10, 20, 30]
        s = Slicer(data)

        # Add an alias tracking dimension 0
        s.names = Alias(["first", "second", "third"], dim=0)

        # Test dictionary-like lookup using the alias
        res = s["second"]
        assert res.o == 20

        # Ensure we can still slice normally
        res_normal = s[:2]
        assert res_normal.o == [10, 20]

    def test_slicer_dynamic_attribute_assignment(self):
        s = Slicer([1, 2, 3])
        s.y = [4, 5, 6]

        assert s.y == [4, 5, 6]
        res = s[1:]
        assert res.o == [2, 3]
        assert res.y == [5, 6]

    def test_slicer_deletion(self):
        s = Slicer(x=[1, 2], y=[3, 4])
        del s.x

        with pytest.raises(AttributeError):
            _ = s.x

        assert s.y == [3, 4]

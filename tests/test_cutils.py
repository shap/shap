"""Tests for the C++ utility functions in shap._cutils."""

import numpy as np

from shap._cutils import mask_delta_score, reverse_window

# Tests for reverse_window


# reverses a window in the middle of the array, leaving surrounding elements untouched
def test_reverse_window_basic():
    order = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    reverse_window(order, 1, 3)
    np.testing.assert_array_equal(order, [0, 3, 2, 1, 4])


# reverses the entire array
def test_reverse_window_full_array():
    order = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    reverse_window(order, 0, 5)
    np.testing.assert_array_equal(order, [4, 3, 2, 1, 0])


# length=1 is a no-op, array must be unchanged
def test_reverse_window_single_element():
    order = np.array([0, 1, 2, 3], dtype=np.int64)
    reverse_window(order, 2, 1)
    np.testing.assert_array_equal(order, [0, 1, 2, 3])


# length=0 is a no-op, array must be unchanged
def test_reverse_window_empty():
    order = np.array([0, 1, 2, 3], dtype=np.int64)
    reverse_window(order, 1, 0)
    np.testing.assert_array_equal(order, [0, 1, 2, 3])


# window positioned at the tail of the array
def test_reverse_window_at_end():
    order = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    reverse_window(order, 2, 3)
    np.testing.assert_array_equal(order, [0, 1, 4, 3, 2])


# Tests for mask_delta_score


# identical masks have zero Hamming distance
def test_mask_delta_score_identical():
    m = np.array([True, False, True, True], dtype=bool)
    assert mask_delta_score(m, m) == 0


# completely opposite masks have distance equal to array length
def test_mask_delta_score_all_different():
    m1 = np.array([True, True, False, False], dtype=bool)
    m2 = np.array([False, False, True, True], dtype=bool)
    assert mask_delta_score(m1, m2) == 4


# partially overlapping masks return the count of differing positions
def test_mask_delta_score_partial():
    m1 = np.array([True, False, True, False], dtype=bool)
    m2 = np.array([True, True, False, False], dtype=bool)
    assert mask_delta_score(m1, m2) == 2


# single-element arrays — one matching, one different
def test_mask_delta_score_single_element():
    assert mask_delta_score(np.array([True], dtype=bool), np.array([True], dtype=bool)) == 0
    assert mask_delta_score(np.array([True], dtype=bool), np.array([False], dtype=bool)) == 1

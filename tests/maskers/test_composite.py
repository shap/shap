"""This file contains tests for the Composite masker."""

import numpy as np
import pytest

import shap


def test_composite_masker_init():
    """Test Composite masker initialization."""
    masker1 = shap.maskers.Fixed()
    masker2 = shap.maskers.Fixed()

    composite = shap.maskers.Composite(masker1, masker2)

    assert len(composite.maskers) == 2
    assert composite.maskers[0] is masker1
    assert composite.maskers[1] is masker2
    assert len(composite.arg_counts) == 2
    assert composite.total_args == 2
    assert composite.text_data is False
    assert composite.image_data is False


def test_composite_masker_with_fixed_maskers():
    """Test Composite masker combining Fixed maskers."""
    masker1 = shap.maskers.Fixed()
    masker2 = shap.maskers.Fixed()

    composite = shap.maskers.Composite(masker1, masker2)

    # Test shape
    shape = composite.shape("arg1", "arg2")
    assert shape == (None, 0)

    # Note: Calling composite with Fixed maskers where num_rows is None causes a bug
    # TODO: check if this code is dead! Line 118 in _composite.py fails when num_rows is None


def test_composite_masker_shape_method():
    """Test Composite masker shape method."""
    masker1 = shap.maskers.Fixed()
    masker2 = shap.maskers.Fixed()

    composite = shap.maskers.Composite(masker1, masker2)

    # Two Fixed maskers, each with shape (None, 0)
    shape = composite.shape("arg1", "arg2")

    assert shape[0] is None
    assert shape[1] == 0


def test_composite_masker_mask_shapes():
    """Test Composite masker mask_shapes method."""
    masker1 = shap.maskers.Fixed()
    masker2 = shap.maskers.Fixed()

    composite = shap.maskers.Composite(masker1, masker2)

    result = composite.mask_shapes("arg1", "arg2")

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == (0,)
    assert result[1] == (0,)


def test_composite_masker_data_transform():
    """Test Composite masker data_transform method."""
    masker1 = shap.maskers.Fixed()
    masker2 = shap.maskers.Fixed()

    composite = shap.maskers.Composite(masker1, masker2)

    # Fixed maskers don't have data_transform, so args should pass through
    result = composite.data_transform("arg1", "arg2")

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == "arg1"
    assert result[1] == "arg2"


def test_composite_masker_arg_count_mismatch():
    """Test Composite masker with wrong number of arguments."""
    masker1 = shap.maskers.Fixed()
    masker2 = shap.maskers.Fixed()

    composite = shap.maskers.Composite(masker1, masker2)

    # Should expect 2 args, but only provide 1
    with pytest.raises(AssertionError, match="number of passed args is incorrect"):
        composite.shape("arg1")

    # Should expect 2 args, but provide 3
    with pytest.raises(AssertionError, match="number of passed args is incorrect"):
        composite.shape("arg1", "arg2", "arg3")


def test_composite_masker_call_arg_count_mismatch():
    """Test Composite masker __call__ with wrong number of arguments."""
    masker1 = shap.maskers.Fixed()
    masker2 = shap.maskers.Fixed()

    composite = shap.maskers.Composite(masker1, masker2)
    mask = np.array([], dtype=bool)

    # Should expect 2 args, but only provide 1
    with pytest.raises(AssertionError, match="number of passed args is incorrect"):
        composite(mask, "arg1")


def test_composite_masker_text_data_flag():
    """Test that text_data flag propagates from submaskers."""
    # Create a simple masker with text_data flag
    masker1 = shap.maskers.Fixed()
    masker1.text_data = True

    masker2 = shap.maskers.Fixed()

    composite = shap.maskers.Composite(masker1, masker2)

    assert composite.text_data is True


def test_composite_masker_image_data_flag():
    """Test that image_data flag propagates from submaskers."""
    # Create a simple masker with image_data flag
    masker1 = shap.maskers.Fixed()
    masker1.image_data = True

    masker2 = shap.maskers.Fixed()

    composite = shap.maskers.Composite(masker1, masker2)

    assert composite.image_data is True


def test_composite_masker_clustering():
    """Test that clustering attribute is set when all maskers have clustering."""
    masker1 = shap.maskers.Fixed()
    masker2 = shap.maskers.Fixed()

    composite = shap.maskers.Composite(masker1, masker2)

    # Both Fixed maskers have clustering attribute
    assert hasattr(composite, "clustering")


def test_composite_masker_no_clustering():
    """Test that clustering is not set when a masker lacks clustering attribute."""

    # Create a simple masker without clustering
    class SimpleMasker(shap.maskers.Masker):
        def __init__(self):
            self.shape = (None, 0)

        def __call__(self, mask, x):
            return ([x],)

    masker1 = shap.maskers.Fixed()
    masker2 = SimpleMasker()

    composite = shap.maskers.Composite(masker1, masker2)

    # masker2 doesn't have clustering, so composite shouldn't have callable clustering
    # The clustering attribute might exist but won't be a callable method
    assert not callable(getattr(composite, "clustering", None))


def test_composite_masker_standardize_mask():
    """Test that _standardize_mask works with Composite masker."""

    # Create maskers with defined row counts to avoid None comparison bug
    class SimpleMasker(shap.maskers.Masker):
        def __init__(self, rows, cols):
            self.shape = (rows, cols)
            self.clustering = np.zeros((0, 4))

        def __call__(self, mask, x):
            return (np.array([x] * self.shape[0]),)

    masker1 = SimpleMasker(5, 3)
    masker2 = SimpleMasker(5, 2)

    composite = shap.maskers.Composite(masker1, masker2)

    # Test with True mask - should create all ones
    result = composite(True, "arg1", "arg2")
    assert isinstance(result, tuple)
    assert len(result) == 2

    # Test with False mask - should create all zeros
    result = composite(False, "arg1", "arg2")
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_composite_masker_single_masker():
    """Test Composite masker with a single submasker."""
    masker = shap.maskers.Fixed()

    composite = shap.maskers.Composite(masker)

    assert len(composite.maskers) == 1
    assert composite.total_args == 1

    shape = composite.shape("arg1")
    assert shape == (None, 0)


def test_composite_masker_three_maskers():
    """Test Composite masker with three submaskers."""
    masker1 = shap.maskers.Fixed()
    masker2 = shap.maskers.Fixed()
    masker3 = shap.maskers.Fixed()

    composite = shap.maskers.Composite(masker1, masker2, masker3)

    assert len(composite.maskers) == 3
    assert composite.total_args == 3

    # Note: Calling composite with Fixed maskers where num_rows is None causes a bug
    # TODO: check if this code is dead! Line 118 in _composite.py fails when num_rows is None


def test_composite_masker_with_realistic_shapes():
    """Test Composite masker with maskers that have realistic shapes."""

    # Create simple maskers with actual row/column counts
    class SimpleMasker(shap.maskers.Masker):
        def __init__(self, rows, cols):
            self.shape = (rows, cols)
            self.clustering = np.zeros((0, 4))

        def __call__(self, mask, x):
            # Return masked data with proper shape
            return (np.array([[x] * self.shape[1]] * self.shape[0]),)

    masker1 = SimpleMasker(10, 5)
    masker2 = SimpleMasker(10, 3)
    masker3 = SimpleMasker(10, 2)

    composite = shap.maskers.Composite(masker1, masker2, masker3)

    assert len(composite.maskers) == 3
    assert composite.total_args == 3

    # Test shape - should sum the columns
    shape = composite.shape("arg1", "arg2", "arg3")
    assert shape == (10, 10)  # 10 rows, 5+3+2 cols

    # Test __call__
    mask = np.zeros(10, dtype=bool)
    result = composite(mask, "arg1", "arg2", "arg3")

    assert isinstance(result, tuple)
    assert len(result) == 3


def test_composite_masker_incompatible_row_counts():
    """Test Composite masker with incompatible row counts."""

    class SimpleMasker(shap.maskers.Masker):
        def __init__(self, rows, cols):
            self.shape = (rows, cols)
            self.clustering = np.zeros((0, 4))

        def __call__(self, mask, x):
            return (np.array([[x] * self.shape[1]] * self.shape[0]),)

    # Incompatible row counts (neither is 1)
    masker1 = SimpleMasker(5, 3)
    masker2 = SimpleMasker(10, 2)

    composite = shap.maskers.Composite(masker1, masker2)

    mask = np.zeros(5, dtype=bool)

    # Should raise InvalidMaskerError
    from shap.utils._exceptions import InvalidMaskerError

    with pytest.raises(InvalidMaskerError, match="compatible number of background rows"):
        composite(mask, "arg1", "arg2")


def test_composite_masker_callable_shape():
    """Test Composite masker with callable shape."""

    class CallableShapeMasker(shap.maskers.Masker):
        def __init__(self, rows, cols):
            self._rows = rows
            self._cols = cols
            self.shape = lambda x: (self._rows, self._cols)
            self.clustering = np.zeros((0, 4))

        def __call__(self, mask, x):
            return (np.array([[x] * self._cols] * self._rows),)

    masker1 = CallableShapeMasker(10, 5)
    masker2 = CallableShapeMasker(10, 3)

    composite = shap.maskers.Composite(masker1, masker2)

    shape = composite.shape("arg1", "arg2")
    assert shape == (10, 8)

    mask = np.zeros(8, dtype=bool)
    result = composite(mask, "arg1", "arg2")

    assert isinstance(result, tuple)
    assert len(result) == 2


def test_composite_masker_with_data_transform():
    """Test Composite masker with maskers that have data_transform."""

    class TransformMasker(shap.maskers.Masker):
        def __init__(self, rows, cols):
            self.shape = (rows, cols)
            self.clustering = np.zeros((0, 4))

        def __call__(self, mask, x):
            return (np.array([[x] * self.shape[1]] * self.shape[0]),)

        def data_transform(self, x):
            # Transform the data by uppercasing if string
            return (x.upper() if isinstance(x, str) else x,)

    masker1 = TransformMasker(10, 5)
    masker2 = TransformMasker(10, 3)

    composite = shap.maskers.Composite(masker1, masker2)

    # Test data_transform
    result = composite.data_transform("arg1", "arg2")

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == "ARG1"
    assert result[1] == "ARG2"


def test_composite_masker_joint_clustering():
    """Test joint_clustering function for Composite masker."""

    class ClusteringMasker(shap.maskers.Masker):
        def __init__(self, rows, cols, clustering_data):
            self.shape = (rows, cols)
            self.clustering = clustering_data

        def __call__(self, mask, x):
            return (np.array([[x] * self.shape[1]] * self.shape[0]),)

    # First masker with non-trivial clustering
    clustering1 = np.array([[0, 0, 1, 1], [1, 1, 2, 2]])
    masker1 = ClusteringMasker(10, 5, clustering1)

    # Second masker with trivial (empty) clustering
    masker2 = ClusteringMasker(10, 3, np.zeros((0, 4)))

    composite = shap.maskers.Composite(masker1, masker2)

    # Should have clustering method
    assert hasattr(composite, "clustering")
    assert callable(composite.clustering)

    # Call clustering
    result = composite.clustering("arg1", "arg2")
    np.testing.assert_array_equal(result, clustering1)

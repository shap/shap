"""Tests for maskers through public explainer APIs.

These tests exercise masker functionality by using public explainers like
shap.Explainer, shap.KernelExplainer, etc., which internally use maskers.
"""

import numpy as np

import shap


def test_masker_with_kernel_explainer():
    """Test masker functionality through KernelExplainer (public API)."""

    # Create simple model
    def model(x):
        return np.sum(x, axis=1)

    # Create background data for masker
    background = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])

    # KernelExplainer uses Independent masker internally
    explainer = shap.KernelExplainer(model, background)

    # Test data
    test_data = np.array([[1, 2, 3]])

    # This exercises masker's __call__ with various masks
    shap_values = explainer.shap_values(test_data, nsamples=10)

    assert shap_values is not None
    assert shap_values.shape == (1, 3)


def test_masker_with_kernel_explainer_and_custom_masker():
    """Test custom masker through KernelExplainer."""

    def model(x):
        return np.sum(x, axis=1)

    background = np.array([[0, 0, 0], [1, 1, 1]])

    # Use Independent masker explicitly (public API)
    masker = shap.maskers.Independent(background)

    explainer = shap.KernelExplainer(model, masker)

    test_data = np.array([[1, 2, 3]])
    shap_values = explainer.shap_values(test_data, nsamples=10)

    assert shap_values is not None


def test_independent_masker_basic():
    """Test Independent masker basic functionality."""
    background = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])

    masker = shap.maskers.Independent(background)

    # Test shape
    assert masker.shape == (3, 3)

    # Test with True mask (should select all features)
    result = masker(True, background[0])
    assert isinstance(result, tuple)


def test_independent_masker_with_false_mask():
    """Test Independent masker with False mask."""
    background = np.array([[0, 0], [1, 1], [2, 2]])

    masker = shap.maskers.Independent(background)

    # False mask should select no features
    result = masker(False, background[0])
    assert isinstance(result, tuple)


def test_independent_masker_with_partial_mask():
    """Test Independent masker with partial mask array."""
    background = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])

    masker = shap.maskers.Independent(background)

    # Partial mask - select first two features
    mask = np.array([True, True, False])
    result = masker(mask, np.array([5, 6, 7]))

    assert isinstance(result, tuple)
    assert len(result) == 1
    # Should return background samples with first two features from input
    assert result[0].shape[0] == 3  # Same as background rows


def test_partition_masker_basic():
    """Test Partition masker basic functionality."""
    background = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])

    masker = shap.maskers.Partition(background)

    # Test with True mask
    result = masker(True, background[0])
    assert isinstance(result, tuple)


def test_partition_masker_with_false_mask():
    """Test Partition masker with False mask."""
    background = np.array([[0, 0], [1, 1]])

    masker = shap.maskers.Partition(background)

    result = masker(False, background[0])
    assert isinstance(result, tuple)


def test_masker_with_explainer_auto_detection():
    """Test that Explainer auto-detects and uses appropriate masker."""

    def model(x):
        return np.sum(x, axis=1)

    background = np.array([[0, 0, 0], [1, 1, 1]])

    # Explainer should auto-select masker based on model and data
    explainer = shap.Explainer(model, background)

    test_data = np.array([[1, 2, 3]])

    # This internally uses maskers
    shap_values = explainer(test_data)

    assert shap_values is not None
    assert shap_values.values.shape == (1, 3)


def test_independent_masker_clustering():
    """Test that Independent masker has clustering attribute."""
    background = np.array([[0, 0, 0], [1, 1, 1]])

    masker = shap.maskers.Independent(background)

    # Should have clustering attribute
    assert hasattr(masker, "clustering")


def test_partition_masker_clustering():
    """Test that Partition masker has clustering attribute."""
    background = np.array([[0, 0, 0], [1, 1, 1]])

    masker = shap.maskers.Partition(background)

    # Should have clustering attribute
    assert hasattr(masker, "clustering")


def test_independent_masker_shape_property():
    """Test Independent masker shape property with different data sizes."""
    # 5 samples, 4 features
    background = np.random.rand(5, 4)

    masker = shap.maskers.Independent(background)

    assert masker.shape == (5, 4)


def test_partition_masker_with_clustering():
    """Test Partition masker with explicit clustering."""
    background = np.array([[0, 0, 0, 0], [1, 1, 1, 1]])

    # Partition masker can take clustering parameter
    masker = shap.maskers.Partition(background, clustering="correlation")

    assert hasattr(masker, "clustering")


def test_composite_masker_with_independent_maskers():
    """Test Composite masker combining Independent maskers."""
    background1 = np.array([[0, 0], [1, 1]])
    background2 = np.array([[2, 2, 2], [3, 3, 3]])

    masker1 = shap.maskers.Independent(background1)
    masker2 = shap.maskers.Independent(background2)

    composite = shap.maskers.Composite(masker1, masker2)

    # Shape should be sum of both
    shape = composite.shape(background1[0], background2[0])
    assert shape == (2, 5)  # 2 rows (min), 2+3 = 5 cols


def test_independent_masker_with_varying_masks():
    """Test Independent masker with different mask patterns."""
    background = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]])

    masker = shap.maskers.Independent(background)

    # All True
    result = masker(np.ones(4, dtype=bool), np.array([5, 6, 7, 8]))
    assert result[0].shape == (3, 4)

    # All False
    result = masker(np.zeros(4, dtype=bool), np.array([5, 6, 7, 8]))
    assert result[0].shape == (3, 4)

    # Alternating
    result = masker(np.array([True, False, True, False]), np.array([5, 6, 7, 8]))
    assert result[0].shape == (3, 4)

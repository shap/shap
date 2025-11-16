"""This file contains tests for the OutputComposite masker."""

import tempfile

import numpy as np

import shap


def test_output_composite_init():
    """Test OutputComposite masker initialization."""
    masker = shap.maskers.Fixed()

    def simple_model(x):
        return np.sum(x)

    output_composite = shap.maskers.OutputComposite(masker, simple_model)

    assert output_composite.masker is masker
    assert output_composite.model is simple_model


def test_output_composite_attribute_propagation():
    """Test that attributes from the underlying masker are propagated."""
    masker = shap.maskers.Fixed()

    def simple_model(x):
        return x

    output_composite = shap.maskers.OutputComposite(masker, simple_model)

    # Check that shape is propagated
    assert hasattr(output_composite, "shape")
    assert output_composite.shape == masker.shape


def test_output_composite_call_with_tuple_output():
    """Test OutputComposite __call__ when model returns a tuple."""
    masker = shap.maskers.Fixed()

    def model_with_tuple_output(x):
        return (np.sum(x), np.mean(x))

    output_composite = shap.maskers.OutputComposite(masker, model_with_tuple_output)

    test_input = np.array([1, 2, 3, 4, 5])
    mask = np.array([], dtype=bool)

    result = output_composite(mask, test_input)

    # Result should be masked input + model output
    assert isinstance(result, tuple)
    # Fixed masker returns 1 element, model returns 2 elements
    assert len(result) == 3


def test_output_composite_call_with_scalar_output():
    """Test OutputComposite __call__ when model returns a scalar."""
    masker = shap.maskers.Fixed()

    def model_with_scalar_output(x):
        return np.sum(x)

    output_composite = shap.maskers.OutputComposite(masker, model_with_scalar_output)

    test_input = np.array([1, 2, 3, 4, 5])
    mask = np.array([], dtype=bool)

    result = output_composite(mask, test_input)

    # Result should be masked input + model output (wrapped in tuple)
    assert isinstance(result, tuple)
    # Fixed masker returns 1 element, model returns 1 element (wrapped)
    assert len(result) == 2


def test_output_composite_call_with_array_output():
    """Test OutputComposite __call__ when model returns an array."""
    masker = shap.maskers.Fixed()

    def model_with_array_output(x):
        return x * 2

    output_composite = shap.maskers.OutputComposite(masker, model_with_array_output)

    test_input = np.array([1, 2, 3])
    mask = np.array([], dtype=bool)

    result = output_composite(mask, test_input)

    assert isinstance(result, tuple)
    assert len(result) == 2
    # Check model output is correct
    np.testing.assert_array_equal(result[1], test_input * 2)


def test_output_composite_serialization():
    """Test OutputComposite serialization and deserialization."""
    masker = shap.maskers.Fixed()

    def simple_model(x):
        return np.sum(x)

    original_composite = shap.maskers.OutputComposite(masker, simple_model)

    with tempfile.TemporaryFile() as temp_file:
        # Serialize
        original_composite.save(temp_file)
        temp_file.seek(0)

        # Deserialize
        loaded_composite = shap.maskers.OutputComposite.load(temp_file)

        # Verify the loaded masker works
        test_input = np.array([1, 2, 3])
        mask = np.array([], dtype=bool)

        original_result = original_composite(mask, test_input)
        loaded_result = loaded_composite(mask, test_input)

        # Check results are the same
        assert len(original_result) == len(loaded_result)


def test_output_composite_with_multiple_args():
    """Test OutputComposite with a model that takes multiple arguments."""

    # Create a custom masker that takes 2 arguments
    class TwoArgMasker(shap.maskers.Masker):
        def __init__(self):
            self.shape = (None, 0)

        def __call__(self, mask, x, y):
            return ([x], [y])

    masker = TwoArgMasker()

    def model_two_args(x, y):
        return np.sum(x) + np.sum(y)

    output_composite = shap.maskers.OutputComposite(masker, model_two_args)

    test_input1 = np.array([1, 2, 3])
    test_input2 = np.array([4, 5, 6])
    mask = np.array([], dtype=bool)

    result = output_composite(mask, test_input1, test_input2)

    assert isinstance(result, tuple)
    # Masker returns 2 elements, model returns 1 (wrapped)
    assert len(result) == 3
    assert result[2] == np.sum(test_input1) + np.sum(test_input2)


def test_output_composite_attribute_none_handling():
    """Test that OutputComposite handles None attributes correctly."""

    # Create a minimal masker without all optional attributes
    class MinimalMasker(shap.maskers.Masker):
        def __init__(self):
            self.shape = (None, 0)
            # Don't set optional attributes

        def __call__(self, mask, x):
            return ([x],)

    masker = MinimalMasker()

    def simple_model(x):
        return x

    output_composite = shap.maskers.OutputComposite(masker, simple_model)

    # Should have shape but not necessarily other attributes
    assert hasattr(output_composite, "shape")
    # These attributes shouldn't be set if not present in masker
    assert not hasattr(output_composite, "invariants") or output_composite.invariants is None


def test_output_composite_text_data_flag():
    """Test that text_data flag propagates from underlying masker."""

    class TextMasker(shap.maskers.Masker):
        def __init__(self):
            self.shape = (None, 0)
            self.text_data = True

        def __call__(self, mask, x):
            return ([x],)

    masker = TextMasker()

    def simple_model(x):
        return x

    output_composite = shap.maskers.OutputComposite(masker, simple_model)

    assert hasattr(output_composite, "text_data")
    assert output_composite.text_data is True


def test_output_composite_image_data_flag():
    """Test that image_data flag propagates from underlying masker."""

    class ImageMasker(shap.maskers.Masker):
        def __init__(self):
            self.shape = (None, 0)
            self.image_data = True

        def __call__(self, mask, x):
            return ([x],)

    masker = ImageMasker()

    def simple_model(x):
        return x

    output_composite = shap.maskers.OutputComposite(masker, simple_model)

    assert hasattr(output_composite, "image_data")
    assert output_composite.image_data is True

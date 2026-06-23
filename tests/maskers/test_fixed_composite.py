"""This file contains tests for the FixedComposite masker."""

import io
import pickle
import tempfile

import numpy as np
import pytest

import shap


@pytest.mark.skip(
    reason="fails on travis and I don't know why yet...Ryan might need to take a look since this API will change soon anyway"
)
def test_fixed_composite_masker_call():
    """Test to make sure the FixedComposite masker works when masking everything."""
    AutoTokenizer = pytest.importorskip("transformers").AutoTokenizer

    args = ("This is a test statement for fixed composite masker",)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    masker = shap.maskers.Text(tokenizer)
    mask = np.zeros(masker.shape(*args)[1], dtype=bool)

    fixed_composite_masker = shap.maskers.FixedComposite(masker)

    expected_fixed_composite_masked_output = (
        np.array([""]),
        np.array(["This is a test statement for fixed composite masker"]),
    )
    fixed_composite_masked_output = fixed_composite_masker(mask, *args)

    assert fixed_composite_masked_output == expected_fixed_composite_masked_output


def test_serialization_fixedcomposite_masker():
    """Make sure fixedcomposite serialization works."""
    AutoTokenizer = pytest.importorskip("transformers").AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased", use_fast=False)
    underlying_masker = shap.maskers.Text(tokenizer)
    original_masker = shap.maskers.FixedComposite(underlying_masker)

    with tempfile.TemporaryFile() as temp_serialization_file:
        original_masker.save(temp_serialization_file)

        temp_serialization_file.seek(0)

        # deserialize masker
        new_masker = shap.maskers.FixedComposite.load(temp_serialization_file)

    test_text = "I ate a Cannoli"
    test_input_mask = np.array([True, False, True, True, False, True, True, True])

    original_masked_output = original_masker(test_input_mask, test_text)
    new_masked_output = new_masker(test_input_mask, test_text)

    assert original_masked_output == new_masked_output


class _MaskerForAttributePropagation(shap.maskers.Masker):
    def __init__(self):
        self.shape = (None, 2)
        self.invariants = np.array([True, False])
        self.clustering = None
        self.feature_names = ["f1", "f2"]
        self.text_data = True

    def __call__(self, mask, *args):  # type: ignore[override]
        x = args[0]
        return ([x],)


class _NonTupleMasker(shap.maskers.Masker):
    def __init__(self):
        self.shape = (None, 0)

    def __call__(self, mask, *args):  # type: ignore[override]
        x = args[0]
        return np.asarray(x) * 2


class _TupleMasker(shap.maskers.Masker):
    def __init__(self):
        self.shape = (None, 0)

    def __call__(self, mask, *args):  # type: ignore[override]
        x = np.asarray(args[0])
        return ([x], [x + 1])


def test_fixed_composite_propagates_only_non_none_attributes():
    masker = _MaskerForAttributePropagation()
    composite = shap.maskers.FixedComposite(masker)

    assert composite.masker is masker
    assert composite.shape == (None, 2)
    np.testing.assert_array_equal(composite.invariants, np.array([True, False]))
    assert composite.feature_names == ["f1", "f2"]
    assert composite.text_data is True
    assert not hasattr(composite, "clustering")


def test_fixed_composite_call_wraps_non_tuple_masked_output_and_args():
    masker = _NonTupleMasker()
    composite = shap.maskers.FixedComposite(masker)

    x = np.array([1, 2, 3])
    result = composite(np.array([], dtype=bool), x, "label")

    assert isinstance(result, tuple)
    assert len(result) == 3
    np.testing.assert_array_equal(result[0], np.array([2, 4, 6]))
    np.testing.assert_array_equal(result[1], np.array([x]))
    np.testing.assert_array_equal(result[2], np.array(["label"]))


def test_fixed_composite_call_keeps_tuple_masked_output_shape():
    masker = _TupleMasker()
    composite = shap.maskers.FixedComposite(masker)

    x = np.array([4, 5])
    result = composite(np.array([], dtype=bool), x)

    assert isinstance(result, tuple)
    assert len(result) == 3
    np.testing.assert_array_equal(result[0][0], x)
    np.testing.assert_array_equal(result[1][0], x + 1)
    np.testing.assert_array_equal(result[2], np.array([x]))


def test_fixed_composite_serialization_round_trip_without_transformers():
    original_masker = shap.maskers.FixedComposite(shap.maskers.Fixed())

    stream = io.BytesIO()
    original_masker.save(stream)
    stream.seek(0)

    loaded_masker = shap.maskers.FixedComposite.load(stream)

    x = np.array([1, 2, 3])
    mask = np.array([], dtype=bool)
    original_result = original_masker(mask, x)
    loaded_result = loaded_masker(mask, x)

    np.testing.assert_array_equal(original_result[0][0], loaded_result[0][0])
    np.testing.assert_array_equal(original_result[1], loaded_result[1])


def test_fixed_composite_load_instantiate_false_returns_kwargs():
    masker = shap.maskers.FixedComposite(shap.maskers.Fixed())
    stream = io.BytesIO()
    masker.save(stream)
    stream.seek(0)

    loaded_type = pickle.load(stream)
    assert loaded_type is shap.maskers.FixedComposite

    kwargs = shap.maskers.FixedComposite.load(stream, instantiate=False)

    assert set(kwargs) == {"masker"}
    assert isinstance(kwargs["masker"], shap.maskers.Fixed)

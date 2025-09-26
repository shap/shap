"""This file contains tests for the FixedComposite masker."""

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

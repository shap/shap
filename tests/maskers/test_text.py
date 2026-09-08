"""This file contains tests for the Text masker."""

import sys
import tempfile

import numpy as np
import pytest

import shap

pytestmark = pytest.mark.skipif(sys.platform == "darwin", reason="Memory error on macOS")

TINY_DISTILBERT = "hf-internal-testing/tiny-random-DistilBertForSequenceClassification"
TINY_MARIAN_MT = "hf-internal-testing/tiny-random-MarianMTModel"
TINY_GPT2 = "hf-internal-testing/tiny-random-GPT2LMHeadModel"
TINY_BART = "hf-internal-testing/tiny-random-BartForConditionalGeneration"


def test_method_token_segments_pretrained_tokenizer():
    """Check that the Text masker produces the same segments as its non-fast pretrained tokenizer."""
    AutoTokenizer = pytest.importorskip("transformers").AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TINY_DISTILBERT, use_fast=False)
    masker = shap.maskers.Text(tokenizer)

    test_text = "I ate a Cannoli"
    output_token_segments, token_ids = masker.token_segments(test_text)

    assert "".join(output_token_segments) == test_text
    assert len(output_token_segments) == len(token_ids)


def test_method_token_segments_pretrained_tokenizer_fast():
    """Check that the Text masker produces the same segments as its fast pretrained tokenizer."""
    AutoTokenizer = pytest.importorskip("transformers").AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TINY_DISTILBERT, use_fast=True)
    masker = shap.maskers.Text(tokenizer)

    test_text = "I ate a Cannoli"
    output_token_segments, token_ids = masker.token_segments(test_text)

    assert "".join(output_token_segments) == test_text
    assert len(output_token_segments) == len(token_ids)


def test_masker_call_pretrained_tokenizer():
    """Check that the Text masker with a non-fast pretrained tokenizer masks correctly."""
    AutoTokenizer = pytest.importorskip("transformers").AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TINY_DISTILBERT, use_fast=False)
    masker = shap.maskers.Text(tokenizer)

    test_text = "I ate a Cannoli"
    test_input_mask = np.ones(masker.shape(test_text)[1], dtype=bool)
    if len(test_input_mask) > 1:
        test_input_mask[1] = False

    output_masked_text = masker(test_input_mask, test_text)
    output_masked_text = output_masked_text[0][0]

    assert "[MASK]" in output_masked_text
    assert "ate" in output_masked_text.lower()


def test_masker_call_pretrained_tokenizer_fast():
    """Check that the Text masker with a fast pretrained tokenizer masks correctly."""
    AutoTokenizer = pytest.importorskip("transformers").AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TINY_DISTILBERT, use_fast=True)
    masker = shap.maskers.Text(tokenizer)

    test_text = "I ate a Cannoli"
    test_input_mask = np.ones(masker.shape(test_text)[1], dtype=bool)
    if len(test_input_mask) > 1:
        test_input_mask[1] = False

    output_masked_text = masker(test_input_mask, test_text)
    output_masked_text = output_masked_text[0][0]

    assert "[MASK]" in output_masked_text
    assert "ate" in output_masked_text.lower()


@pytest.mark.filterwarnings(r"ignore:Recommended. pip install sacremoses")
def test_sentencepiece_tokenizer_output():
    """Tests for output for sentencepiece tokenizers to not have '_' in output of masker when passed a mask of ones."""
    AutoTokenizer = pytest.importorskip("transformers").AutoTokenizer
    pytest.importorskip("sentencepiece")

    tokenizer = AutoTokenizer.from_pretrained(TINY_MARIAN_MT)
    masker = shap.maskers.Text(tokenizer)

    s = "This is a test statement for sentencepiece tokenizer"
    mask = np.ones(masker.shape(s)[1], dtype=bool)

    sentencepiece_tokenizer_output_processed = masker(mask, s)
    expected_sentencepiece_tokenizer_output_processed = "This is a test statement for sentencepiece tokenizer"
    # since we expect output wrapped in a tuple hence the indexing [0][0] to extract the string
    assert sentencepiece_tokenizer_output_processed[0][0] == expected_sentencepiece_tokenizer_output_processed


def test_keep_prefix_suffix_tokenizer_parsing():
    """Checks parsed keep prefix and keep suffix for different tokenizers."""
    AutoTokenizer = pytest.importorskip("transformers").AutoTokenizer

    tokenizer_mt = AutoTokenizer.from_pretrained(TINY_MARIAN_MT)
    tokenizer_gpt = AutoTokenizer.from_pretrained(TINY_GPT2)
    tokenizer_bart = AutoTokenizer.from_pretrained(TINY_BART)

    masker_mt = shap.maskers.Text(tokenizer_mt)
    masker_gpt = shap.maskers.Text(tokenizer_gpt)
    masker_bart = shap.maskers.Text(tokenizer_bart)

    assert masker_gpt.keep_prefix == 0
    assert masker_gpt.keep_suffix == 0
    assert masker_mt.keep_prefix + masker_mt.keep_suffix == len(tokenizer_mt("")["input_ids"])
    assert masker_bart.keep_prefix + masker_bart.keep_suffix == len(tokenizer_bart("")["input_ids"])


def test_keep_prefix_suffix_tokenizer_parsing_mistralai():
    AutoTokenizer = pytest.importorskip("transformers").AutoTokenizer

    tokenizer_mistral = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    masker_mistral = shap.maskers.Text(tokenizer_mistral)
    masker_mistral_expected_keep_prefix, masker_mistral_expected_keep_suffix = 1, 0

    assert masker_mistral.keep_prefix == masker_mistral_expected_keep_prefix
    assert masker_mistral.keep_suffix == masker_mistral_expected_keep_suffix


def test_text_infill_with_collapse_mask_token_mistralai():
    AutoTokenizer = pytest.importorskip("transformers").AutoTokenizer

    tokenizer_mistral = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    masker_mistral = shap.maskers.Text(tokenizer_mistral, mask_token="...", collapse_mask_token=True)

    s = "This is a test string to be infilled"

    # s_masked_with_infill_ex1 = "This is a test string ... ... ..."
    mask_ex1 = np.array([True, True, True, True, True, False, False, False, False])
    # s_masked_with_infill_ex2 = "This is a ... ... to be infilled"
    mask_ex2 = np.array([True, True, True, False, False, True, True, True, True])
    # s_masked_with_infill_ex3 = "... ... ... test string to be infilled"
    mask_ex3 = np.array([False, False, False, True, True, True, True, True, True])
    # s_masked_with_infill_ex4 = "... ... ... ... ... ... ... ..."
    mask_ex4 = np.array([False, False, False, False, False, False, False, False, False])

    text_infilled_ex1_mist = masker_mistral(np.append(True, mask_ex1), s)[0][0]
    expected_text_infilled_ex1 = "This is a test string ..."

    text_infilled_ex2_mist = masker_mistral(np.append(True, mask_ex2), s)[0][0]
    expected_text_infilled_ex2 = "This is a ... to be infilled"

    text_infilled_ex3_mist = masker_mistral(np.append(True, mask_ex3), s)[0][0]
    expected_text_infilled_ex3 = "... test string to be infilled"

    text_infilled_ex4_mist = masker_mistral(np.append(True, mask_ex4), s)[0][0]
    expected_text_infilled_ex4 = "..."

    assert text_infilled_ex1_mist == expected_text_infilled_ex1
    assert text_infilled_ex2_mist == expected_text_infilled_ex2
    assert text_infilled_ex3_mist == expected_text_infilled_ex3
    assert text_infilled_ex4_mist == expected_text_infilled_ex4


def test_text_infill_with_collapse_mask_token():
    """Tests for different text infilling output combinations with collapsing mask token."""
    AutoTokenizer = pytest.importorskip("transformers").AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TINY_GPT2)
    masker = shap.maskers.Text(tokenizer, mask_token="...", collapse_mask_token=True)

    s = "This is a test string to be infilled"
    num_tokens = masker.shape(s)[1]

    # mask suffix
    mask_ex1 = np.ones(num_tokens, dtype=bool)
    split1 = max(1, num_tokens // 2)
    mask_ex1[split1:] = False

    # mask middle span
    mask_ex2 = np.ones(num_tokens, dtype=bool)
    start2 = max(1, num_tokens // 3)
    end2 = min(num_tokens, max(start2 + 1, (2 * num_tokens) // 3))
    mask_ex2[start2:end2] = False

    # mask prefix
    mask_ex3 = np.ones(num_tokens, dtype=bool)
    end3 = max(1, num_tokens // 3)
    mask_ex3[:end3] = False

    # mask all
    mask_ex4 = np.zeros(num_tokens, dtype=bool)

    text_infilled_ex1 = masker(mask_ex1, s)[0][0]
    text_infilled_ex2 = masker(mask_ex2, s)[0][0]
    text_infilled_ex3 = masker(mask_ex3, s)[0][0]
    text_infilled_ex4 = masker(mask_ex4, s)[0][0]

    # Each contiguous masked region should collapse to one mask token.
    assert text_infilled_ex1.count("...") == 1
    assert text_infilled_ex2.count("...") == 1
    assert text_infilled_ex3.count("...") == 1
    assert text_infilled_ex4 == "..."

    assert "This" in text_infilled_ex1
    assert "infilled" in text_infilled_ex2
    assert text_infilled_ex3.startswith("...")


def test_serialization_text_masker():
    """Make sure text serialization works."""
    AutoTokenizer = pytest.importorskip("transformers").AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TINY_DISTILBERT, use_fast=False)
    original_masker = shap.maskers.Text(tokenizer)

    with tempfile.TemporaryFile() as temp_serialization_file:
        original_masker.save(temp_serialization_file)

        temp_serialization_file.seek(0)

        # deserialize masker
        new_masker = shap.maskers.Text.load(temp_serialization_file)

    test_text = "I ate a Cannoli"
    test_input_mask = np.ones(original_masker.shape(test_text)[1], dtype=bool)
    if len(test_input_mask) > 1:
        test_input_mask[1] = False

    original_masked_output = original_masker(test_input_mask, test_text)
    new_masked_output = new_masker(test_input_mask, test_text)

    assert original_masked_output == new_masked_output


def test_serialization_text_masker_custom_mask():
    """Make sure text serialization works with custom mask."""
    AutoTokenizer = pytest.importorskip("transformers").AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TINY_DISTILBERT, use_fast=True)
    original_masker = shap.maskers.Text(tokenizer, mask_token="[CUSTOM-MASK]")

    with tempfile.TemporaryFile() as temp_serialization_file:
        original_masker.save(temp_serialization_file)

        temp_serialization_file.seek(0)

        # deserialize masker
        new_masker = shap.maskers.Text.load(temp_serialization_file)

    test_text = "I ate a Cannoli"
    test_input_mask = np.ones(original_masker.shape(test_text)[1], dtype=bool)
    if len(test_input_mask) > 1:
        test_input_mask[1] = False

    original_masked_output = original_masker(test_input_mask, test_text)
    new_masked_output = new_masker(test_input_mask, test_text)

    assert original_masked_output == new_masked_output


def test_serialization_text_masker_collapse_mask_token():
    """Make sure text serialization works with collapse mask token."""
    AutoTokenizer = pytest.importorskip("transformers").AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TINY_DISTILBERT, use_fast=True)
    original_masker = shap.maskers.Text(tokenizer, collapse_mask_token=True)

    with tempfile.TemporaryFile() as temp_serialization_file:
        original_masker.save(temp_serialization_file)

        temp_serialization_file.seek(0)

        # deserialize masker
        new_masker = shap.maskers.Text.load(temp_serialization_file)

    test_text = "I ate a Cannoli"
    test_input_mask = np.ones(original_masker.shape(test_text)[1], dtype=bool)
    if len(test_input_mask) > 1:
        test_input_mask[1] = False

    original_masked_output = original_masker(test_input_mask, test_text)
    new_masked_output = new_masker(test_input_mask, test_text)

    assert original_masked_output == new_masked_output

"""This file contains tests for the Text masker."""

import tempfile

import numpy as np
import pytest

import shap


def test_method_token_segments_pretrained_tokenizer():
    """Check that the Text masker produces the same segments as its non-fast pretrained tokenizer."""
    AutoTokenizer = pytest.importorskip("transformers").AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased", use_fast=False)
    masker = shap.maskers.Text(tokenizer)

    test_text = "I ate a Cannoli"
    output_token_segments, _ = masker.token_segments(test_text)
    correct_token_segments = ["", " I", " ate", " a", " Can", "no", "li", ""]

    assert output_token_segments == correct_token_segments


def test_method_token_segments_pretrained_tokenizer_fast():
    """Check that the Text masker produces the same segments as its fast pretrained tokenizer."""
    AutoTokenizer = pytest.importorskip("transformers").AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=True)
    masker = shap.maskers.Text(tokenizer)

    test_text = "I ate a Cannoli"
    output_token_segments, _ = masker.token_segments(test_text)
    correct_token_segments = ["", "I ", "ate ", "a ", "Can", "no", "li", ""]

    assert output_token_segments == correct_token_segments


def test_masker_call_pretrained_tokenizer():
    """Check that the Text masker with a non-fast pretrained tokenizer masks correctly."""
    AutoTokenizer = pytest.importorskip("transformers").AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=False)
    masker = shap.maskers.Text(tokenizer)

    test_text = "I ate a Cannoli"
    test_input_mask = np.array([True, False, True, True, False, True, True, True])

    output_masked_text = masker(test_input_mask, test_text)
    correct_masked_text = "[MASK] ate a [MASK]noli"

    assert output_masked_text[0] == correct_masked_text


def test_masker_call_pretrained_tokenizer_fast():
    """Check that the Text masker with a fast pretrained tokenizer masks correctly."""
    AutoTokenizer = pytest.importorskip("transformers").AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=True)
    masker = shap.maskers.Text(tokenizer)

    test_text = "I ate a Cannoli"
    test_input_mask = np.array([True, False, True, True, False, True, True, True])

    output_masked_text = masker(test_input_mask, test_text)
    correct_masked_text = "[MASK]ate a [MASK]noli"

    assert output_masked_text[0] == correct_masked_text


@pytest.mark.filterwarnings(r"ignore:Recommended. pip install sacremoses")
def test_sentencepiece_tokenizer_output():
    """Tests for output for sentencepiece tokenizers to not have '_' in output of masker when passed a mask of ones."""
    AutoTokenizer = pytest.importorskip("transformers").AutoTokenizer
    pytest.importorskip("sentencepiece")

    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
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

    tokenizer_mt = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
    tokenizer_gpt = AutoTokenizer.from_pretrained("gpt2")
    tokenizer_bart = AutoTokenizer.from_pretrained("sshleifer/distilbart-xsum-12-6")

    masker_mt = shap.maskers.Text(tokenizer_mt)
    masker_gpt = shap.maskers.Text(tokenizer_gpt)
    masker_bart = shap.maskers.Text(tokenizer_bart)

    masker_mt_expected_keep_prefix, masker_mt_expected_keep_suffix = 0, 1
    masker_gpt_expected_keep_prefix, masker_gpt_expected_keep_suffix = 0, 0
    masker_bart_expected_keep_prefix, masker_bart_expected_keep_suffix = 1, 1

    assert masker_mt.keep_prefix == masker_mt_expected_keep_prefix
    assert masker_mt.keep_suffix == masker_mt_expected_keep_suffix
    assert masker_gpt.keep_prefix == masker_gpt_expected_keep_prefix
    assert masker_gpt.keep_suffix == masker_gpt_expected_keep_suffix
    assert masker_bart.keep_prefix == masker_bart_expected_keep_prefix
    assert masker_bart.keep_suffix == masker_bart_expected_keep_suffix


@pytest.mark.xfail(reason="gated repository. Find alternative.")
def test_keep_prefix_suffix_tokenizer_parsing_mistralai():
    AutoTokenizer = pytest.importorskip("transformers").AutoTokenizer

    tokenizer_mistral = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    masker_mistral = shap.maskers.Text(tokenizer_mistral)
    masker_mistral_expected_keep_prefix, masker_mistral_expected_keep_suffix = 1, 0

    assert masker_mistral.keep_prefix == masker_mistral_expected_keep_prefix
    assert masker_mistral.keep_suffix == masker_mistral_expected_keep_suffix


@pytest.mark.xfail(reason="gated repository. Find alternative.")
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

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    masker = shap.maskers.Text(tokenizer, mask_token="...", collapse_mask_token=True)

    s = "This is a test string to be infilled"

    # s_masked_with_infill_ex1 = "This is a test string ... ... ..."
    mask_ex1 = np.array([True, True, True, True, True, False, False, False, False])
    # s_masked_with_infill_ex2 = "This is a ... ... to be infilled"
    mask_ex2 = np.array([True, True, True, False, False, True, True, True, True])
    # s_masked_with_infill_ex3 = "... ... ... test string to be infilled"
    mask_ex3 = np.array([False, False, False, True, True, True, True, True, True])
    # s_masked_with_infill_ex4 = "... ... ... ... ... ... ... ..."
    mask_ex4 = np.array([False, False, False, False, False, False, False, False, False])

    text_infilled_ex1 = masker(mask_ex1, s)[0][0]
    expected_text_infilled_ex1 = "This is a test string ..."

    text_infilled_ex2 = masker(mask_ex2, s)[0][0]
    expected_text_infilled_ex2 = "This is a ... to be infilled"

    text_infilled_ex3 = masker(mask_ex3, s)[0][0]
    expected_text_infilled_ex3 = "... test string to be infilled"

    text_infilled_ex4 = masker(mask_ex4, s)[0][0]
    expected_text_infilled_ex4 = "..."

    assert text_infilled_ex1 == expected_text_infilled_ex1
    assert text_infilled_ex2 == expected_text_infilled_ex2
    assert text_infilled_ex3 == expected_text_infilled_ex3
    assert text_infilled_ex4 == expected_text_infilled_ex4


def test_serialization_text_masker():
    """Make sure text serialization works."""
    AutoTokenizer = pytest.importorskip("transformers").AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased", use_fast=False)
    original_masker = shap.maskers.Text(tokenizer)

    with tempfile.TemporaryFile() as temp_serialization_file:
        original_masker.save(temp_serialization_file)

        temp_serialization_file.seek(0)

        # deserialize masker
        new_masker = shap.maskers.Text.load(temp_serialization_file)

    test_text = "I ate a Cannoli"
    test_input_mask = np.array([True, False, True, True, False, True, True, True])

    original_masked_output = original_masker(test_input_mask, test_text)
    new_masked_output = new_masker(test_input_mask, test_text)

    assert original_masked_output == new_masked_output


def test_serialization_text_masker_custom_mask():
    """Make sure text serialization works with custom mask."""
    AutoTokenizer = pytest.importorskip("transformers").AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased", use_fast=True)
    original_masker = shap.maskers.Text(tokenizer, mask_token="[CUSTOM-MASK]")

    with tempfile.TemporaryFile() as temp_serialization_file:
        original_masker.save(temp_serialization_file)

        temp_serialization_file.seek(0)

        # deserialize masker
        new_masker = shap.maskers.Text.load(temp_serialization_file)

    test_text = "I ate a Cannoli"
    test_input_mask = np.array([True, False, True, True, False, True, True, True])

    original_masked_output = original_masker(test_input_mask, test_text)
    new_masked_output = new_masker(test_input_mask, test_text)

    assert original_masked_output == new_masked_output


def test_serialization_text_masker_collapse_mask_token():
    """Make sure text serialization works with collapse mask token."""
    AutoTokenizer = pytest.importorskip("transformers").AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased", use_fast=True)
    original_masker = shap.maskers.Text(tokenizer, collapse_mask_token=True)

    with tempfile.TemporaryFile() as temp_serialization_file:
        original_masker.save(temp_serialization_file)

        temp_serialization_file.seek(0)

        # deserialize masker
        new_masker = shap.maskers.Text.load(temp_serialization_file)

    test_text = "I ate a Cannoli"
    test_input_mask = np.array([True, False, True, True, False, True, True, True])

    original_masked_output = original_masker(test_input_mask, test_text)
    new_masked_output = new_masker(test_input_mask, test_text)

    assert original_masked_output == new_masked_output

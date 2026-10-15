from unittest.mock import MagicMock, patch

import pytest

import shap
from shap.utils.transformers import (
    getattr_silent,
    is_transformers_lm,
    parse_prefix_suffix_for_tokenizer,
)


def test_getattr_silent_existing_attr():
    class Obj:
        x = 5

    assert getattr_silent(Obj(), "x") == 5


def test_getattr_silent_missing_attr():
    class Obj:
        pass

    assert getattr_silent(Obj(), "missing") is None


def test_getattr_silent_none_string_bug():
    class Obj:
        x = "None"

    assert getattr_silent(Obj(), "x") is None


def test_getattr_silent_verbose_reset():
    class Obj:
        verbose = True
        x = 42

    obj = Obj()
    getattr_silent(obj, "x")
    assert obj.verbose


def test_getattr_silent_via_text_masker():
    tok = MagicMock()
    tok.mask_token = "[MASK]"
    tok.mask_token_id = 103
    tok.return_value = {"input_ids": []}
    tok.special_tokens_map = {}
    masker = shap.maskers.Text(tok)
    assert masker.mask_token == "[MASK]"


def test_is_transformers_lm_returns_false_for_non_model():
    assert is_transformers_lm("not a model") is False


def test_is_transformers_lm_returns_false_for_non_lm():
    mock_model = MagicMock()
    with patch("shap.utils.transformers.safe_isinstance", return_value=True):
        with patch("transformers.MODEL_FOR_CAUSAL_LM_MAPPING") as mock_causal:
            with patch("transformers.MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING") as mock_seq:
                mock_causal.values.return_value = []
                mock_seq.values.return_value = []
                assert is_transformers_lm(mock_model) is False


def test_parse_prefix_suffix_even_null_tokens():
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {"input_ids": [0, 2]}
    result = parse_prefix_suffix_for_tokenizer(mock_tokenizer)
    assert result["keep_prefix"] == 1
    assert result["keep_suffix"] == 1
    assert result["null_tokens"] == [0, 2]


def test_parse_prefix_suffix_eos_token():
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {"input_ids": [2]}
    mock_tokenizer.special_tokens_map = {"eos_token": "</s>"}
    mock_tokenizer.decode.return_value = "</s>"
    result = parse_prefix_suffix_for_tokenizer(mock_tokenizer)
    assert result["keep_prefix"] == 0
    assert result["keep_suffix"] == 1


def test_parse_prefix_suffix_bos_token():
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {"input_ids": [0]}
    mock_tokenizer.special_tokens_map = {"bos_token": "<s>"}
    mock_tokenizer.decode.return_value = "<s>"
    result = parse_prefix_suffix_for_tokenizer(mock_tokenizer)
    assert result["keep_prefix"] == 1
    assert result["keep_suffix"] == 0


def test_parse_prefix_suffix_raises_exception():
    class MinimalTokenizer:
        def __call__(self, text):
            return {"input_ids": [2]}

    with pytest.raises(Exception):
        parse_prefix_suffix_for_tokenizer(MinimalTokenizer())

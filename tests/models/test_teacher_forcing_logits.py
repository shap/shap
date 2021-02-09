""" This file contains tests for the TeacherForcingLogits class.
"""

import pytest
import numpy as np
import shap
from shap.utils.transformers import parse_prefix_suffix_for_tokenizer


def test_method_get_teacher_forced_logits_for_encoder_decoder_model():
    """ Tests if get_teacher_forced_logits() works for encoder-decoder models.
    """

    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    tokenizer = transformers.AutoTokenizer.from_pretrained("sshleifer/distilbart-xsum-12-6")
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-xsum-12-6")

    wrapped_model = shap.models.TeacherForcingLogits(model, tokenizer, device='cpu')

    source_sentence = "This is a test statement for verifying working of teacher forcing logits functionality"
    target_sentence = "Testing teacher forcing logits functionality"

    source_sentence_ids = torch.tensor([wrapped_model.similarity_tokenizer.encode(source_sentence)])

    parsed_tokenizer_dict = parse_prefix_suffix_for_tokenizer(wrapped_model.similarity_tokenizer)
    keep_prefix, keep_suffix = parsed_tokenizer_dict['keep_prefix'], parsed_tokenizer_dict['keep_suffix']

    if keep_suffix > 0:
        target_sentence_ids = torch.tensor([wrapped_model.similarity_tokenizer.encode(target_sentence)])[:, keep_prefix:-keep_suffix]
    else:
        target_sentence_ids = torch.tensor([wrapped_model.similarity_tokenizer.encode(target_sentence)])[:, keep_prefix:]

    # call the get teacher forced logits function
    logits = wrapped_model.get_teacher_forced_logits(source_sentence_ids, target_sentence_ids)

    assert not np.isnan(np.sum(logits))

def test_method_get_teacher_forced_logits_for_decoder_model():
    """ Tests if get_teacher_forced_logits() works for decoder only models.
    """

    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    model.config.is_decoder = True

    wrapped_model = shap.models.TeacherForcingLogits(model, tokenizer, device='cpu')

    source_sentence = "This is a test statement for verifying"
    target_sentence = "working of teacher forcing logits functionality"

    source_sentence_ids = torch.tensor([wrapped_model.similarity_tokenizer.encode(source_sentence)])

    parsed_tokenizer_dict = parse_prefix_suffix_for_tokenizer(wrapped_model.similarity_tokenizer)
    keep_prefix, keep_suffix = parsed_tokenizer_dict['keep_prefix'], parsed_tokenizer_dict['keep_suffix']

    if keep_suffix > 0:
        target_sentence_ids = torch.tensor([wrapped_model.similarity_tokenizer.encode(target_sentence)])[:, keep_prefix:-keep_suffix]
    else:
        target_sentence_ids = torch.tensor([wrapped_model.similarity_tokenizer.encode(target_sentence)])[:, keep_prefix:]

    # call the get teacher forced logits function
    logits = wrapped_model.get_teacher_forced_logits(source_sentence_ids, target_sentence_ids)

    assert not np.isnan(np.sum(logits))

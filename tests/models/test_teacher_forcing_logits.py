"""This file contains tests for the TeacherForcingLogits class."""

import numpy as np
import pytest

import shap


def test_falcon():
    transformers = pytest.importorskip("transformers")
    requests = pytest.importorskip("requests")
    name = "fxmarty/really-tiny-falcon-testing"
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(name)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            name, trust_remote_code=True, load_in_8bit=False, low_cpu_mem_usage=False
        )
    except requests.exceptions.RequestException:
        pytest.xfail(reason="Connection error to transformers model")

    model = model.eval()

    s = ["I enjoy walking with my cute dog"]
    gen_dict = dict(
        max_new_tokens=100,
        num_beams=5,
        renormalize_logits=True,
        no_repeat_ngram_size=8,
    )

    model.config.task_specific_params = dict()
    model.config.task_specific_params["text-generation"] = gen_dict
    shap_model = shap.models.TeacherForcing(model, tokenizer)

    explainer = shap.Explainer(shap_model, tokenizer)
    shap_values = explainer(s)
    assert not np.isnan(np.sum(shap_values.values))


def test_method_get_teacher_forced_logits_for_encoder_decoder_model():
    """Tests if get_teacher_forced_logits() works for encoder-decoder models."""
    transformers = pytest.importorskip("transformers")
    requests = pytest.importorskip("requests")

    name = "hf-internal-testing/tiny-random-BartModel"
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(name)
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(name)
    except requests.exceptions.RequestException:
        pytest.xfail(reason="Connection error to transformers model")

    wrapped_model = shap.models.TeacherForcing(model, tokenizer, device="cpu")

    source_sentence = np.array(
        ["This is a test statement for verifying working of teacher forcing logits functionality"]
    )
    target_sentence = np.array(["Testing teacher forcing logits functionality"])

    # call the get teacher forced logits function
    logits = wrapped_model.get_teacher_forced_logits(source_sentence, target_sentence)

    assert not np.isnan(np.sum(logits))


def test_method_get_teacher_forced_logits_for_decoder_model():
    """Tests if get_teacher_forced_logits() works for decoder only models."""
    transformers = pytest.importorskip("transformers")
    requests = pytest.importorskip("requests")

    name = "hf-internal-testing/tiny-random-gpt2"
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(name)
        model = transformers.AutoModelForCausalLM.from_pretrained(name)
    except requests.exceptions.RequestException:
        pytest.xfail(reason="Connection error to transformers model")

    model.config.is_decoder = True

    wrapped_model = shap.models.TeacherForcing(model, tokenizer, device="cpu")

    source_sentence = np.array(["This is a test statement for verifying"])
    target_sentence = np.array(["working of teacher forcing logits functionality"])

    # call the get teacher forced logits function
    logits = wrapped_model.get_teacher_forced_logits(source_sentence, target_sentence)

    assert not np.isnan(np.sum(logits))

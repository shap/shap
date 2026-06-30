"""This file contains tests for the TeacherForcingLogits class."""

import platform

import numpy as np
import pytest

import shap


def _from_pretrained_or_skip(loader, name, **kwargs):
    try:
        return loader.from_pretrained(name, **kwargs)
    except Exception as exc:
        pytest.skip(f"Could not load {name}: {exc}")


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping on MacOS due to torch segmentation error, see GH #4075.",
)
def test_falcon():
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    name = "hf-internal-testing/tiny-random-gpt2"

    tokenizer = _from_pretrained_or_skip(transformers.AutoTokenizer, name)
    model = _from_pretrained_or_skip(transformers.AutoModelForCausalLM, name)

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
    assert not np.isnan(np.sum(shap_values.values))  # type: ignore[union-attr]


def test_method_get_teacher_forced_logits_for_encoder_decoder_model():
    """Tests if get_teacher_forced_logits() works for encoder-decoder models."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    name = "hf-internal-testing/tiny-random-BartModel"
    tokenizer = _from_pretrained_or_skip(transformers.AutoTokenizer, name)
    model = _from_pretrained_or_skip(transformers.AutoModelForSeq2SeqLM, name)

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
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    name = "hf-internal-testing/tiny-random-gpt2"
    tokenizer = _from_pretrained_or_skip(transformers.AutoTokenizer, name)
    model = _from_pretrained_or_skip(transformers.AutoModelForCausalLM, name)

    model.config.is_decoder = True

    wrapped_model = shap.models.TeacherForcing(model, tokenizer, device="cpu")

    source_sentence = np.array(["This is a test statement for verifying"])
    target_sentence = np.array(["working of teacher forcing logits functionality"])

    # call the get teacher forced logits function
    logits = wrapped_model.get_teacher_forced_logits(source_sentence, target_sentence)

    assert not np.isnan(np.sum(logits))

"""Unit tests for shap.models.TeacherForcing internals."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import shap
import shap.models._teacher_forcing as teacher_forcing_module


class DummyTokenizer:
    """Minimal tokenizer mock for TeacherForcing unit tests."""

    def __init__(self, token_map: dict[str, list[int]] | None = None):
        self.pad_token = None
        self.eos_token = "<eos>"
        self.padding_side = "right"
        self.token_map = token_map or {}
        self.decode_calls = []
        self.call_inputs = None
        self.call_return_tensors = None
        self.call_padding_side = None

    def decode(self, token_ids):
        self.decode_calls.append(token_ids)
        return f"tok-{token_ids[0]}"

    def __call__(self, sentences, return_tensors=None, padding=True):
        self.call_inputs = list(sentences)
        self.call_return_tensors = return_tensors
        self.call_padding_side = self.padding_side

        if self.token_map:
            input_ids = [self.token_map[s] for s in sentences]
        else:
            input_ids = [[1, 2, 3] for _ in sentences]

        return {"input_ids": input_ids, "attention_mask": [[1] * len(ids) for ids in input_ids]}


def _make_model_agnostic_wrapper(tokenizer: DummyTokenizer | None = None):
    tokenizer = tokenizer or DummyTokenizer()
    similarity_model = SimpleNamespace(config=SimpleNamespace(is_encoder_decoder=False, is_decoder=True))
    return shap.models.TeacherForcing(
        lambda x: [f"generated-{v}" for v in x],
        similarity_model=similarity_model,
        similarity_tokenizer=tokenizer,
        batch_size=2,
    )


def test_update_output_names_only_recomputes_for_new_outputs():
    wrapped_model = _make_model_agnostic_wrapper()
    calls = []

    def fake_get_output_names(output):
        calls.append(output.copy())
        return ["name"]

    wrapped_model.get_output_names = fake_get_output_names

    first_output = np.array(["hello"])
    wrapped_model.update_output_names(first_output)
    wrapped_model.update_output_names(first_output.copy())
    wrapped_model.update_output_names(np.array(["new target"]))

    assert len(calls) == 2
    assert wrapped_model.output_names == ["name"]


def test_get_outputs_handles_text_and_token_ids(monkeypatch):
    tokenizer = DummyTokenizer(token_map={"first": [100, 11, 12, 13, 200], "second": [100, 21, 22, 23, 200]})
    wrapped_model = _make_model_agnostic_wrapper(tokenizer)

    monkeypatch.setattr(
        teacher_forcing_module,
        "parse_prefix_suffix_for_tokenizer",
        lambda _: {"keep_prefix": 1, "keep_suffix": 1},
    )

    as_text = np.array(["first", "second"])
    as_ids = np.array([[7, 8, 9], [10, 11, 12]])

    sliced_ids = wrapped_model.get_outputs(as_text)
    passthrough_ids = wrapped_model.get_outputs(as_ids)

    np.testing.assert_array_equal(sliced_ids, np.array([[11, 12, 13], [21, 22, 23]]))
    np.testing.assert_array_equal(passthrough_ids, as_ids)


def test_get_inputs_model_agnostic_uses_inner_model_output_and_restores_padding():
    tokenizer = DummyTokenizer()
    wrapped_model = _make_model_agnostic_wrapper(tokenizer)

    inputs = wrapped_model.get_inputs(np.array([1, 2, 3]), padding_side="left")

    assert tokenizer.call_inputs == ["generated-1", "generated-2", "generated-3"]
    assert tokenizer.call_padding_side == "left"
    assert tokenizer.padding_side == "right"
    assert set(inputs) == {"input_ids", "attention_mask"}


def test_get_teacher_forced_logits_encoder_decoder_uses_decoder_start_token():
    wrapped_model = _make_model_agnostic_wrapper()
    wrapped_model.similarity_model.config = SimpleNamespace(is_encoder_decoder=True, decoder_start_token_id=9)

    wrapped_model.get_outputs = lambda y: np.array([[4, 5]])
    wrapped_model.get_inputs = lambda x, padding_side="right": {"from_padding": padding_side}
    captured = {}

    def fake_model_inference(inputs, output_ids):
        captured["inputs"] = inputs
        captured["output_ids"] = output_ids
        return np.arange(1 * 3 * 6).reshape(1, 3, 6).astype("float64")

    wrapped_model.model_inference = fake_model_inference

    logits = wrapped_model.get_teacher_forced_logits(np.array(["x"]), np.array(["y"]))

    assert captured["inputs"] == {"from_padding": "right"}
    np.testing.assert_array_equal(captured["output_ids"], np.array([[9, 4, 5]]))
    assert logits.shape == (1, 2, 6)


def test_get_teacher_forced_logits_decoder_only_extracts_target_window():
    wrapped_model = _make_model_agnostic_wrapper()
    wrapped_model.similarity_model.config = SimpleNamespace(is_encoder_decoder=False, is_decoder=True)

    wrapped_model.get_outputs = lambda y: np.array([[7, 8]])
    wrapped_model.get_inputs = lambda x, padding_side="left": {"from_padding": padding_side}
    wrapped_model.model_inference = lambda inputs, output_ids: np.arange(1 * 6 * 4).reshape(1, 6, 4).astype("float64")

    logits = wrapped_model.get_teacher_forced_logits(np.array(["x"]), np.array(["y"]))

    assert logits.shape == (1, 2, 4)
    np.testing.assert_array_equal(logits, np.arange(1 * 6 * 4).reshape(1, 6, 4).astype("float64")[:, -3:-1, :])


def test_get_teacher_forced_logits_encoder_decoder_requires_start_or_bos():
    wrapped_model = _make_model_agnostic_wrapper()
    wrapped_model.similarity_model.config = SimpleNamespace(
        is_encoder_decoder=True,
        decoder_start_token_id=None,
        bos_token_id=None,
    )
    wrapped_model.get_outputs = lambda y: np.array([[4, 5]])
    wrapped_model.get_inputs = lambda x, padding_side="right": {"from_padding": padding_side}

    with pytest.raises(ValueError, match="No decoder_start_token_id or bos_token_id"):
        wrapped_model.get_teacher_forced_logits(np.array(["x"]), np.array(["y"]))


def test_call_batches_and_concatenates_outputs():
    wrapped_model = _make_model_agnostic_wrapper()
    captured = {"updates": 0}

    def fake_update_output_names(_):
        captured["updates"] += 1

    wrapped_model.update_output_names = fake_update_output_names
    wrapped_model.get_teacher_forced_logits = lambda x, y: np.zeros((len(x), 2, 3))
    wrapped_model.get_logodds = lambda logits: np.full((logits.shape[0], 2), logits.shape[0], dtype="float64")

    x = np.array(["a", "b", "c", "d", "e"])
    y = np.array(["A", "B", "C", "D", "E"])
    scores = wrapped_model(x, y)

    assert captured["updates"] == 1
    assert scores.shape == (5, 2)
    np.testing.assert_array_equal(scores, np.array([[2, 2], [2, 2], [2, 2], [2, 2], [1, 1]], dtype="float64"))

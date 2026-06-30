"""Tests for shap.models.TransformersPipeline."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import scipy.special

from shap.models import TransformersPipeline


def _make_mock_pipeline(label2id=None, id2label=None, num_labels=None, output=None):
    """Build a minimal mock pipeline used by TransformersPipeline."""
    config = MagicMock()
    config.label2id = label2id
    config.id2label = id2label
    config.num_labels = num_labels

    model = MagicMock()
    model.config = config

    pipe = MagicMock()
    pipe.model = model
    if output is not None:
        pipe.return_value = output
    return pipe


def test_init_with_label2id_and_id2label():
    pipe = _make_mock_pipeline(
        label2id={"NEGATIVE": "0", "POSITIVE": "1"},
        id2label={"0": "NEGATIVE", "1": "POSITIVE"},
    )

    model = TransformersPipeline(pipe)

    assert model.label2id == {"NEGATIVE": 0, "POSITIVE": 1}
    assert model.id2label == {0: "NEGATIVE", 1: "POSITIVE"}
    assert model.output_shape == (2,)
    assert model.output_names == ["NEGATIVE", "POSITIVE"]


def test_init_fallback_to_id2label():
    pipe = _make_mock_pipeline(label2id=None, id2label={"0": "NEGATIVE", "1": "POSITIVE"})

    model = TransformersPipeline(pipe)

    assert model.label2id == {"NEGATIVE": 0, "POSITIVE": 1}
    assert model.output_shape == (2,)


def test_init_fallback_to_num_labels():
    pipe = _make_mock_pipeline(label2id=None, id2label=None, num_labels=3)

    model = TransformersPipeline(pipe)

    assert model.label2id == {"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2}
    assert model.id2label == {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}
    assert model.output_shape == (3,)


def test_init_missing_mappings_raises():
    pipe = _make_mock_pipeline(label2id=None, id2label=None, num_labels=None)

    with pytest.raises(ValueError, match="Could not determine label mapping"):
        TransformersPipeline(pipe)


def test_init_missing_model_or_config_raises():
    pipe_no_model = MagicMock(spec=[])
    with pytest.raises(AttributeError, match="does not expose a config"):
        TransformersPipeline(pipe_no_model)

    model_no_config = MagicMock(spec=[])
    pipe_no_config = MagicMock()
    pipe_no_config.model = model_no_config
    with pytest.raises(AttributeError, match="does not expose a config"):
        TransformersPipeline(pipe_no_config)


def test_call_nested_list_output():
    output = [[{"label": "NEGATIVE", "score": 0.1}, {"label": "POSITIVE", "score": 0.9}]]
    pipe = _make_mock_pipeline(
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        output=output,
    )

    model = TransformersPipeline(pipe)
    result = model(["hello world"])

    assert result.shape == (1, 2)
    np.testing.assert_allclose(result, [[0.1, 0.9]])


def test_call_flat_dict_output_topk1_style():
    output = [{"label": "POSITIVE", "score": 0.9}]
    pipe = _make_mock_pipeline(
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        output=output,
    )

    model = TransformersPipeline(pipe)
    result = model(["hello world"])

    assert result.shape == (1, 2)
    np.testing.assert_allclose(result, [[0.0, 0.9]])


def test_call_rescale_to_logits():
    output = [[{"label": "NEGATIVE", "score": 0.2}, {"label": "POSITIVE", "score": 0.8}]]
    pipe = _make_mock_pipeline(
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        output=output,
    )

    model = TransformersPipeline(pipe, rescale_to_logits=True)
    result = model(["some text"])

    expected = [[scipy.special.logit(0.2), scipy.special.logit(0.8)]]
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_call_string_input_raises():
    pipe = _make_mock_pipeline(
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        output=[{"label": "POSITIVE", "score": 0.9}],
    )

    model = TransformersPipeline(pipe)

    with pytest.raises(AssertionError):
        model("just a single string")

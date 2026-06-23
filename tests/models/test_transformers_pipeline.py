"""Tests for shap.models.TransformersPipeline."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import scipy.special

from shap.models import TransformersPipeline


def _make_mock_pipeline(
    label2id=None,
    id2label=None,
    num_labels=None,
    output=None,
):
    """Build a minimal mock that mimics the parts of a transformers pipeline
    that TransformersPipeline.__init__ and __call__ touch.

    Parameters
    ----------
    label2id, id2label, num_labels:
        Values placed on the mock config.
    output:
        If given, the mock pipeline's ``__call__`` will return this value.
    """
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


# __init__ – label mapping resolution


def test_init_with_label2id():
    """Standard case: config has both label2id and id2label."""
    pipe = _make_mock_pipeline(
        label2id={"NEGATIVE": "0", "POSITIVE": "1"},
        id2label={"0": "NEGATIVE", "1": "POSITIVE"},
    )
    m = TransformersPipeline(pipe)
    assert m.label2id == {"NEGATIVE": 0, "POSITIVE": 1}
    assert m.id2label == {0: "NEGATIVE", 1: "POSITIVE"}
    assert m.output_shape == (2,)
    assert m.output_names == ["NEGATIVE", "POSITIVE"]


def test_init_label2id_integer_values():
    """label2id values that are already ints should also work."""
    pipe = _make_mock_pipeline(
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
    )
    m = TransformersPipeline(pipe)
    assert m.label2id == {"NEGATIVE": 0, "POSITIVE": 1}


def test_init_fallback_to_id2label_only():
    """label2id is absent (None) but id2label is present – derive label2id."""
    pipe = _make_mock_pipeline(
        label2id=None,
        id2label={"0": "NEGATIVE", "1": "POSITIVE"},
    )
    m = TransformersPipeline(pipe)
    assert m.label2id == {"NEGATIVE": 0, "POSITIVE": 1}
    assert m.output_shape == (2,)


def test_init_fallback_to_num_labels():
    """Neither label2id nor id2label – derive synthetic mapping from num_labels."""
    pipe = _make_mock_pipeline(label2id=None, id2label=None, num_labels=3)
    m = TransformersPipeline(pipe)
    assert m.label2id == {"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2}
    assert m.output_shape == (3,)
    assert m.output_names == ["LABEL_0", "LABEL_1", "LABEL_2"]


def test_init_no_mapping_raises():
    """All label sources absent – should raise a descriptive ValueError."""
    pipe = _make_mock_pipeline(label2id=None, id2label=None, num_labels=None)
    with pytest.raises(ValueError):
        TransformersPipeline(pipe)


def test_init_no_model_attribute_raises():
    """Pipeline without .model should raise a clear AttributeError."""
    pipe = MagicMock(spec=[])  # spec=[] → no attributes at all
    with pytest.raises(AttributeError):
        TransformersPipeline(pipe)


def test_init_no_config_attribute_raises():
    """pipeline.model without .config should raise a clear AttributeError."""
    model = MagicMock(spec=[])  # no .config
    pipe = MagicMock()
    pipe.model = model
    with pytest.raises(AttributeError):
        TransformersPipeline(pipe)


# __call__ – output shape and score extraction


def test_call_nested_list_output():
    """top_k=None / modern transformers: returns [[{...}, {...}]] per batch."""
    output = [[{"label": "NEGATIVE", "score": 0.1}, {"label": "POSITIVE", "score": 0.9}]]
    pipe = _make_mock_pipeline(
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        output=output,
    )
    m = TransformersPipeline(pipe)
    result = m(["hello world"])
    assert result.shape == (1, 2)
    np.testing.assert_allclose(result, [[0.1, 0.9]])


def test_call_flat_dict_per_sample():
    """top_k=1 returns one dict per sample (not a nested list).

    The ``if not isinstance(val, list): val = [val]`` guard in ``__call__``
    handles this so the score is written to the correct column.
    """
    # One sample, top_k=1: pipeline returns [{"label": "POSITIVE", "score": 0.9}]
    output = [{"label": "POSITIVE", "score": 0.9}]
    pipe = _make_mock_pipeline(
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        output=output,
    )
    m = TransformersPipeline(pipe)
    result = m(["hello world"])
    assert result.shape == (1, 2)
    # Only POSITIVE column is written; NEGATIVE stays 0.0
    np.testing.assert_allclose(result, [[0.0, 0.9]])


def test_call_multi_sample():
    """Multiple samples should produce one row each."""
    output = [
        [{"label": "NEGATIVE", "score": 0.8}, {"label": "POSITIVE", "score": 0.2}],
        [{"label": "NEGATIVE", "score": 0.3}, {"label": "POSITIVE", "score": 0.7}],
    ]
    pipe = _make_mock_pipeline(
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        output=output,
    )
    m = TransformersPipeline(pipe)
    result = m(["bad movie", "great movie"])
    assert result.shape == (2, 2)
    np.testing.assert_allclose(result[0], [0.8, 0.2])
    np.testing.assert_allclose(result[1], [0.3, 0.7])


def test_call_rescale_to_logits():
    """rescale_to_logits=True should logit-transform each score."""
    output = [[{"label": "NEGATIVE", "score": 0.2}, {"label": "POSITIVE", "score": 0.8}]]
    pipe = _make_mock_pipeline(
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        output=output,
    )
    m = TransformersPipeline(pipe, rescale_to_logits=True)
    result = m(["some text"])
    expected = [[scipy.special.logit(0.2), scipy.special.logit(0.8)]]
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_call_string_input_raises():
    """Passing a bare string should raise TypeError (not silently iterate chars)."""
    pipe = _make_mock_pipeline(
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
    )
    m = TransformersPipeline(pipe)
    with pytest.raises(AssertionError):
        m("just a single string")


# Integration-style test using a real transformers pipeline
# (skipped when transformers is not installed or model can't be downloaded)


def test_integration_sentiment_pipeline():
    """Smoke test against a real lightweight sentiment-analysis pipeline.

    Uses ``top_k=None`` (the modern API) to ensure end-to-end compatibility
    with the current transformers release.
    """
    transformers = pytest.importorskip("transformers")
    # distilbert-base-uncased-finetuned-sst-2-english is the default checkpoint
    # for sentiment-analysis; it's small and widely cached in CI environments.
    try:
        classifier = transformers.pipeline(
            "sentiment-analysis",
            top_k=None,
        )
    except Exception as exc:  # pragma: no cover – only fails without network
        pytest.skip(f"Could not load transformers pipeline: {exc}")

    m = TransformersPipeline(classifier)
    result = m(["I love this!", "I hate this!"])

    assert result.shape == (2, 2), f"Unexpected shape: {result.shape}"
    # Scores are probabilities – each row should sum to ~1.
    np.testing.assert_allclose(result.sum(axis=1), [1.0, 1.0], atol=1e-5)
    # Positive text should score higher on the POSITIVE label.
    positive_id = m.label2id.get("POSITIVE", m.label2id.get("positive", None))
    if positive_id is not None:
        assert result[0, positive_id] > result[1, positive_id]

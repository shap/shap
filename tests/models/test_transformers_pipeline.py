"""Tests for TransformersPipeline model wrapper."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.special

import shap


class _DummyConfig:
    def __init__(self):
        self.label2id = {"NEGATIVE": "0", "POSITIVE": "1"}
        self.id2label = {0: "NEGATIVE", 1: "POSITIVE"}


class _DummyModel:
    def __init__(self):
        self.config = _DummyConfig()


class _DummyPipeline:
    def __init__(self, outputs):
        self.model = _DummyModel()
        self._outputs = outputs

    def __call__(self, strings):
        assert isinstance(strings, list)
        return self._outputs


def test_transformers_pipeline_initialization_sets_metadata():
    outputs = [{"label": "NEGATIVE", "score": 0.9}, {"label": "POSITIVE", "score": 0.2}]
    pipeline = _DummyPipeline(outputs)

    wrapper = shap.models.TransformersPipeline(pipeline)

    assert wrapper.output_shape == (2,)
    assert wrapper.output_names == ["NEGATIVE", "POSITIVE"]
    assert wrapper.label2id == {"NEGATIVE": 0, "POSITIVE": 1}


def test_transformers_pipeline_call_handles_dict_and_list_outputs():
    outputs = [
        {"label": "NEGATIVE", "score": 0.8},
        [
            {"label": "NEGATIVE", "score": 0.3},
            {"label": "POSITIVE", "score": 0.7},
        ],
    ]
    pipeline = _DummyPipeline(outputs)
    wrapper = shap.models.TransformersPipeline(pipeline)

    result = wrapper(["a", "b"])

    expected = np.array([[0.8, 0.0], [0.3, 0.7]])
    np.testing.assert_allclose(result, expected)


def test_transformers_pipeline_call_can_rescale_to_logits():
    outputs = [
        [
            {"label": "NEGATIVE", "score": 0.2},
            {"label": "POSITIVE", "score": 0.8},
        ]
    ]
    pipeline = _DummyPipeline(outputs)
    wrapper = shap.models.TransformersPipeline(pipeline, rescale_to_logits=True)

    result = wrapper(["example"])

    expected = np.array([[scipy.special.logit(0.2), scipy.special.logit(0.8)]])
    np.testing.assert_allclose(result, expected)


def test_transformers_pipeline_call_rejects_single_string_input():
    outputs = [{"label": "NEGATIVE", "score": 0.5}]
    pipeline = _DummyPipeline(outputs)
    wrapper = shap.models.TransformersPipeline(pipeline)

    with pytest.raises(AssertionError, match="expects a list of strings"):
        wrapper("single string")

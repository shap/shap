""" This file contains tests for partition explainer.
"""

# pylint: disable=missing-function-docstring
import sys
import pickle
import pytest
import shap
from . import common

@pytest.mark.skipif(sys.platform == 'win32', reason="Integer division bug in HuggingFace on Windows")
def test_translation():
    model, tokenizer, data = common.basic_translation_scenario()
    common.test_additivity(shap.explainers.Partition, model, tokenizer, data)

@pytest.mark.skipif(sys.platform == 'win32', reason="Integer division bug in HuggingFace on Windows")
def test_translation_auto():
    model, tokenizer, data = common.basic_translation_scenario()
    common.test_additivity(shap.Explainer, model, tokenizer, data)

@pytest.mark.skipif(sys.platform == 'win32', reason="Integer division bug in HuggingFace on Windows")
def test_translation_algorithm_arg():
    model, tokenizer, data = common.basic_translation_scenario()
    common.test_additivity(shap.Explainer, model, tokenizer, data, algorithm="partition")

def test_tabular_single_output():
    model, data = common.basic_xgboost_scenario(100)
    common.test_additivity(shap.explainers.Partition, model.predict, shap.maskers.Partition(data), data)

def test_tabular_multi_output():
    model, data = common.basic_xgboost_scenario(100)
    common.test_additivity(shap.explainers.Partition, model.predict_proba, shap.maskers.Partition(data), data)

@pytest.mark.skipif(sys.platform == 'win32', reason="Integer division bug in HuggingFace on Windows")
def test_serialization():
    model, tokenizer, data = common.basic_translation_scenario()
    common.test_serialization(shap.explainers.Partition, model, tokenizer, data)

@pytest.mark.skipif(sys.platform == 'win32', reason="Integer division bug in HuggingFace on Windows")
def test_serialization_no_model_or_masker():
    model, tokenizer, data = common.basic_translation_scenario()
    common.test_serialization(
        shap.explainers.Partition, model, tokenizer, data, model_saver=None, masker_saver=None,
        model_loader=lambda _: model, masker_loader=lambda _: tokenizer
    )

@pytest.mark.skipif(sys.platform == 'win32', reason="Integer division bug in HuggingFace on Windows")
def test_serialization_custom_model_save():
    model, tokenizer, data = common.basic_translation_scenario()
    common.test_serialization(shap.explainers.Partition, model, tokenizer, data, model_saver=pickle.dump, model_loader=pickle.load)

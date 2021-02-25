""" This file contains tests for partition explainer.
"""
import tempfile
import pytest
import numpy as np
import shap

def test_serialization_partition():
    """ This tests the serialization of partition explainers.
    """
    AutoTokenizer = pytest.importorskip("transformers").AutoTokenizer
    AutoModelForSeq2SeqLM = pytest.importorskip("transformers").AutoModelForSeq2SeqLM

    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-es")

    # define the input sentences we want to translate
    data = [
        "In this picture, there are four persons: my father, my mother, my brother and my sister.",
        "Transformers have rapidly become the model of choice for NLP problems, replacing older recurrent neural network models"
    ]

    explainer_original = shap.Explainer(model, tokenizer)
    shap_values_original = explainer_original(data)

    temp_serialization_file = tempfile.TemporaryFile()
    # Serialization
    explainer_original.save(temp_serialization_file)

    temp_serialization_file.seek(0)

    # Deserialization
    explainer_new = shap.Explainer.load(temp_serialization_file)

    temp_serialization_file.close()

    shap_values_new = explainer_new(data)

    assert np.array_equal(shap_values_original[0].base_values, shap_values_new[0].base_values)
    assert np.array_equal(shap_values_original[0].values, shap_values_new[0].values)
    assert isinstance(explainer_original, type(explainer_new))
    assert isinstance(explainer_original.masker, type(explainer_new.masker))

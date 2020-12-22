import numpy as np
import shap 
import shap.benchmark as benchmark 
from transformers import AutoTokenizer
from shap.maskers import Independent, Partition, Impute, Text, Image, FixedComposite

model = lambda x, y: x 
sort_order = 'positive'
perturbation = 'keep'
X = np.random.random((10,13))

def test_init():
    tabular_masker = Independent(X)
    sequential_perturbation = benchmark.perturbation.SequentialPerturbation(model, tabular_masker, sort_order, perturbation)
    assert sequential_perturbation.data_type == "tabular" 

    tabular_masker = Partition(X)
    sequential_perturbation = benchmark.perturbation.SequentialPerturbation(model, tabular_masker, sort_order, perturbation)
    assert sequential_perturbation.data_type == "tabular" 

    tabular_masker = Impute(X)
    sequential_perturbation = benchmark.perturbation.SequentialPerturbation(model, tabular_masker, sort_order, perturbation)
    assert sequential_perturbation.data_type == "tabular" 

    text_masker = Text(AutoTokenizer.from_pretrained("nateraw/bert-base-uncased-emotion",use_fast=True))
    sequential_perturbation = benchmark.perturbation.SequentialPerturbation(model, text_masker, sort_order, perturbation)
    assert sequential_perturbation.data_type == "text" 

    image_masker = Image("inpaint_telea", shape=(224,224,3))
    sequential_perturbation = benchmark.perturbation.SequentialPerturbation(model, image_masker, sort_order, perturbation)
    assert sequential_perturbation.data_type == "image" 

    fc_masker = FixedComposite(text_masker)
    sequential_perturbation = benchmark.perturbation.SequentialPerturbation(model, fc_masker, sort_order, perturbation)
    assert sequential_perturbation.data_type == "text" 

    fc_masker = FixedComposite(image_masker)
    sequential_perturbation = benchmark.perturbation.SequentialPerturbation(model, fc_masker, sort_order, perturbation)
    assert sequential_perturbation.data_type == "image" 

import tempfile
import numpy as np
import pytest
import shap


def basic_xgboost_scenario(max_samples=None, dataset=shap.datasets.adult):
    """ Create a basic XGBoost model on a data set.
    """
    xgboost = pytest.importorskip('xgboost')

    # get a dataset on income prediction
    X, y = dataset()
    if max_samples is not None:
        X = X.iloc[:max_samples]
        y = y[:max_samples]
    X = X.values

    # train an XGBoost model (but any other model type would also work)
    model = xgboost.XGBClassifier()
    model.fit(X, y)

    return model, X

def basic_translation_scenario():
    """ Create a basic transformers translation model and tokenizer.
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

    return model, tokenizer, data

def test_additivity(explainer_type, model, masker, data, **kwargs):
    """ Test explainer and masker for additivity on a single output prediction problem.
    """
    explainer = explainer_type(model, masker, **kwargs)
    shap_values = explainer(data)

    # a multi-output additivity check
    if len(shap_values.shape) == 3:

        # this works with ragged arrays and for models that we can't call directly (they get auto-wrapped)
        for i in range(shap_values.shape[0]):
            row = shap_values[i]
            if callable(explainer.masker.shape):
                all_on_masked = explainer.masker(np.ones(explainer.masker.shape(data[i])[1], dtype=bool), data[i])
            else:
                all_on_masked = explainer.masker(np.ones(explainer.masker.shape[1], dtype=bool), data[i])
            if not isinstance(all_on_masked, tuple):
                all_on_masked = (all_on_masked,)
            out = explainer.model(*all_on_masked)
            assert np.max(np.abs(row.base_values + row.values.sum(0) - out) < 1e6)
    else:
        assert np.max(np.abs(shap_values.base_values + shap_values.values.sum(1) - model(data)) < 1e6)

def test_interactions_additivity(explainer_type, model, masker, data, **kwargs):
    """ Test explainer and masker for additivity on a single output prediction problem.
    """
    explainer = explainer_type(model, masker, **kwargs)
    shap_values = explainer(data, interactions=True)

    assert np.max(np.abs(shap_values.base_values + shap_values.values.sum((1, 2)) - model(data)) < 1e6)

# def test_multi_class(explainer_type, model, masker, data, **kwargs):
#     """ Test explainer and masker for additivity on a multi-class prediction problem.
#     """
#     explainer_kwargs = {k: kwargs[k] for k in kwargs if k in ["algorithm"]}
#     explainer = explainer_type(model.predict_proba, masker, **explainer_kwargs)
#     shap_values = explainer(data)

#     assert np.max(np.abs(shap_values.base_values + shap_values.values.sum(1) - model.predict_proba(data)) < 1e6)

# def test_interactions(explainer_type):
#     """ Check that second order interactions have additivity.
#     """
#     model, X = basic_xgboost(100)

#     # build an Exact explainer and explain the model predictions on the given dataset
#     explainer = explainer_type(model.predict, X)
#     shap_values = explainer(X, interactions=True)

#     assert np.max(np.abs(shap_values.base_values + shap_values.values.sum((1, 2)) - model.predict(X[:100])) < 1e6)

def test_serialization(explainer_type, model, masker, data, rtol=1e-05, atol=1e-8, **kwargs):
    """ Test serialization with a given explainer algorithm.
    """

    explainer_kwargs = {k: kwargs[k] for k in kwargs if k in ["algorithm"]}
    explainer_original = explainer_type(model, masker, **explainer_kwargs)
    shap_values_original = explainer_original(data[:1])

    # Serialization
    temp_serialization_file = tempfile.TemporaryFile()
    save_kwargs = {k: kwargs[k] for k in kwargs if k in ["model_saver", "masker_saver"]}
    explainer_original.save(temp_serialization_file, **save_kwargs)

    # Deserialization
    temp_serialization_file.seek(0)
    load_kwargs = {k: kwargs[k] for k in kwargs if k in ["model_loader", "masker_loader"]}
    explainer_new = explainer_type.load(temp_serialization_file, **load_kwargs)
    temp_serialization_file.close()

    call_kwargs = {k: kwargs[k] for k in kwargs if k in ["max_evals"]}
    shap_values_new = explainer_new(data[:1], **call_kwargs)

    assert np.allclose(shap_values_original.base_values, shap_values_new.base_values, rtol=rtol, atol=atol)
    assert np.allclose(shap_values_original[0].values, shap_values_new[0].values, rtol=rtol, atol=atol)
    assert isinstance(explainer_original, type(explainer_new))
    assert isinstance(explainer_original.masker, type(explainer_new.masker))

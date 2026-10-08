"""Unit tests for the Sampling explainer."""

import numpy as np
import pandas as pd
import pytest

import shap


def test_null_model_small():
    explainer = shap.SamplingExplainer(lambda x: np.zeros(x.shape[0]), np.ones((2, 4)), nsamples=100)
    shap_values = explainer.shap_values(np.ones((1, 4)))
    assert np.sum(np.abs(shap_values)) < 1e-8


def test_null_model_small_pandas_dataframe():
    explainer = shap.SamplingExplainer(lambda x: pd.DataFrame(np.zeros(x.shape[0])), np.ones((2, 4)), nsamples=100)
    shap_values = explainer.shap_values(np.ones((1, 4)))
    assert np.sum(np.abs(shap_values)) < 1e-8


def test_null_model_small_pandas_series():
    explainer = shap.SamplingExplainer(lambda x: pd.Series(np.zeros(x.shape[0])), np.ones((2, 4)), nsamples=100)
    shap_values = explainer.shap_values(np.ones((1, 4)))
    assert np.sum(np.abs(shap_values)) < 1e-8


def test_null_model_small_new():
    explainer = shap.explainers.SamplingExplainer(lambda x: np.zeros(x.shape[0]), np.ones((2, 4)), nsamples=100)
    shap_values = explainer(np.ones((1, 4)))
    assert np.sum(np.abs(shap_values.values)) < 1e-8


def test_null_model():
    explainer = shap.SamplingExplainer(lambda x: np.zeros(x.shape[0]), np.ones((2, 10)), nsamples=100)
    shap_values = explainer.shap_values(np.ones((1, 10)))
    assert np.sum(np.abs(shap_values)) < 1e-8


def test_front_page_model_agnostic():
    sklearn = pytest.importorskip("sklearn")
    train_test_split = pytest.importorskip("sklearn.model_selection").train_test_split

    # print the JS visualization code to the notebook
    shap.initjs()

    # train a SVM classifier
    X_train, X_test, Y_train, _ = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)
    svm = sklearn.svm.SVC(kernel="rbf", probability=True)
    svm.fit(X_train, Y_train)

    # use Kernel SHAP to explain test set predictions
    explainer = shap.SamplingExplainer(svm.predict_proba, X_train, nsamples=100)
    explainer.shap_values(X_test)


def test_additivity_property():
    def model(x):
        return np.sum(x, axis=1)

    background = np.zeros((10, 3))
    explainer = shap.SamplingExplainer(model, background, nsamples=200)

    x = np.array([[1.0, 2.0, 3.0]])
    shap_values = explainer.shap_values(x)

    fx = model(x)[0]
    expected_value = explainer.expected_value

    assert np.isclose(np.sum(shap_values), fx - expected_value, atol=1e-3)


def test_explain_no_varying_features():
    background = np.ones((2, 4))
    instance = np.ones((1, 4))

    explainer = shap.SamplingExplainer(lambda x: np.sum(x, axis=1), background)
    shap_values = explainer.explain(instance)

    assert np.allclose(shap_values, 0.0, atol=1e-8)


def test_explain_one_varying_feature():
    background = np.zeros((2, 4))
    instance = np.array([[5.0, 0.0, 0.0, 0.0]])

    explainer = shap.SamplingExplainer(lambda x: np.sum(x, axis=1), background)
    shap_values = explainer.explain(instance)

    assert np.isclose(shap_values[0], 5.0, atol=1e-3)
    assert np.allclose(shap_values[1:], 0.0, atol=1e-3)


def test_multi_output_stacking():
    def model(x):
        return np.column_stack((np.sum(x, axis=1), 2 * np.sum(x, axis=1)))

    background = np.zeros((2, 3))
    instance = np.array([[1.0, 2.0, 3.0]])

    explainer = shap.SamplingExplainer(model, background)
    explanation = explainer(instance)

    assert explanation.values.shape == (1, 3, 2)

    # Check additivity for first output
    assert np.isclose(np.sum(explanation.values[:, :, 0]), np.sum(instance), atol=1e-3)


def test_sampling_explainer_invalid_link():
    with pytest.raises(ValueError, match="only supports the identity link"):
        shap.SamplingExplainer(lambda x: np.zeros(x.shape[0]), np.ones((2, 4)), link="logit")


def test_call_pandas_dataframe_feature_names():
    df_background = pd.DataFrame({"A": [1, 1], "B": [2, 2], "C": [3, 3]})
    df_instance = pd.DataFrame({"A": [2], "B": [2], "C": [3]})

    explainer = shap.explainers.SamplingExplainer(lambda x: np.sum(x, axis=1), df_background)
    explanation = explainer(df_instance)

    assert explanation.feature_names == ["A", "B", "C"]
    assert explanation.values.shape == (1, 3)

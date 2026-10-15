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


def test_sampling_explainer_with_pandas_input_call():
    """Test __call__ method with pandas DataFrame input returns Explanation object."""
    explainer = shap.SamplingExplainer(lambda x: np.zeros(x.shape[0]), np.ones((5, 4)), nsamples=100)
    X = pd.DataFrame(np.ones((2, 4)), columns=["a", "b", "c", "d"])
    result = explainer(X, nsamples=100)
    assert hasattr(result, "values")
    assert hasattr(result, "base_values")
    assert result.values.shape[0] == 2


def test_sampling_explainer_with_numpy_input_call():
    """Test __call__ method with numpy array input returns Explanation object."""
    explainer = shap.SamplingExplainer(lambda x: np.zeros(x.shape[0]), np.ones((5, 4)), nsamples=100)
    X = np.ones((2, 4))
    result = explainer(X, nsamples=100)
    assert hasattr(result, "values")
    assert result.values.shape[0] == 2


def test_sampling_explainer_single_varying_feature():
    """Test explain() when only one feature varies — phi should capture all effect."""
    background = np.zeros((10, 3))
    # model returns sum of features
    explainer = shap.SamplingExplainer(lambda x: x.sum(axis=1), background, nsamples=100)
    # only first feature varies from background (all zeros)
    x = np.array([[5.0, 0.0, 0.0]])
    shap_values = explainer.shap_values(x, nsamples=100)
    # all effect should be on feature 0
    assert abs(shap_values[0][0] - 5.0) < 1.0


def test_sampling_explainer_no_varying_feature():
    """Test explain() when no features vary — all phi should be zero."""
    background = np.ones((10, 3))
    explainer = shap.SamplingExplainer(lambda x: x.sum(axis=1), background, nsamples=100)
    # input same as background mean — no variation
    x = np.ones((1, 3))
    shap_values = explainer.shap_values(x, nsamples=100)
    assert np.sum(np.abs(shap_values)) < 1e-6


def test_sampling_explainer_invalid_link():
    """Test that non-identity link raises ValueError."""
    with pytest.raises(ValueError, match="identity"):
        shap.SamplingExplainer(
            lambda x: np.zeros(x.shape[0]),
            np.ones((5, 4)),
            link="logit",
        )


def test_sampling_estimate_shape():
    """Test that sampling_estimate returns correct shapes."""
    background = np.random.rand(20, 4)
    explainer = shap.SamplingExplainer(lambda x: x.sum(axis=1), background, nsamples=100)
    x = np.random.rand(4)
    explainer.X_masked = np.zeros((200, 4))
    mean, var = explainer.sampling_estimate(0, lambda x: x.sum(axis=1), x, background, nsamples=10)
    assert mean.shape == var.shape


def test_sampling_explainer_multioutput():
    """Test SamplingExplainer with multi-output model."""
    background = np.zeros((10, 3))

    # model returns 2 outputs
    def multi_output(x):
        return np.column_stack([x.sum(axis=1), x.sum(axis=1) * 2])

    explainer = shap.SamplingExplainer(multi_output, background, nsamples=100)
    x = np.random.rand(1, 3)
    shap_values = explainer.shap_values(x, nsamples=100)
    # shap_values shape is (1, n_features, n_outputs)
    assert shap_values.shape[-1] == 2

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


def test_non_identity_link_raises():
    """Lines 59-60: ValueError when non-identity link is used."""
    with pytest.raises(ValueError, match="identity link"):
        shap.SamplingExplainer(
            lambda x: np.zeros(x.shape[0]),
            np.ones((2, 4)),
            link="logit",
        )


def test_call_with_dataframe():
    """Lines 69-70: __call__ with pandas DataFrame input."""
    explainer = shap.SamplingExplainer(
        lambda x: np.zeros(x.shape[0]),
        np.ones((2, 4)),
        nsamples=100,
    )
    X_df = pd.DataFrame(np.ones((1, 4)), columns=["a", "b", "c", "d"])
    result = explainer(X_df, nsamples=100)
    assert result.feature_names == ["a", "b", "c", "d"]
    assert np.sum(np.abs(result.values)) < 1e-8


def test_call_multioutput():
    """Line 76: __call__ when shap_values returns a list (multi-output)."""
    # SamplingExplainer returns a list when model has multiple outputs
    # and shap_values is called internally — force this by using
    # a model that returns a 2-column output so KernelExplainer
    # produces a list
    background = np.ones((5, 4))

    def multi_output(x):
        out = np.zeros((x.shape[0], 2))
        return out

    explainer = shap.SamplingExplainer(
        multi_output,
        background,
        nsamples=100,
    )
    result = explainer(np.ones((1, 4)), nsamples=100)
    # result.values should have shape (1, 4, 2) for 2 outputs
    assert result.values.ndim == 3


def test_single_varying_feature():
    """Lines 120-124: M==1 path — exactly one feature varies."""
    np.random.seed(42)
    # background is all zeros
    background = np.zeros((5, 3))
    explainer = shap.SamplingExplainer(
        lambda x: x[:, 0],  # only feature 0 matters
        background,
    )
    # only feature 0 differs from background (zeros)
    instance = np.array([[1.0, 0.0, 0.0]])
    result = explainer.explain(instance, nsamples=100)
    # all effect should be on feature 0
    assert result.shape == (3,)
    assert abs(result[0]) > abs(result[1])


def test_nsamples_remainder_distribution():
    """Line 143: remainder loop when round1_samples % (M*2) > 0."""
    np.random.seed(42)
    background = np.random.randn(10, 3)
    explainer = shap.SamplingExplainer(
        lambda x: x[:, 0] + x[:, 1] + x[:, 2],
        background,
    )
    instance = np.random.randn(1, 3)
    # nsamples=101 causes remainder: 101 % (3*2) = 5, so loop runs
    result = explainer.explain(instance, nsamples=101)
    assert result.shape == (3,)


def test_phi_var_zero_spread():
    """Line 156: phi_var += 1 when phi_var.sum() == 0 (null model, large nsamples)."""
    np.random.seed(42)
    background = np.random.randn(10, 3)
    explainer = shap.SamplingExplainer(
        lambda x: np.zeros(x.shape[0]),  # null model — all phi_var will be 0
        background,
    )
    instance = np.random.randn(1, 3)
    # large nsamples triggers round2 where phi_var.sum()==0 check happens
    result = explainer.explain(instance, nsamples=3000)
    assert result.shape == (3,)

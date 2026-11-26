import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shap.plots._embedding import embedding


@pytest.fixture(autouse=True)
def close_plots():
    yield
    plt.close("all")


def test_embedding_pca_basic():
    shap_values = np.random.randn(20, 5)
    embedding(0, shap_values, show=False)


def test_embedding_with_feature_names():
    shap_values = np.random.randn(20, 5)
    feature_names = ["f1", "f2", "f3", "f4", "f5"]
    embedding(0, shap_values, feature_names=feature_names, show=False)


def test_embedding_sum():
    shap_values = np.random.randn(20, 5)
    embedding("sum()", shap_values, show=False)


def test_embedding_custom_array():
    shap_values = np.random.randn(20, 5)
    custom_embedding = np.random.randn(20, 2)
    embedding(0, shap_values, method=custom_embedding, show=False)


def test_embedding_alpha():
    shap_values = np.random.randn(20, 5)
    embedding(0, shap_values, alpha=0.5, show=False)

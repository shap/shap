import numpy as np
import shap


def test_tied_pair():
    beta = np.array([1, 0, 0])
    mu = np.zeros(3)
    Sigma = np.array([[1, 0.999999, 0], [0.999999, 1, 0], [0, 0, 1]])
    X = np.ones((1,3))
    explainer = shap.LinearExplainer((beta, 0), (mu, Sigma))
    assert np.abs(explainer.shap_values(X) - np.array([0.5, 0.5, 0])).max() < 0.05

def test_tied_triple():
    beta = np.array([0, 1, 0, 0])
    mu = 1*np.ones(4)
    Sigma = np.array([[1, 0.999999, 0.999999, 0], [0.999999, 1, 0.999999, 0], [0.999999, 0.999999, 1, 0], [0, 0, 0, 1]])
    X = 2*np.ones((1,4))
    explainer = shap.LinearExplainer((beta, 0), (mu, Sigma))
    assert np.abs(explainer.shap_values(X) - np.array([0.33333, 0.33333, 0.33333, 0])).max() < 0.05

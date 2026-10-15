from unittest.mock import patch

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from shap.benchmark import metrics


# - model generator -
def make_model_generator():
    def model_generator():
        return LinearRegression()

    model_generator.__name__ = "linear_regression"
    return model_generator


model_generator = make_model_generator()
X_zeros = np.zeros((100, 3))


def fake_method(model, X):
    def attr_function(X_test):
        return np.zeros((X_test.shape[0], X.shape[1]))

    return attr_function


# - consistency guarantees tests -
def test_consistency_guarantees_perfect():
    _, val = metrics.consistency_guarantees(None, None, None, "linear_shap_corr")
    assert val == 1.0


def test_consistency_guarantees_none():
    _, val = metrics.consistency_guarantees(None, None, None, "random")
    assert val == 0.0


def test_consistency_guarantees_sampling():
    _, val = metrics.consistency_guarantees(None, None, None, "kernel_shap_1000_meanref")
    assert val == 0.8


def test_consistency_guarantees_deep_shap():
    _, val = metrics.consistency_guarantees(None, None, None, "deep_shap")
    assert val == 0.6


def test_consistency_guarantees_tree_shap():
    _, val = metrics.consistency_guarantees(None, None, None, "tree_shap_tree_path_dependent")
    assert val == 1.0


def test_consistency_guarantees_lime():
    _, val = metrics.consistency_guarantees(None, None, None, "lime_tabular_regression_1000")
    assert val == 0.8


def test_consistency_guarantees_saabas():
    _, val = metrics.consistency_guarantees(None, None, None, "saabas")
    assert val == 0.0


def test_consistency_guarantees_unknown_method():
    with pytest.raises(KeyError):
        metrics.consistency_guarantees(None, None, None, "nonexistent_method")


# - human tests (slow) -
@pytest.mark.xslow
def test_human_and_00():
    with patch("shap.benchmark.metrics.methods") as mock_methods:
        mock_methods.fake = fake_method
        label, (human, attrs) = metrics.human_and_00(X_zeros, None, model_generator, "fake")
        assert label == "human"
        assert len(human) == 3
        assert attrs.shape == (3,)


@pytest.mark.xslow
def test_human_and_01():
    with patch("shap.benchmark.metrics.methods") as mock_methods:
        mock_methods.fake = fake_method
        label, (human, attrs) = metrics.human_and_01(X_zeros, None, model_generator, "fake")
        assert label == "human"
        assert len(human) == 3


@pytest.mark.xslow
def test_human_and_11():
    with patch("shap.benchmark.metrics.methods") as mock_methods:
        mock_methods.fake = fake_method
        label, (human, attrs) = metrics.human_and_11(X_zeros, None, model_generator, "fake")
        assert label == "human"
        assert len(human) == 3


@pytest.mark.xslow
def test_human_or_00():
    with patch("shap.benchmark.metrics.methods") as mock_methods:
        mock_methods.fake = fake_method
        label, (human, attrs) = metrics.human_or_00(X_zeros, None, model_generator, "fake")
        assert label == "human"
        assert len(human) == 3


@pytest.mark.xslow
def test_human_or_01():
    with patch("shap.benchmark.metrics.methods") as mock_methods:
        mock_methods.fake = fake_method
        label, (human, attrs) = metrics.human_or_01(X_zeros, None, model_generator, "fake")
        assert label == "human"
        assert len(human) == 3


@pytest.mark.xslow
def test_human_or_11():
    with patch("shap.benchmark.metrics.methods") as mock_methods:
        mock_methods.fake = fake_method
        label, (human, attrs) = metrics.human_or_11(X_zeros, None, model_generator, "fake")
        assert label == "human"
        assert len(human) == 3


@pytest.mark.xslow
def test_human_xor_00():
    with patch("shap.benchmark.metrics.methods") as mock_methods:
        mock_methods.fake = fake_method
        label, (human, attrs) = metrics.human_xor_00(X_zeros, None, model_generator, "fake")
        assert label == "human"
        assert len(human) == 3


@pytest.mark.xslow
def test_human_xor_01():
    with patch("shap.benchmark.metrics.methods") as mock_methods:
        mock_methods.fake = fake_method
        label, (human, attrs) = metrics.human_xor_01(X_zeros, None, model_generator, "fake")
        assert label == "human"
        assert len(human) == 3


@pytest.mark.xslow
def test_human_xor_11():
    with patch("shap.benchmark.metrics.methods") as mock_methods:
        mock_methods.fake = fake_method
        label, (human, attrs) = metrics.human_xor_11(X_zeros, None, model_generator, "fake")
        assert label == "human"
        assert len(human) == 3


@pytest.mark.xslow
def test_human_sum_00():
    with patch("shap.benchmark.metrics.methods") as mock_methods:
        mock_methods.fake = fake_method
        label, (human, attrs) = metrics.human_sum_00(X_zeros, None, model_generator, "fake")
        assert label == "human"
        assert len(human) == 3


@pytest.mark.xslow
def test_human_sum_01():
    with patch("shap.benchmark.metrics.methods") as mock_methods:
        mock_methods.fake = fake_method
        label, (human, attrs) = metrics.human_sum_01(X_zeros, None, model_generator, "fake")
        assert label == "human"
        assert len(human) == 3


@pytest.mark.xslow
def test_human_sum_11():
    with patch("shap.benchmark.metrics.methods") as mock_methods:
        mock_methods.fake = fake_method
        label, (human, attrs) = metrics.human_sum_11(X_zeros, None, model_generator, "fake")
        assert label == "human"
        assert len(human) == 3

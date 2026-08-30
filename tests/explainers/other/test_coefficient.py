import numpy as np

from shap.explainers.other._coefficient import Coefficient


# Checks that the class accepts a valid model
class DummyModel:
    coef_ = np.array([1, 2, 3])


def test_init_valid_model():
    model = DummyModel()
    explainer = Coefficient(model)
    assert explainer.model is model


class BadModel:
    pass


# Checks that invalid input is rejected
def test_init_invalid_model():
    model = BadModel()
    try:
        Coefficient(model)
        assert False, "Expected AssertionError"
    except AssertionError:
        assert True


# Checks core functionality of attributions method with basic input
def test_attributions_basic():
    class DummyModel:
        coef_ = np.array([1, 2, 3])

    model = DummyModel()
    explainer = Coefficient(model)

    X = np.array([[10, 20, 30], [40, 50, 60]])

    result = explainer.attributions(X)

    expected = np.array([[1, 2, 3], [1, 2, 3]])

    assert np.array_equal(result, expected)


# Ensures output shape is correct
def test_attributions_shape():
    class DummyModel:
        coef_ = np.array([5, 6])

    explainer = Coefficient(DummyModel())

    X = np.zeros((4, 2))  # 4 samples
    result = explainer.attributions(X)

    assert result.shape == (4, 2)


# Tests edge case of empty input
def test_attributions_empty_input():
    class DummyModel:
        coef_ = np.array([1, 2, 3])

    explainer = Coefficient(DummyModel())

    X = np.empty((0, 3))  # zero rows
    result = explainer.attributions(X)

    assert result.shape == (0, 3)

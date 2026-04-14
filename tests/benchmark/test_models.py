import numpy as np
import pytest
import sklearn

# Safely import the module we are testing
from shap.benchmark import models

def test_keras_wrap():
    """Test the KerasWrap class."""
    tf = pytest.importorskip("tensorflow")

    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(2, input_dim=3, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    tf_model.compile(optimizer="adam", loss="mse")

    wrap_flatten = models.KerasWrap(tf_model, epochs=1, flatten_output=True)
    wrap_no_flatten = models.KerasWrap(tf_model, epochs=1, flatten_output=False)

    X = np.random.randn(10, 3)
    y = np.random.randn(10)

    wrap_flatten.fit(X, y, verbose=0)
    assert wrap_flatten.init_weights is not None
    wrap_flatten.fit(X, y, verbose=0)

    preds_flat = wrap_flatten.predict(X)
    assert preds_flat.ndim == 1

    wrap_no_flatten.fit(X, y, verbose=0)
    preds_no_flat = wrap_no_flatten.predict(X)
    assert preds_no_flat.ndim == 2

def test_sklearn_factories():
    """Test standard model instantiations."""
    assert models.corrgroups60__lasso() is not None
    assert models.corrgroups60__ridge() is not None
    assert models.corrgroups60__decision_tree() is not None
    assert models.corrgroups60__random_forest() is not None
    
    assert models.independentlinear60__lasso() is not None
    assert models.independentlinear60__ridge() is not None
    assert models.independentlinear60__decision_tree() is not None
    assert models.independentlinear60__random_forest() is not None

def test_xgboost_factories():
    """Test XGBoost model instantiations."""
    pytest.importorskip("xgboost")
    assert models.corrgroups60__gbm() is not None
    assert models.independentlinear60__gbm() is not None

def test_tensorflow_factories():
    """Test TensorFlow/Keras model instantiations."""
    pytest.importorskip("tensorflow")
    assert models.corrgroups60__ffnn() is not None
    assert models.independentlinear60__ffnn() is not None
    assert models.cric__ffnn() is not None

def test_cric_lambdas():
    """
    Test the cric functions by mocking the underlying predict_proba 
    to ensure the lambda overrides are executed and covered.
    """
    dummy_probs = np.array([[0.1, 0.9], [0.8, 0.2]])
    expected_output = [0.9, 0.2]

    lasso = models.cric__lasso()
    lasso.predict_proba = lambda X: dummy_probs
    assert np.allclose(lasso.predict(None), expected_output)

    ridge = models.cric__ridge()
    ridge.predict_proba = lambda X: dummy_probs
    assert np.allclose(ridge.predict(None), expected_output)

    dt = models.cric__decision_tree()
    dt.predict_proba = lambda X: dummy_probs
    assert np.allclose(dt.predict(None), expected_output)

    rf = models.cric__random_forest()
    rf.predict_proba = lambda X: dummy_probs
    assert np.allclose(rf.predict(None), expected_output)

def test_cric_gbm_lambda():
    """Test the specific XGBoost lambda override."""
    pytest.importorskip("xgboost")
    gbm = models.cric__gbm()
    
    # Mock the original predict method to bypass fitting
    gbm.__orig_predict = lambda X, output_margin: [0.5, -0.5] if output_margin else [1, 0]
    assert gbm.predict(None) == [0.5, -0.5]

def test_human_decision_tree():
    """Test the standalone decision tree function that fits dummy data."""
    model = models.human__decision_tree()
    assert model is not None
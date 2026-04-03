import pytest

try:
    import tensorflow as tf
    from shap.utils._keras import clone_keras_layers, split_keras_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")


def create_simple_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(2)
    ])


def test_clone_keras_layers_basic():
    model = create_simple_model()
    new_model = clone_keras_layers(model, 0, len(model.layers) - 1)

    assert new_model is not None
    assert isinstance(new_model, tf.keras.Model)


def test_split_keras_model_basic():
    model = create_simple_model()
    model1, model2 = split_keras_model(model, 1)

    assert isinstance(model1, tf.keras.Model)
    assert isinstance(model2, tf.keras.Model)


def test_split_keras_model_invalid_layer():
    model = create_simple_model()

    with pytest.raises(Exception):
        split_keras_model(model, 100)
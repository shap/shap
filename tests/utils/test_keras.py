import numpy as np
import pytest

try:
    import tensorflow as tf

    from shap.utils._keras import clone_keras_layers, split_keras_model

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")


def create_simple_model():
    """Create a Keras model with known weights for reproducible testing."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(4,)),
            tf.keras.layers.Dense(8, activation="relu", name="dense1"),
            tf.keras.layers.Dense(2, name="output"),
        ]
    )
    model.build((None, 4))
    return model


def test_clone_keras_layers_output_correctness():
    """Test that cloned layers produce correct output matching original."""
    model = create_simple_model()
    X = np.random.randn(5, 4).astype(np.float32)

    # Get predictions from full model
    full_output = model.predict(X, verbose=0)

    # Clone layers and test
    cloned = clone_keras_layers(model, 1, len(model.layers) - 1)
    cloned_output = cloned.predict(X, verbose=0)

    # Outputs should be identical
    np.testing.assert_allclose(full_output, cloned_output, rtol=1e-5)


def test_split_keras_model_composition():
    """Test that split models recombine correctly: model2(model1(X)) = model(X)."""
    model = create_simple_model()
    X = np.random.randn(5, 4).astype(np.float32)

    # Original prediction
    original_output = model.predict(X, verbose=0)

    # Split model at layer 1
    model1, model2 = split_keras_model(model, 1)

    # Pass through both parts
    intermediate = model1.predict(X, verbose=0)
    recombined_output = model2.predict(intermediate, verbose=0)

    # Should match original
    np.testing.assert_allclose(original_output, recombined_output, rtol=1e-5)


def test_split_keras_model_with_layer_name():
    """Test split_keras_model using layer name instead of index."""
    model = create_simple_model()
    model1, model2 = split_keras_model(model, "dense1")

    assert isinstance(model1, tf.keras.Model)
    assert isinstance(model2, tf.keras.Model)
    assert len(model1.layers) > 0
    assert len(model2.layers) > 0


def test_clone_keras_layers_with_layer_object():
    """Test clone_keras_layers with layer objects instead of indices."""
    model = create_simple_model()
    layer1 = model.layers[1]
    layer2 = model.layers[-1]

    cloned = clone_keras_layers(model, layer1, layer2)
    assert isinstance(cloned, tf.keras.Model)


def test_split_keras_model_invalid_layer():
    """Test error handling for invalid layer index."""
    model = create_simple_model()
    with pytest.raises(Exception):
        split_keras_model(model, 100)

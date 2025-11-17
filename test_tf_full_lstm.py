"""
Test TensorFlow's full LSTM layer (not just LSTMCell)
tf.keras.layers.LSTM handles sequences, not just single timesteps
"""
import numpy as np

try:
    import tensorflow as tf
    import shap

    print("="*80)
    print("TensorFlow Full LSTM Layer Test")
    print("="*80)
    print(f"TensorFlow version: {tf.__version__}\n")

    tf.random.set_seed(42)
    np.random.seed(42)

    # Model dimensions
    sequence_length = 5
    input_size = 3
    hidden_size = 4

    print(f"Sequence length: {sequence_length}")
    print(f"Input size: {input_size}")
    print(f"Hidden size: {hidden_size}\n")

    # Create full LSTM layer (not just LSTMCell!)
    # This is tf.keras.layers.LSTM - the full sequence processor
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(hidden_size, return_sequences=False, input_shape=(sequence_length, input_size))
    ])

    print(f"Model type: {type(model.layers[0])}")
    print(f"Model layer: {model.layers[0].__class__.__name__}")
    print(f"Total parameters: {model.count_params()}\n")

    # Create test data (batch, sequence, features)
    baseline = np.random.randn(1, sequence_length, input_size).astype(np.float32) * 0.1
    test_input = np.random.randn(1, sequence_length, input_size).astype(np.float32)

    # Expected output difference
    output = model(test_input).numpy()
    output_base = model(baseline).numpy()
    expected_diff = (output - output_base).sum()

    print(f"Expected output difference: {expected_diff:.6f}")

    # Test with DeepExplainer
    print("\nTesting with DeepExplainer...")
    e = shap.DeepExplainer(model, baseline)
    shap_values = e.shap_values(test_input, check_additivity=False)

    shap_total = shap_values.sum()
    error = abs(shap_total - expected_diff)

    print(f"SHAP total: {shap_total:.6f}")
    print(f"Additivity error: {error:.6f}")
    print(f"Relative error: {error / (abs(expected_diff) + 1e-10) * 100:.2f}%")

    if error < 0.01:
        print("\n✅ TensorFlow full LSTM layer works perfectly!")
    else:
        print(f"\n⚠️ TensorFlow full LSTM has additivity error: {error:.6f}")

except ImportError as e:
    print(f"TensorFlow not installed: {e}")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

"""
Test TensorFlow native LSTMCell with DeepExplainer
"""
import numpy as np

try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")

    # Set random seed
    tf.random.set_seed(42)
    np.random.seed(42)

    # Model dimensions
    input_size = 3
    hidden_size = 2

    # Create native TensorFlow LSTMCell
    lstm_cell = tf.keras.layers.LSTMCell(hidden_size)

    # Build the cell by calling it once
    x_dummy = tf.constant([[0.0, 0.0, 0.0]], dtype=tf.float32)
    h_dummy = tf.constant([[0.0, 0.0]], dtype=tf.float32)
    c_dummy = tf.constant([[0.0, 0.0]], dtype=tf.float32)
    _ = lstm_cell(x_dummy, states=[h_dummy, c_dummy])

    print(f"\nLSTMCell created with {lstm_cell.count_params()} parameters")

    # Create wrapper using Lambda layer to extract c_new
    class ExtractCNew(tf.keras.layers.Layer):
        def __init__(self, lstm_cell):
            super().__init__()
            self.lstm_cell = lstm_cell

        def call(self, inputs):
            x = inputs[:, :input_size]
            h = inputs[:, input_size:input_size + hidden_size]
            c = inputs[:, input_size + hidden_size:]
            # LSTMCell returns (output, [h_new, c_new])
            output, states = self.lstm_cell(x, states=[h, c])
            return states[1]  # Return c_new

    combined_input = tf.keras.Input(shape=(input_size + 2*hidden_size,))
    new_c = ExtractCNew(lstm_cell)(combined_input)
    model = tf.keras.Model(inputs=combined_input, outputs=new_c)

    print(f"Model output type: {type(model.output)}")
    print(f"Model output shape: {model.output.shape}")

    # Baseline and test inputs
    x_base = np.array([[0.01, 0.02, 0.03]], dtype=np.float32)
    h_base = np.array([[0.0, 0.01]], dtype=np.float32)
    c_base = np.array([[0.1, 0.05]], dtype=np.float32)
    baseline = np.concatenate([x_base, h_base, c_base], axis=1)

    x = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    h = np.array([[0.0, 0.1]], dtype=np.float32)
    c = np.array([[0.5, 0.3]], dtype=np.float32)
    test_input = np.concatenate([x, h, c], axis=1)

    # Get expected output difference
    output = model(test_input).numpy()
    output_base = model(baseline).numpy()
    expected_diff = (output - output_base).sum()

    print(f"\nExpected output difference: {expected_diff:.6f}")

    # Test with DeepExplainer
    print("\nTesting with DeepExplainer...")
    import shap

    e = shap.DeepExplainer(model, baseline)
    shap_values = e.shap_values(test_input, check_additivity=False)

    shap_total = shap_values.sum()
    error = abs(shap_total - expected_diff)

    print(f"SHAP total: {shap_total:.6f}")
    print(f"Additivity error: {error:.6f}")
    print(f"Relative error: {error / (abs(expected_diff) + 1e-10) * 100:.2f}%")

    if error < 0.01:
        print("\n✅ TensorFlow LSTMCell works perfectly with standard gradients!")
        print("No custom handler needed.")
    else:
        print(f"\n⚠️ TensorFlow LSTMCell has additivity error: {error:.6f}")
        print("Custom handler may be needed.")

except ImportError:
    print("TensorFlow not installed, skipping test")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

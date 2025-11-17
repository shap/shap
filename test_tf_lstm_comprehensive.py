"""
Comprehensive test of TensorFlow LSTMCell with various configurations
"""
import numpy as np

try:
    import tensorflow as tf
    import shap

    print("="*80)
    print("TensorFlow LSTM Comprehensive Test")
    print("="*80)
    print(f"TensorFlow version: {tf.__version__}\n")

    def test_lstm_cell(input_size, hidden_size, test_name):
        """Test a specific LSTM configuration"""
        print(f"\n{'='*80}")
        print(f"Test: {test_name}")
        print(f"Input size: {input_size}, Hidden size: {hidden_size}")
        print(f"{'='*80}")

        tf.random.set_seed(42)
        np.random.seed(42)

        # Create LSTMCell
        lstm_cell = tf.keras.layers.LSTMCell(hidden_size)

        # Build the cell
        x_dummy = tf.constant([[0.0] * input_size], dtype=tf.float32)
        h_dummy = tf.constant([[0.0] * hidden_size], dtype=tf.float32)
        c_dummy = tf.constant([[0.0] * hidden_size], dtype=tf.float32)
        _ = lstm_cell(x_dummy, states=[h_dummy, c_dummy])

        # Create wrapper
        class ExtractCNew(tf.keras.layers.Layer):
            def __init__(self, lstm_cell, input_size, hidden_size):
                super().__init__()
                self.lstm_cell = lstm_cell
                self.input_size = input_size
                self.hidden_size = hidden_size

            def call(self, inputs):
                x = inputs[:, :self.input_size]
                h = inputs[:, self.input_size:self.input_size + self.hidden_size]
                c = inputs[:, self.input_size + self.hidden_size:]
                output, states = self.lstm_cell(x, states=[h, c])
                return states[1]  # c_new

        combined_input = tf.keras.Input(shape=(input_size + 2*hidden_size,))
        new_c = ExtractCNew(lstm_cell, input_size, hidden_size)(combined_input)
        model = tf.keras.Model(inputs=combined_input, outputs=new_c)

        # Create test data
        x_base = np.random.randn(1, input_size).astype(np.float32) * 0.1
        h_base = np.random.randn(1, hidden_size).astype(np.float32) * 0.1
        c_base = np.random.randn(1, hidden_size).astype(np.float32) * 0.1
        baseline = np.concatenate([x_base, h_base, c_base], axis=1)

        x = np.random.randn(1, input_size).astype(np.float32)
        h = np.random.randn(1, hidden_size).astype(np.float32)
        c = np.random.randn(1, hidden_size).astype(np.float32)
        test_input = np.concatenate([x, h, c], axis=1)

        # Expected difference
        output = model(test_input).numpy()
        output_base = model(baseline).numpy()
        expected_diff = (output - output_base).sum()

        # SHAP values
        e = shap.DeepExplainer(model, baseline)
        shap_values = e.shap_values(test_input, check_additivity=False)

        shap_total = shap_values.sum()
        error = abs(shap_total - expected_diff)
        rel_error = error / (abs(expected_diff) + 1e-10) * 100

        print(f"Expected output diff: {expected_diff:.6f}")
        print(f"SHAP total: {shap_total:.6f}")
        print(f"Additivity error: {error:.6f}")
        print(f"Relative error: {rel_error:.2f}%")

        if error < 0.01:
            print("✅ PASSED")
            return True
        else:
            print(f"❌ FAILED (error = {error:.6f})")
            return False

    # Test various configurations
    results = []

    # Test 1: Small LSTM
    results.append(test_lstm_cell(3, 2, "Small LSTM (3x2)"))

    # Test 2: Medium LSTM
    results.append(test_lstm_cell(10, 20, "Medium LSTM (10x20)"))

    # Test 3: Large hidden size
    results.append(test_lstm_cell(5, 50, "Large hidden LSTM (5x50)"))

    # Test 4: Large input size
    results.append(test_lstm_cell(100, 10, "Large input LSTM (100x10)"))

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Tests passed: {sum(results)}/{len(results)}")

    if all(results):
        print("\n✅ All tests PASSED - TensorFlow LSTMCell works perfectly!")
    else:
        print(f"\n❌ Some tests FAILED")

except ImportError as e:
    print(f"TensorFlow not installed: {e}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

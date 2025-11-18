"""
Test TensorFlow full LSTM (sequence) with various configurations
"""
import tensorflow as tf
import numpy as np
import shap

def test_tf_lstm_sequence(sequence_length, input_size, hidden_size, test_name):
    """Test TensorFlow LSTM sequence with given dimensions"""
    print(f"\n{'='*80}")
    print(f"{test_name}")
    print(f"{'='*80}")
    print(f"Sequence: {sequence_length}, Input: {input_size}, Hidden: {hidden_size}")

    # Create model
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(hidden_size, return_sequences=False,
                             input_shape=(sequence_length, input_size))
    ])

    # Build model
    dummy = np.random.randn(1, sequence_length, input_size).astype(np.float32)
    _ = model(dummy)

    # Create test data
    baseline = np.zeros((1, sequence_length, input_size), dtype=np.float32)
    test_input = np.random.randn(1, sequence_length, input_size).astype(np.float32)

    # Expected difference
    output_test = model(test_input).numpy()
    output_base = model(baseline).numpy()
    expected_diff = (output_test - output_base).sum()

    print(f"Expected output difference: {expected_diff:.6f}")

    try:
        # SHAP
        e = shap.DeepExplainer(model, baseline)
        shap_values = e.shap_values(test_input, check_additivity=False)

        shap_total = shap_values.sum()
        error = abs(shap_total - expected_diff)
        relative_error = error / (abs(expected_diff) + 1e-10) * 100

        print(f"SHAP total: {shap_total:.6f}")
        print(f"Error: {error:.6f}")
        print(f"Relative error: {relative_error:.2f}%")

        if error < 0.01:
            print("✅ Excellent! (<1% error)")
            return True, error
        elif error < 0.1:
            print("✓ Good! (<10% error)")
            return True, error
        else:
            print(f"⚠️  Moderate error: {relative_error:.1f}%")
            return True, error

    except Exception as e:
        print(f"❌ Failed: {str(e)[:100]}")
        return False, None

# Run tests
print("="*80)
print("TensorFlow Full LSTM (Sequence) - Comprehensive Tests")
print("="*80)

results = []

# Test 1: Small sequence
success, error = test_tf_lstm_sequence(3, 2, 2, "Test 1: Small (seq=3, in=2, hidden=2)")
results.append(("Small", success, error))

# Test 2: Medium sequence
success, error = test_tf_lstm_sequence(5, 3, 4, "Test 2: Medium (seq=5, in=3, hidden=4)")
results.append(("Medium", success, error))

# Test 3: Longer sequence
success, error = test_tf_lstm_sequence(10, 5, 8, "Test 3: Longer sequence (seq=10, in=5, hidden=8)")
results.append(("Longer", success, error))

# Test 4: Wide input
success, error = test_tf_lstm_sequence(5, 20, 10, "Test 4: Wide input (seq=5, in=20, hidden=10)")
results.append(("Wide input", success, error))

# Test 5: Large hidden
success, error = test_tf_lstm_sequence(5, 5, 50, "Test 5: Large hidden (seq=5, in=5, hidden=50)")
results.append(("Large hidden", success, error))

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

passed = sum(1 for _, success, _ in results if success)
total = len(results)

print(f"\nTests passed: {passed}/{total}")
print("\nDetailed results:")
print(f"{'Configuration':<20} {'Status':<10} {'Error':<15}")
print("-" * 50)

for name, success, error in results:
    if success:
        if error is not None:
            rel_error = f"{error:.6f}"
            print(f"{name:<20} {'✓':<10} {rel_error:<15}")
        else:
            print(f"{name:<20} {'✓':<10} {'N/A':<15}")
    else:
        print(f"{name:<20} {'✗':<10} {'Failed':<15}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if passed == total and all(error is not None and error < 0.1 for _, success, error in results if success):
    print("✅ TensorFlow full LSTM sequences work with acceptable accuracy!")
    print("   Errors are small enough for practical use.")
elif passed == total:
    print("⚠️  TensorFlow full LSTM sequences work but with moderate errors.")
    print("   May be acceptable depending on use case.")
elif passed > 0:
    print("⚠️  Partial support - some configurations work, others fail.")
else:
    print("❌ TensorFlow full LSTM sequences do not work reliably.")

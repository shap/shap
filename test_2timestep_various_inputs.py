"""
Test 2-timestep LSTM with various input magnitudes
"""
import tensorflow as tf
import numpy as np
import shap

def test_with_magnitude(magnitude, test_name):
    """Test with given input magnitude"""
    print(f"\n{'='*80}")
    print(f"{test_name}")
    print(f"{'='*80}")

    # Create 2-timestep LSTM
    sequence_length = 2
    input_size = 3
    hidden_size = 4

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(hidden_size, return_sequences=False,
                             input_shape=(sequence_length, input_size))
    ])

    dummy = np.random.randn(1, sequence_length, input_size).astype(np.float32)
    _ = model(dummy)

    # Create inputs with specific magnitude
    baseline = np.zeros((1, sequence_length, input_size), dtype=np.float32)
    test_input = np.random.randn(1, sequence_length, input_size).astype(np.float32) * magnitude

    print(f"Input magnitude: {magnitude}")
    print(f"Test input range: [{test_input.min():.3f}, {test_input.max():.3f}]")

    # Expected difference
    output_test = model(test_input).numpy()
    output_base = model(baseline).numpy()
    expected_diff = (output_test - output_base).sum()

    print(f"Expected: {expected_diff:.6f}")

    # SHAP
    try:
        explainer = shap.DeepExplainer(model, baseline)
        shap_values = explainer.shap_values(test_input, check_additivity=False)

        shap_total = shap_values.sum()
        abs_error = abs(shap_total - expected_diff)
        rel_error = abs_error / (abs(expected_diff) + 1e-10) * 100

        print(f"SHAP total: {shap_total:.6f}")
        print(f"Absolute error: {abs_error:.6f}")
        print(f"Relative error: {rel_error:.2f}%")

        return True, abs_error, rel_error

    except Exception as e:
        print(f"❌ Error: {str(e)[:100]}")
        return False, None, None

print("="*80)
print("Testing 2-Timestep LSTM with Various Input Magnitudes")
print("="*80)

results = []

# Test with different magnitudes
magnitudes = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]

for mag in magnitudes:
    success, abs_err, rel_err = test_with_magnitude(mag, f"Magnitude: {mag}")
    results.append((mag, success, abs_err, rel_err))

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\n{'Magnitude':<15} {'Status':<10} {'Abs Error':<15} {'Rel Error %':<15}")
print("-" * 60)

for mag, success, abs_err, rel_err in results:
    if success:
        print(f"{mag:<15} {'✓':<10} {abs_err:<15.6f} {rel_err:<15.2f}")
    else:
        print(f"{mag:<15} {'✗':<10} {'N/A':<15} {'N/A':<15}")

# Check if errors increase with magnitude
if all(success for _, success, _, _ in results):
    rel_errors = [rel_err for _, success, _, rel_err in results if success]
    print(f"\nRelative error range: {min(rel_errors):.2f}% - {max(rel_errors):.2f}%")

    if max(rel_errors) < 1.0:
        print("✅ All tests have <1% error - excellent!")
    elif max(rel_errors) < 5.0:
        print("✓ All tests have <5% error - good!")
    else:
        print(f"⚠️  Some tests have significant error (up to {max(rel_errors):.2f}%)")

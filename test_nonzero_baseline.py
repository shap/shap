"""
Test 2-timestep LSTM with non-zero baseline
Hypothesis: Zero baseline causes numerical issues in rescale rule
"""
import tensorflow as tf
import numpy as np
import shap

def test_with_baseline(baseline_type, test_name):
    """Test with different baseline types"""
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

    # Set random seed for reproducibility
    np.random.seed(42)

    # Create test input
    test_input = np.random.randn(1, sequence_length, input_size).astype(np.float32) * 0.5

    # Create different baselines
    if baseline_type == "zeros":
        baseline = np.zeros((1, sequence_length, input_size), dtype=np.float32)
    elif baseline_type == "small":
        baseline = np.random.randn(1, sequence_length, input_size).astype(np.float32) * 0.01
    elif baseline_type == "medium":
        baseline = np.random.randn(1, sequence_length, input_size).astype(np.float32) * 0.1
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")

    print(f"Baseline type: {baseline_type}")
    print(f"Baseline range: [{baseline.min():.6f}, {baseline.max():.6f}]")
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
print("Testing 2-Timestep LSTM with Different Baselines")
print("="*80)
print("\nHypothesis: Zero baseline causes numerical issues in DeepLift rescale rule")
print("when hidden states from baseline pass forward become near-zero denominators\n")

results = []

# Test with different baselines
baseline_types = ["zeros", "small", "medium"]

for btype in baseline_types:
    success, abs_err, rel_err = test_with_baseline(btype, f"Baseline: {btype}")
    results.append((btype, success, abs_err, rel_err))

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\n{'Baseline':<15} {'Status':<10} {'Abs Error':<15} {'Rel Error %':<15}")
print("-" * 60)

for btype, success, abs_err, rel_err in results:
    if success:
        print(f"{btype:<15} {'✓':<10} {abs_err:<15.6f} {rel_err:<15.2f}")
    else:
        print(f"{btype:<15} {'✗':<10} {'N/A':<15} {'N/A':<15}")

# Analysis
if all(success for _, success, _, _ in results):
    rel_errors = [rel_err for _, success, _, rel_err in results if success]
    print(f"\nRelative error range: {min(rel_errors):.2f}% - {max(rel_errors):.2f}%")

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    # Check if non-zero baseline helps
    zero_error = results[0][2]  # zeros baseline
    nonzero_errors = [err for btype, _, err, _ in results[1:] if _ and err is not None]

    if nonzero_errors:
        avg_nonzero = sum(nonzero_errors) / len(nonzero_errors)
        print(f"\nZero baseline error: {results[0][3]:.2f}%")
        print(f"Non-zero baseline avg error: {sum([r[3] for r in results[1:] if r[1]])/len(results[1:]):.2f}%")

        if avg_nonzero < zero_error:
            print("\n✓ Non-zero baseline DOES help reduce error!")
        else:
            print("\n✗ Non-zero baseline does NOT help - issue is elsewhere")

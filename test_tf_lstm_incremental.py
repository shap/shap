"""
Incremental TensorFlow LSTM Sequence Testing
Test sequences of increasing length to identify where errors start
"""
import tensorflow as tf
import numpy as np
import shap

def test_sequence_length(seq_len, input_size=3, hidden_size=4):
    """Test TensorFlow LSTM with specific sequence length"""
    print(f"\n{'='*80}")
    print(f"Testing Sequence Length: {seq_len}")
    print(f"{'='*80}")

    # Create model
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(hidden_size, return_sequences=False,
                             input_shape=(seq_len, input_size))
    ])

    # Build
    dummy = np.random.randn(1, seq_len, input_size).astype(np.float32)
    _ = model(dummy)

    # Test data - use small values to avoid saturation
    np.random.seed(42)  # Reproducible
    baseline = np.zeros((1, seq_len, input_size), dtype=np.float32)
    test_input = np.random.randn(1, seq_len, input_size).astype(np.float32) * 0.1  # Small values

    # Expected
    output_test = model(test_input).numpy()
    output_base = model(baseline).numpy()
    expected_diff = (output_test - output_base).sum()

    print(f"Expected difference: {expected_diff:.6f}")

    try:
        # SHAP
        e = shap.DeepExplainer(model, baseline)
        shap_values = e.shap_values(test_input, check_additivity=False)

        shap_total = shap_values.sum()
        error = abs(shap_total - expected_diff)
        relative_error = error / (abs(expected_diff) + 1e-10) * 100

        print(f"SHAP total: {shap_total:.6f}")
        print(f"Absolute error: {error:.6f}")
        print(f"Relative error: {relative_error:.2f}%")

        # Classification
        if error < 0.001:
            status = "✅ PERFECT"
            print(status)
        elif error < 0.01:
            status = "✅ Excellent"
            print(status)
        elif relative_error < 5:
            status = "✓ Very good"
            print(status)
        elif relative_error < 10:
            status = "✓ Good"
            print(status)
        elif relative_error < 20:
            status = "⚠️ Acceptable"
            print(status)
        else:
            status = "❌ Poor"
            print(status)

        return True, error, relative_error, status

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, "FAILED"

print("="*80)
print("TensorFlow LSTM - Incremental Sequence Length Testing")
print("="*80)
print("\nStrategy: Start with 1 timestep, increase gradually")
print("Goal: Identify at what length errors become significant\n")

# Test incrementally
results = []

for seq_len in [1, 2, 3, 4, 5, 7, 10, 15, 20]:
    success, abs_error, rel_error, status = test_sequence_length(seq_len)
    results.append({
        'seq_len': seq_len,
        'success': success,
        'abs_error': abs_error,
        'rel_error': rel_error,
        'status': status
    })

# Summary
print("\n" + "="*80)
print("SUMMARY - Error Evolution")
print("="*80)
print(f"\n{'Seq Len':<10} {'Abs Error':<15} {'Rel Error %':<15} {'Status':<20}")
print("-" * 65)

for r in results:
    if r['success']:
        abs_str = f"{r['abs_error']:.6f}" if r['abs_error'] is not None else "N/A"
        rel_str = f"{r['rel_error']:.2f}" if r['rel_error'] is not None else "N/A"
        print(f"{r['seq_len']:<10} {abs_str:<15} {rel_str:<15} {r['status']:<20}")
    else:
        print(f"{r['seq_len']:<10} {'FAILED':<15} {'FAILED':<15} {'FAILED':<20}")

# Analysis
print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

successful = [r for r in results if r['success']]
if successful:
    # Find where errors start to become significant
    perfect = [r for r in successful if r['abs_error'] is not None and r['abs_error'] < 0.001]
    good = [r for r in successful if r['abs_error'] is not None and r['rel_error'] < 10]
    acceptable = [r for r in successful if r['abs_error'] is not None and r['rel_error'] < 20]

    print(f"\nPerfect (<0.1% error): {len(perfect)} configurations")
    if perfect:
        print(f"  Max seq length: {max(r['seq_len'] for r in perfect)}")

    print(f"\nGood (<10% rel error): {len(good)} configurations")
    if good:
        print(f"  Max seq length: {max(r['seq_len'] for r in good)}")

    print(f"\nAcceptable (<20% rel error): {len(acceptable)} configurations")
    if acceptable:
        print(f"  Max seq length: {max(r['seq_len'] for r in acceptable)}")

    # Show error trend
    print("\nError trend:")
    for r in successful:
        if r['abs_error'] is not None:
            bars = int(r['rel_error'] / 5)  # Each bar = 5%
            bar_str = '█' * min(bars, 40)
            print(f"  Seq {r['seq_len']:2d}: {bar_str} {r['rel_error']:.1f}%")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)

# Identify the problematic transition
successful_with_errors = [(r['seq_len'], r['rel_error'])
                          for r in successful if r['rel_error'] is not None]

if successful_with_errors:
    # Find where error jumps significantly
    print("\nWhere do errors start to grow?")
    for i in range(len(successful_with_errors) - 1):
        seq1, err1 = successful_with_errors[i]
        seq2, err2 = successful_with_errors[i + 1]
        delta = err2 - err1

        if delta > 5:  # Significant jump
            print(f"  ⚠️ Jump from seq={seq1} ({err1:.1f}%) to seq={seq2} ({err2:.1f}%)")
            print(f"     Δ = {delta:.1f}% - investigate this transition!")

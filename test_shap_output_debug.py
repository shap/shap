"""
Debug: Check if SHAP values are computed but all zero, or not computed at all
"""
import numpy as np
import tensorflow as tf
import shap

print("="*80)
print("Debugging SHAP Values Output")
print("="*80)

tf.random.set_seed(42)
np.random.seed(42)

# Small LSTM for easier debugging
sequence_length = 2
input_size = 2
hidden_size = 2

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(hidden_size, return_sequences=False, input_shape=(sequence_length, input_size))
])

# Create inputs
baseline = np.zeros((1, sequence_length, input_size), dtype=np.float32)
test_input = np.ones((1, sequence_length, input_size), dtype=np.float32)

# Expected difference
output = model(test_input).numpy()
output_base = model(baseline).numpy()
expected_diff = (output - output_base).sum()

print(f"Test input shape: {test_input.shape}")
print(f"Baseline shape: {baseline.shape}")
print(f"Output: {output}")
print(f"Baseline output: {output_base}")
print(f"Expected diff: {expected_diff:.6f}")

# Get SHAP values
print("\nComputing SHAP values...")
e = shap.DeepExplainer(model, baseline)
shap_values = e.shap_values(test_input, check_additivity=False)

print(f"\nSHAP values type: {type(shap_values)}")
print(f"SHAP values shape: {shap_values.shape if hasattr(shap_values, 'shape') else 'N/A'}")
print(f"SHAP values dtype: {shap_values.dtype if hasattr(shap_values, 'dtype') else 'N/A'}")
print(f"\nSHAP values array:")
print(shap_values)
print(f"\nSHAP total: {shap_values.sum():.6f}")
print(f"Expected: {expected_diff:.6f}")
print(f"Error: {abs(shap_values.sum() - expected_diff):.6f}")

# Check if any values are non-zero
nonzero_count = np.count_nonzero(shap_values)
print(f"\nNon-zero SHAP values: {nonzero_count}/{shap_values.size}")

if nonzero_count == 0:
    print("\n⚠️ ALL SHAP VALUES ARE EXACTLY ZERO!")
    print("This means gradients are either:")
    print("1. Not being computed")
    print("2. Being computed as zero")
    print("3. Not being returned properly")
else:
    print(f"\n✓ {nonzero_count} non-zero values found")
    print(f"Mean absolute SHAP value: {np.abs(shap_values).mean():.6f}")

"""
Debug 3-timestep LSTM - the next progression
"""
import tensorflow as tf
import numpy as np
import shap

print("="*80)
print("Debugging 3-Timestep LSTM")
print("="*80)

# Create 3-timestep LSTM
seq_len = 3
input_size = 3
hidden_size = 4

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(hidden_size, return_sequences=False,
                         input_shape=(seq_len, input_size))
])

dummy = np.random.randn(1, seq_len, input_size).astype(np.float32)
_ = model(dummy)

# Use moderate values
np.random.seed(123)
baseline = np.zeros((1, seq_len, input_size), dtype=np.float32)
test_input = np.random.randn(1, seq_len, input_size).astype(np.float32) * 0.5

# Expected
output_test = model(test_input).numpy()
output_base = model(baseline).numpy()
expected_diff = (output_test - output_base).sum()

print(f"\nTest input shape: {test_input.shape}")
print(f"Expected difference: {expected_diff:.6f}")
print()

# SHAP
e = shap.DeepExplainer(model, baseline)
shap_values = e.shap_values(test_input, check_additivity=False)

shap_total = shap_values.sum()
error = abs(shap_total - expected_diff)
relative_error = error / (abs(expected_diff) + 1e-10) * 100

print(f"SHAP total: {shap_total:.6f}")
print(f"Absolute error: {error:.6f}")
print(f"Relative error: {relative_error:.2f}%")
print()

# Analyze per timestep
shap_array = np.array(shap_values)
print(f"SHAP values shape: {shap_array.shape}")

for t in range(seq_len):
    timestep_shap = shap_array[0, t, :].sum()
    print(f"Timestep {t}: SHAP sum = {timestep_shap:.6f}")

print()

# Diagnosis
print("="*80)
print("DIAGNOSIS")
print("="*80)

if error < 0.01:
    print("✅ Excellent! Error < 1%")
elif relative_error < 5:
    print("✅ Very good! Error < 5%")
elif relative_error < 10:
    print("✓ Good! Error < 10%")
else:
    print(f"⚠️ Error: {relative_error:.1f}%")

# Compare to 2-timestep
print(f"\nComparison:")
print(f"  2 timesteps: ~2.87% error (from previous test)")
print(f"  3 timesteps: {relative_error:.2f}% error")

if relative_error > 5:
    print(f"\n⚠️ Error increased significantly from 2 to 3 timesteps!")
    print(f"   This suggests error is compounding through recurrence.")
else:
    print(f"\n✅ Error remains low - implementation working well!")

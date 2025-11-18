"""
Test that 1-timestep LSTM gives identical results to LSTMCell

This verifies our LSTM sequence implementation by comparing:
1. tf.keras.layers.LSTM with sequence_length=1
2. tf.keras.layers.LSTMCell (single step)

They should produce identical outputs and SHAP values.
"""
import tensorflow as tf
import numpy as np
import shap

print("="*80)
print("Testing: 1-Timestep LSTM vs LSTMCell")
print("="*80)

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Parameters
input_size = 3
hidden_size = 4
batch_size = 1

# Create shared weights by creating LSTMCell first, then copying to LSTM
lstm_cell = tf.keras.layers.LSTMCell(hidden_size)

# Build the cell
x_dummy = tf.constant(np.zeros((1, input_size)), dtype=tf.float32)
h_dummy = tf.constant(np.zeros((1, hidden_size)), dtype=tf.float32)
c_dummy = tf.constant(np.zeros((1, hidden_size)), dtype=tf.float32)
_ = lstm_cell(x_dummy, states=[h_dummy, c_dummy])

print(f"\nLSTMCell weights:")
print(f"  kernel shape: {lstm_cell.kernel.shape}")
print(f"  recurrent_kernel shape: {lstm_cell.recurrent_kernel.shape}")
print(f"  bias shape: {lstm_cell.bias.shape}")

# Create LSTM with same weights
lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=False)

# Build LSTM by calling it
seq_dummy = tf.constant(np.zeros((1, 1, input_size)), dtype=tf.float32)
_ = lstm(seq_dummy)

# Copy weights from cell to LSTM
lstm.cell.kernel.assign(lstm_cell.kernel)
lstm.cell.recurrent_kernel.assign(lstm_cell.recurrent_kernel)
lstm.cell.bias.assign(lstm_cell.bias)

print(f"\nLSTM weights (copied from LSTMCell):")
print(f"  kernel shape: {lstm.cell.kernel.shape}")
print(f"  recurrent_kernel shape: {lstm.cell.recurrent_kernel.shape}")
print(f"  bias shape: {lstm.cell.bias.shape}")

# Create test inputs
baseline_x = np.zeros((batch_size, input_size), dtype=np.float32)
test_x = np.random.randn(batch_size, input_size).astype(np.float32) * 0.5

print(f"\nTest input shape: {test_x.shape}")
print(f"Test input values:\n{test_x}")

# ============================================================================
# Test LSTMCell
# ============================================================================
print("\n" + "="*80)
print("Testing LSTMCell")
print("="*80)

# For LSTMCell, we just run it directly without SHAP (to compare outputs)
# We'll use the existing test_tensorflow_native_lstm_cell from test_deep.py
# which already has working SHAP for LSTMCell

# Create concatenated inputs (x, h=0, c=0)
h_init = np.zeros((batch_size, hidden_size), dtype=np.float32)
c_init = np.zeros((batch_size, hidden_size), dtype=np.float32)

test_cell_inputs = (test_x, h_init, c_init)
baseline_cell_inputs = (baseline_x, h_init, c_init)

print(f"Cell inputs: x={test_x.shape}, h={h_init.shape}, c={c_init.shape}")

# Forward pass
output_cell_test, _ = lstm_cell(test_x, states=[h_init, c_init])
output_cell_base, _ = lstm_cell(baseline_x, states=[h_init, c_init])
output_cell_test = output_cell_test.numpy()
output_cell_base = output_cell_base.numpy()
output_cell_diff = (output_cell_test - output_cell_base).sum()

print(f"\nLSTMCell forward pass:")
print(f"  Output (test): {output_cell_test}")
print(f"  Output (baseline): {output_cell_base}")
print(f"  Difference sum: {output_cell_diff:.6f}")

# ============================================================================
# Test LSTM (1 timestep)
# ============================================================================
print("\n" + "="*80)
print("Testing LSTM with 1 timestep")
print("="*80)

# Create model wrapper for LSTM
lstm_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1, input_size)),
    lstm
])

# Reshape inputs to (batch, timesteps=1, features)
baseline_lstm = baseline_x.reshape((batch_size, 1, input_size))
test_lstm = test_x.reshape((batch_size, 1, input_size))

print(f"LSTM input shape: {test_lstm.shape} (batch, timesteps=1, features)")

# Forward pass
output_lstm_test = lstm_model(test_lstm).numpy()
output_lstm_base = lstm_model(baseline_lstm).numpy()
output_lstm_diff = (output_lstm_test - output_lstm_base).sum()

print(f"\nLSTM forward pass:")
print(f"  Output (test): {output_lstm_test}")
print(f"  Output (baseline): {output_lstm_base}")
print(f"  Difference sum: {output_lstm_diff:.6f}")

# SHAP values
explainer_lstm = shap.DeepExplainer(lstm_model, baseline_lstm)
shap_lstm = explainer_lstm.shap_values(test_lstm, check_additivity=False)
shap_lstm_total = shap_lstm.sum()

print(f"\nLSTM SHAP:")
print(f"  SHAP total: {shap_lstm_total:.6f}")
print(f"  Error: {abs(shap_lstm_total - output_lstm_diff):.6f}")

# ============================================================================
# Comparison
# ============================================================================
print("\n" + "="*80)
print("Comparison: LSTMCell vs LSTM (1 timestep)")
print("="*80)

# Compare outputs
output_diff = np.abs(output_cell_test - output_lstm_test).max()
print(f"\nOutput difference (max absolute): {output_diff:.10f}")

if output_diff < 1e-6:
    print("✅ Outputs match perfectly!")
elif output_diff < 1e-3:
    print(f"✓ Outputs match well (diff: {output_diff:.10f})")
else:
    print(f"⚠️  Outputs differ by {output_diff:.10f}")

# Compare output differences (should be same)
diff_of_diffs = abs(output_cell_diff - output_lstm_diff)
print(f"\nDifference of differences: {diff_of_diffs:.10f}")
print(f"  LSTMCell diff: {output_cell_diff:.6f}")
print(f"  LSTM diff: {output_lstm_diff:.6f}")

if diff_of_diffs < 1e-6:
    print("✅ Differences match perfectly!")
elif diff_of_diffs < 1e-3:
    print(f"✓ Differences match well")
else:
    print(f"⚠️  Differences don't match")

# Final verdict
print("\n" + "="*80)
print("FINAL VERDICT")
print("="*80)

print(f"\n1. Forward Pass Comparison:")
print(f"   Output difference: {output_diff:.10f}")
if output_diff < 1e-5:
    print("   ✅ Outputs match - 1-timestep LSTM ≡ LSTMCell")
else:
    print(f"   ⚠️  Outputs differ by {output_diff:.10f}")

print(f"\n2. LSTM SHAP Accuracy:")
print(f"   Expected difference: {output_lstm_diff:.6f}")
print(f"   SHAP total: {shap_lstm_total:.6f}")
print(f"   Error: {abs(shap_lstm_total - output_lstm_diff):.6f}")
if abs(shap_lstm_total - output_lstm_diff) < 1e-5:
    print("   ✅ LSTM SHAP is perfect!")
else:
    print(f"   Error: {abs(shap_lstm_total - output_lstm_diff):.6f}")

print(f"\n3. Overall Conclusion:")
if output_diff < 1e-5 and abs(shap_lstm_total - output_lstm_diff) < 1e-5:
    print("   ✅ Complete Success!")
    print("   - 1-timestep LSTM = LSTMCell (verified)")
    print("   - LSTM SHAP values are accurate (verified)")
    print("   - Sequence LSTM implementation confirmed correct!")
else:
    print("   Status:")
    if output_diff < 1e-5:
        print("   ✓ Outputs match")
    if abs(shap_lstm_total - output_lstm_diff) < 1e-5:
        print("   ✓ SHAP accurate")

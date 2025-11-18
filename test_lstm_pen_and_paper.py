"""
Pen-and-Paper Verifiable LSTM Test

This test uses simple, manually-set weights to verify LSTM SHAP calculations.
All values are chosen to be simple enough to verify with a calculator.

LSTM equations:
  i_t = sigmoid(W_i @ x_t + U_i @ h_{t-1} + b_i)  # input gate
  f_t = sigmoid(W_f @ x_t + U_f @ h_{t-1} + b_f)  # forget gate
  g_t = tanh(W_g @ x_t + U_g @ h_{t-1} + b_g)     # cell gate
  o_t = sigmoid(W_o @ x_t + U_o @ h_{t-1} + b_o)  # output gate
  c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t                 # cell state
  h_t = o_t ⊙ tanh(c_t)                            # hidden state

For 1 timestep with h_0=0, c_0=0:
  i = sigmoid(W_i @ x + b_i)
  f = sigmoid(W_f @ x + b_f)
  g = tanh(W_g @ x + b_g)
  o = sigmoid(W_o @ x + b_o)
  c = i ⊙ g
  h = o ⊙ tanh(c)
"""
import tensorflow as tf
import numpy as np
import shap

print("="*80)
print("Pen-and-Paper Verifiable LSTM Test")
print("="*80)

# Simple dimensions
input_size = 2
hidden_size = 2
batch_size = 1

print(f"\nDimensions:")
print(f"  Input size: {input_size}")
print(f"  Hidden size: {hidden_size}")
print(f"  Batch size: {batch_size}")

# Create LSTM
lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=False)

# Build it
dummy = tf.constant(np.zeros((1, 1, input_size)), dtype=np.float32)
_ = lstm(dummy)

print(f"\nLSTM weight shapes:")
print(f"  kernel: {lstm.cell.kernel.shape}  # [input_size, 4*hidden_size]")
print(f"  recurrent_kernel: {lstm.cell.recurrent_kernel.shape}  # [hidden_size, 4*hidden_size]")
print(f"  bias: {lstm.cell.bias.shape}  # [4*hidden_size]")

# Set simple weights manually
# kernel = [W_i | W_f | W_g | W_o] where each is [input_size x hidden_size]
# We'll use very simple values

# For simplicity, let's use identity-like weights with small values
kernel = np.array([
    # Input gate, Forget gate, Cell gate, Output gate (4 * hidden_size columns)
    [1.0, 0.0,  0.5, 0.0,  0.5, 0.0,  1.0, 0.0],  # x[0] weights
    [0.0, 1.0,  0.0, 0.5,  0.0, 0.5,  0.0, 1.0],  # x[1] weights
], dtype=np.float32)

# Since h_0 = 0, recurrent_kernel doesn't matter for 1st timestep
recurrent_kernel = np.zeros((hidden_size, 4 * hidden_size), dtype=np.float32)

# Biases - use small values
bias = np.array([
    # i0, i1, f0, f1, g0, g1, o0, o1
    0.0, 0.0,  # input gate bias
    0.0, 0.0,  # forget gate bias
    0.0, 0.0,  # cell gate bias
    0.0, 0.0,  # output gate bias
], dtype=np.float32)

# Set weights
lstm.cell.kernel.assign(kernel)
lstm.cell.recurrent_kernel.assign(recurrent_kernel)
lstm.cell.bias.assign(bias)

print(f"\nManually set weights:")
print(f"  kernel:\n{kernel}")
print(f"  recurrent_kernel: all zeros (doesn't matter for h_0=0)")
print(f"  bias: all zeros")

# Create simple test input
test_x = np.array([[
    [1.0, 0.5]  # timestep 0
]], dtype=np.float32)

baseline_x = np.zeros((1, 1, input_size), dtype=np.float32)

print(f"\nInputs:")
print(f"  Test input x: {test_x[0, 0, :]}")
print(f"  Baseline input: {baseline_x[0, 0, :]}")

# Manual calculation
print("\n" + "="*80)
print("Manual Calculation (Pen & Paper)")
print("="*80)

x = test_x[0, 0, :]  # [1.0, 0.5]
print(f"\nInput x = {x}")

# Split kernel into gates
W_i = kernel[:, 0:2]   # input gate
W_f = kernel[:, 2:4]   # forget gate
W_g = kernel[:, 4:6]   # cell gate
W_o = kernel[:, 6:8]   # output gate

print(f"\nWeights:")
print(f"  W_i (input gate):\n{W_i}")
print(f"  W_f (forget gate):\n{W_f}")
print(f"  W_g (cell gate):\n{W_g}")
print(f"  W_o (output gate):\n{W_o}")

# Compute pre-activations
z_i = W_i.T @ x  # [2]
z_f = W_f.T @ x
z_g = W_g.T @ x
z_o = W_o.T @ x

print(f"\nPre-activations:")
print(f"  z_i = W_i^T @ x = {z_i}")
print(f"  z_f = W_f^T @ x = {z_f}")
print(f"  z_g = W_g^T @ x = {z_g}")
print(f"  z_o = W_o^T @ x = {z_o}")

# Gates
i = 1 / (1 + np.exp(-z_i))  # sigmoid
f = 1 / (1 + np.exp(-z_f))
g = np.tanh(z_g)
o = 1 / (1 + np.exp(-z_o))

print(f"\nGates (after activation):")
print(f"  i (input gate) = sigmoid(z_i) = {i}")
print(f"  f (forget gate) = sigmoid(z_f) = {f}")
print(f"  g (cell gate) = tanh(z_g) = {g}")
print(f"  o (output gate) = sigmoid(z_o) = {o}")

# Cell and hidden state (with c_0 = 0, h_0 = 0)
c = i * g  # f * c_0 = 0
h = o * np.tanh(c)

print(f"\nCell and hidden states:")
print(f"  c = i ⊙ g = {c}")
print(f"  h = o ⊙ tanh(c) = {h}")
print(f"  h[0] + h[1] = {h[0] + h[1]}")

manual_output = h
manual_sum = h.sum()

print(f"\nManual calculation result:")
print(f"  h = {manual_output}")
print(f"  sum(h) = {manual_sum:.6f}")

# TensorFlow forward pass
print("\n" + "="*80)
print("TensorFlow Forward Pass")
print("="*80)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1, input_size)),
    lstm
])

output_test = model(test_x).numpy()
output_base = model(baseline_x).numpy()

print(f"\nTensorFlow outputs:")
print(f"  h (test) = {output_test[0]}")
print(f"  h (baseline) = {output_base[0]}")
print(f"  difference = {output_test[0] - output_base[0]}")
print(f"  sum(difference) = {(output_test - output_base).sum():.6f}")

# Verify manual calculation matches TF
tf_sum = (output_test - output_base).sum()
manual_vs_tf = abs(manual_sum - tf_sum)

print(f"\nManual vs TensorFlow:")
print(f"  Manual sum: {manual_sum:.6f}")
print(f"  TF sum: {tf_sum:.6f}")
print(f"  Difference: {manual_vs_tf:.10f}")

if manual_vs_tf < 1e-6:
    print("  ✅ Perfect match!")
else:
    print(f"  ⚠️  Difference: {manual_vs_tf}")

# SHAP calculation
print("\n" + "="*80)
print("SHAP Calculation")
print("="*80)

explainer = shap.DeepExplainer(model, baseline_x)
shap_values = explainer.shap_values(test_x, check_additivity=False)

print(f"\nSHAP values shape: {shap_values.shape}")
print(f"SHAP values (for each input feature at each timestep):")
print(shap_values)

shap_total = shap_values.sum()
expected_diff = (output_test - output_base).sum()

print(f"\nSHAP verification:")
print(f"  Expected difference: {expected_diff:.6f}")
print(f"  SHAP total: {shap_total:.6f}")
print(f"  Error: {abs(shap_total - expected_diff):.10f}")

if abs(shap_total - expected_diff) < 1e-6:
    print("  ✅ Perfect additivity!")
else:
    print(f"  Error: {abs(shap_total - expected_diff):.6f}")

# Final summary
print("\n" + "="*80)
print("SUMMARY - Pen & Paper Verification")
print("="*80)

print(f"\nGiven:")
print(f"  x = {test_x[0, 0, :]}")
print(f"  kernel (simple weights)")
print(f"  bias = 0")
print(f"  h_0 = c_0 = 0")

print(f"\nExpected output (manual calculation):")
print(f"  h = {manual_output}")
print(f"  sum(h) = {manual_sum:.6f}")

print(f"\nTensorFlow output:")
print(f"  h = {output_test[0]}")
print(f"  sum(h) = {tf_sum:.6f}")

print(f"\nSHAP values:")
print(f"  sum(SHAP) = {shap_total:.6f}")

print(f"\nVerification:")
if manual_vs_tf < 1e-6:
    print(f"  ✅ Manual calculation matches TensorFlow")
else:
    print(f"  ⚠️  Manual vs TF differ by {manual_vs_tf:.6f}")

if abs(shap_total - expected_diff) < 1e-6:
    print(f"  ✅ SHAP values are perfect")
else:
    print(f"  ⚠️  SHAP error: {abs(shap_total - expected_diff):.6f}")

print("\n" + "="*80)
print("Instructions for manual verification:")
print("="*80)
print("""
You can verify this with a calculator:

1. Input: x = [1.0, 0.5]

2. Calculate pre-activations (matrix multiply):
   z_i = [1.0, 0.0] (from W_i^T @ x)
   z_f = [0.5, 0.25] (from W_f^T @ x)
   z_g = [0.5, 0.25] (from W_g^T @ x)
   z_o = [1.0, 0.5] (from W_o^T @ x)

3. Apply activations:
   i = sigmoid(z_i)
   f = sigmoid(z_f)
   g = tanh(z_g)
   o = sigmoid(z_o)

4. Compute cell state: c = i ⊙ g

5. Compute hidden state: h = o ⊙ tanh(c)

6. Sum the hidden state values: sum(h)

This should match the SHAP total!
""")

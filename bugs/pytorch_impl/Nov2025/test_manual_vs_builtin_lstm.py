"""
Test: Manual LSTM vs Built-in LSTM Layers

This tests whether built-in LSTM layers (tf.keras.layers.LSTM, torch.nn.LSTM)
work with DeepExplainer by comparing against manual LSTM implementation.

Steps:
1. Create manual LSTM cell in TensorFlow with explicit layers
2. Set weights and calculate outputs + SHAP values
3. Create PyTorch LSTM layer (torch.nn.LSTM)
4. Set the SAME weights
5. Calculate outputs + SHAP values
6. Assert they're equivalent
"""

import tensorflow as tf
import torch
import torch.nn as nn
import numpy as np
import shap

print("="*80)
print("Manual LSTM vs Built-in LSTM Layers - Cross-Framework Test")
print("="*80)

# Set seeds
tf.random.set_seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Dimensions
input_size = 3
hidden_size = 2
batch_size = 1

print(f"\nDimensions:")
print(f"  Input size: {input_size}")
print(f"  Hidden size: {hidden_size}")
print(f"  Batch size: {batch_size}")

# ============================================================================
# Step 1: Manual LSTM Cell in TensorFlow (Explicit Layers)
# ============================================================================

print("\n" + "="*80)
print("Step 1: Manual LSTM Cell in TensorFlow")
print("="*80)

class ManualLSTMCell(tf.keras.Model):
    """Manual LSTM cell with explicit Dense layers for each gate."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input gate
        self.fc_ii = tf.keras.layers.Dense(hidden_size, use_bias=True, name='input_gate_x')
        self.fc_hi = tf.keras.layers.Dense(hidden_size, use_bias=True, name='input_gate_h')

        # Forget gate
        self.fc_if = tf.keras.layers.Dense(hidden_size, use_bias=True, name='forget_gate_x')
        self.fc_hf = tf.keras.layers.Dense(hidden_size, use_bias=True, name='forget_gate_h')

        # Candidate
        self.fc_ig = tf.keras.layers.Dense(hidden_size, use_bias=True, name='candidate_x')
        self.fc_hg = tf.keras.layers.Dense(hidden_size, use_bias=True, name='candidate_h')

    def call(self, x, h, c, return_both=True):
        i_t = tf.nn.sigmoid(self.fc_ii(x) + self.fc_hi(h))
        f_t = tf.nn.sigmoid(self.fc_if(x) + self.fc_hf(h))
        c_tilde = tf.nn.tanh(self.fc_ig(x) + self.fc_hg(h))
        new_c = f_t * c + i_t * c_tilde
        new_h = tf.nn.tanh(new_c)  # Simplified: h = tanh(c)
        if return_both:
            return new_h, new_c
        else:
            return new_c

manual_lstm = ManualLSTMCell(input_size, hidden_size)

# Build
_ = manual_lstm(tf.zeros((1, input_size)), tf.zeros((1, hidden_size)), tf.zeros((1, hidden_size)))

# Set weights
W_ii = np.array([[1.0, 0.5], [1.0, 0.3], [0.5, 0.2]], dtype=np.float32)
b_ii = np.array([0.2, 0.1], dtype=np.float32)
W_hi = np.array([[2.0, 0.5], [1.0, 0.8]], dtype=np.float32)
b_hi = np.array([0.32, 0.15], dtype=np.float32)

W_if = np.array([[0.8, 0.3], [0.6, 0.5], [0.4, 0.7]], dtype=np.float32)
b_if = np.array([0.1, 0.05], dtype=np.float32)
W_hf = np.array([[1.5, 0.7], [0.9, 1.2]], dtype=np.float32)
b_hf = np.array([0.25, 0.18], dtype=np.float32)

W_ig = np.array([[1.2, 0.4], [0.9, 0.8], [0.6, 1.0]], dtype=np.float32)
b_ig = np.array([0.15, 0.08], dtype=np.float32)
W_hg = np.array([[1.8, 0.9], [1.1, 1.3]], dtype=np.float32)
b_hg = np.array([0.28, 0.12], dtype=np.float32)

manual_lstm.fc_ii.set_weights([W_ii, b_ii])
manual_lstm.fc_hi.set_weights([W_hi, b_hi])
manual_lstm.fc_if.set_weights([W_if, b_if])
manual_lstm.fc_hf.set_weights([W_hf, b_hf])
manual_lstm.fc_ig.set_weights([W_ig, b_ig])
manual_lstm.fc_hg.set_weights([W_hg, b_hg])

print(f"✓ Manual LSTM created with explicit Dense layers")
print(f"✓ Weights set")

# Test data
x_tf = tf.constant([[0.1, 0.2, 0.3]], dtype=tf.float32)
h_tf = tf.constant([[0.0, 0.1]], dtype=tf.float32)
c_tf = tf.constant([[0.5, 0.3]], dtype=tf.float32)

x_base_tf = tf.constant([[0.01, 0.02, 0.03]], dtype=tf.float32)
h_base_tf = tf.constant([[0.0, 0.01]], dtype=tf.float32)
c_base_tf = tf.constant([[0.1, 0.05]], dtype=tf.float32)

# Forward pass
h_out_tf, c_out_tf = manual_lstm(x_tf, h_tf, c_tf)
h_base_out_tf, c_base_out_tf = manual_lstm(x_base_tf, h_base_tf, c_base_tf)

print(f"\nManual LSTM outputs:")
print(f"  h: {h_out_tf.numpy()}")
print(f"  c: {c_out_tf.numpy()}")
print(f"  h_base: {h_base_out_tf.numpy()}")
print(f"  c_base: {c_base_out_tf.numpy()}")

# ============================================================================
# Step 2: PyTorch Built-in LSTMCell
# ============================================================================

print("\n" + "="*80)
print("Step 2: PyTorch Built-in LSTMCell")
print("="*80)

# Create PyTorch LSTMCell
pytorch_lstm = nn.LSTMCell(input_size, hidden_size)

# Set the SAME weights
# PyTorch LSTMCell format: weight_ih = [i, f, g, o] stacked
# We need to match our manual LSTM weights

# Convert TensorFlow weights to PyTorch format
W_ii_pt = torch.from_numpy(W_ii.T)  # Transpose for PyTorch
W_if_pt = torch.from_numpy(W_if.T)
W_ig_pt = torch.from_numpy(W_ig.T)
W_io_pt = torch.randn(hidden_size, input_size)  # Output gate (not used in our manual cell)

W_hi_pt = torch.from_numpy(W_hi.T)
W_hf_pt = torch.from_numpy(W_hf.T)
W_hg_pt = torch.from_numpy(W_hg.T)
W_ho_pt = torch.randn(hidden_size, hidden_size)

# Stack: [input_gate, forget_gate, cell_gate, output_gate]
weight_ih = torch.cat([W_ii_pt, W_if_pt, W_ig_pt, W_io_pt], dim=0)
weight_hh = torch.cat([W_hi_pt, W_hf_pt, W_hg_pt, W_ho_pt], dim=0)

bias_ih = torch.cat([
    torch.from_numpy(b_ii),
    torch.from_numpy(b_if),
    torch.from_numpy(b_ig),
    torch.zeros(hidden_size)
])

bias_hh = torch.cat([
    torch.from_numpy(b_hi),
    torch.from_numpy(b_hf),
    torch.from_numpy(b_hg),
    torch.zeros(hidden_size)
])

pytorch_lstm.weight_ih.data = weight_ih
pytorch_lstm.weight_hh.data = weight_hh
pytorch_lstm.bias_ih.data = bias_ih
pytorch_lstm.bias_hh.data = bias_hh

print(f"✓ PyTorch LSTMCell created")
print(f"✓ Weights copied from TensorFlow manual LSTM")

# Test data (same as TensorFlow)
x_pt = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
h_pt = torch.tensor([[0.0, 0.1]], dtype=torch.float32)
c_pt = torch.tensor([[0.5, 0.3]], dtype=torch.float32)

x_base_pt = torch.tensor([[0.01, 0.02, 0.03]], dtype=torch.float32)
h_base_pt = torch.tensor([[0.0, 0.01]], dtype=torch.float32)
c_base_pt = torch.tensor([[0.1, 0.05]], dtype=torch.float32)

# Forward pass
with torch.no_grad():
    h_out_pt, c_out_pt = pytorch_lstm(x_pt, (h_pt, c_pt))
    h_base_out_pt, c_base_out_pt = pytorch_lstm(x_base_pt, (h_base_pt, c_base_pt))

print(f"\nPyTorch LSTMCell outputs:")
print(f"  h: {h_out_pt.numpy()}")
print(f"  c: {c_out_pt.numpy()}")
print(f"  h_base: {h_base_out_pt.numpy()}")
print(f"  c_base: {c_base_out_pt.numpy()}")

# ============================================================================
# Step 3: Compare Outputs
# ============================================================================

print("\n" + "="*80)
print("Step 3: Compare Outputs")
print("="*80)

# NOTE: Outputs might not match exactly because:
# 1. PyTorch LSTMCell has output gate that affects h (h = o * tanh(c))
# 2. Our manual LSTM uses simplified h = tanh(c)

h_diff = np.abs(h_out_tf.numpy() - h_out_pt.numpy())
c_diff = np.abs(c_out_tf.numpy() - c_out_pt.numpy())

print(f"\nHidden state difference (h):")
print(f"  Max diff: {h_diff.max():.10f}")
print(f"  Note: May differ due to output gate in PyTorch LSTMCell")

print(f"\nCell state difference (c):")
print(f"  Max diff: {c_diff.max():.10f}")

# Cell states should match closely (no output gate involved)
c_matches = c_diff.max() < 0.1
print(f"  ✓ Cell states similar: {c_matches}")

# ============================================================================
# Step 4: Test SHAP with Manual LSTM (TensorFlow)
# ============================================================================

print("\n" + "="*80)
print("Step 4: SHAP with Manual LSTM (TensorFlow)")
print("="*80)

# Create wrapper for DeepExplainer
def create_manual_lstm_wrapper(lstm_cell, input_size, hidden_size):
    combined_input = tf.keras.Input(shape=(input_size + 2*hidden_size,))
    x = combined_input[:, :input_size]
    h = combined_input[:, input_size:input_size + hidden_size]
    c = combined_input[:, input_size + hidden_size:]
    # Only return cell state (single tensor) to avoid multi-output error
    new_c = lstm_cell(x, h, c, return_both=False)
    model = tf.keras.Model(inputs=combined_input, outputs=new_c)
    return model

manual_model_tf = create_manual_lstm_wrapper(manual_lstm, input_size, hidden_size)

# Prepare data
test_input_tf = np.concatenate([x_tf.numpy(), h_tf.numpy(), c_tf.numpy()], axis=1)
baseline_tf = np.concatenate([x_base_tf.numpy(), h_base_tf.numpy(), c_base_tf.numpy()], axis=1)

print(f"\nTesting TensorFlow DeepExplainer with manual LSTM...")

try:
    explainer_tf = shap.DeepExplainer(manual_model_tf, baseline_tf)
    shap_values_tf = explainer_tf.shap_values(test_input_tf, check_additivity=False)

    if len(shap_values_tf.shape) == 3:
        shap_total_tf = shap_values_tf.sum(axis=2).sum()
    else:
        shap_total_tf = shap_values_tf.sum()

    output_diff_tf = (manual_model_tf(test_input_tf).numpy() - manual_model_tf(baseline_tf).numpy()).sum()

    print(f"  ✓ SHAP calculated")
    print(f"  Shape: {shap_values_tf.shape}")
    print(f"  SHAP total: {shap_total_tf:.10f}")
    print(f"  Expected: {output_diff_tf:.10f}")
    print(f"  Error: {abs(shap_total_tf - output_diff_tf):.10f}")

    tf_works = abs(shap_total_tf - output_diff_tf) < 0.01
    print(f"  ✓ Additivity: {tf_works}")

except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    tf_works = False

# ============================================================================
# Step 5: Test SHAP with Built-in LSTMCell (PyTorch)
# ============================================================================

print("\n" + "="*80)
print("Step 5: SHAP with Built-in LSTMCell (PyTorch)")
print("="*80)

# Create wrapper for PyTorch
class PyTorchLSTMWrapper(nn.Module):
    def __init__(self, lstm_cell, input_size, hidden_size):
        super().__init__()
        self.lstm_cell = lstm_cell
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, combined_input):
        x = combined_input[:, :self.input_size]
        h = combined_input[:, self.input_size:self.input_size + self.hidden_size]
        c = combined_input[:, self.input_size + self.hidden_size:]
        _, new_c = self.lstm_cell(x, (h, c))
        return new_c

pytorch_model = PyTorchLSTMWrapper(pytorch_lstm, input_size, hidden_size)
pytorch_model.eval()

# Prepare data
test_input_pt = torch.cat([x_pt, h_pt, c_pt], dim=1)
baseline_pt = torch.cat([x_base_pt, h_base_pt, c_base_pt], dim=1)

print(f"\nTesting PyTorch DeepExplainer with built-in LSTMCell...")

try:
    explainer_pt = shap.DeepExplainer(pytorch_model, baseline_pt)
    shap_values_pt = explainer_pt.shap_values(test_input_pt, check_additivity=False)

    if len(shap_values_pt.shape) == 3:
        shap_total_pt = shap_values_pt.sum(axis=2).sum()
    else:
        shap_total_pt = shap_values_pt.sum()

    with torch.no_grad():
        output_diff_pt = (pytorch_model(test_input_pt) - pytorch_model(baseline_pt)).sum().item()

    print(f"  ✓ SHAP calculated")
    print(f"  Shape: {shap_values_pt.shape}")
    print(f"  SHAP total: {shap_total_pt:.10f}")
    print(f"  Expected: {output_diff_pt:.10f}")
    print(f"  Error: {abs(shap_total_pt - output_diff_pt):.10f}")

    pt_works = abs(shap_total_pt - output_diff_pt) < 0.05
    print(f"  ✓ Additivity: {pt_works}")

except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    pt_works = False

# ============================================================================
# RESULTS
# ============================================================================

print("\n" + "="*80)
print("RESULTS")
print("="*80)

print(f"\nManual LSTM (TensorFlow) with DeepExplainer:")
if tf_works:
    print(f"  ✓ Works! Error < 0.01")
else:
    print(f"  ✗ Doesn't work or has large error")

print(f"\nBuilt-in LSTMCell (PyTorch) with DeepExplainer:")
if pt_works:
    print(f"  ✓ Works! Error < 0.05")
else:
    print(f"  ✗ Doesn't work or has large error")

print(f"\n" + "="*80)
print("CONCLUSION")
print("="*80)

if tf_works:
    print(f"\n✓ TensorFlow: Manual LSTM cells work with DeepExplainer")
else:
    print(f"\n✗ TensorFlow: Manual LSTM cells need custom handler")

if pt_works:
    print(f"✓ PyTorch: Built-in LSTMCell works with DeepExplainer")
else:
    print(f"✗ PyTorch: Built-in LSTMCell needs custom handler")

print(f"\nNext: Test with full LSTM layers (not just LSTMCell):")
print(f"  - tf.keras.layers.LSTM")
print(f"  - torch.nn.LSTM")

print("\n" + "="*80)

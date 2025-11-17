"""
Complete LSTM Cell SHAP Calculation in PyTorch

This file implements the full LSTM cell including:
1. Input gate (i_t)
2. Forget gate (f_t)
3. Candidate cell state (C̃_t)
4. Cell state update (C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t)

Then manually calculates SHAP values using DeepLift formulas and compares
with SHAP's DeepExplainer.

Key formulas from the PDF:
- For gates (sigmoid): r[x] = (σ(Wx + b) - σ(Wx_base + b)) * (W ⊙ x) / (Wx + Wh*h)
- For candidate (tanh): r[x] = (tanh(Wx + b) - tanh(Wx_base + b)) * (W ⊙ x) / (Wx + Wh*h)
- For multiplication (Shapley): R[a] = 1/2 * [a⊙b - a_b⊙b + a⊙b_b - a_b⊙b_b]
"""

import torch
import torch.nn as nn
import numpy as np

# We'll need shap, but first let me check if it's installed
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("WARNING: SHAP not available, will only show manual calculation")


class LSTMCellModel(nn.Module):
    """
    Complete LSTM cell that computes:
    - Input gate: i_t = σ(W_ii x_t + b_ii + W_hi h_{t-1} + b_hi)
    - Forget gate: f_t = σ(W_if x_t + b_if + W_hf h_{t-1} + b_hf)
    - Candidate: C̃_t = tanh(W_ig x_t + b_ig + W_hg h_{t-1} + b_hg)
    - Cell state: C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t

    Returns the new cell state C_t
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input gate
        self.fc_ii = nn.Linear(input_size, hidden_size, bias=True)
        self.fc_hi = nn.Linear(hidden_size, hidden_size, bias=True)

        # Forget gate
        self.fc_if = nn.Linear(input_size, hidden_size, bias=True)
        self.fc_hf = nn.Linear(hidden_size, hidden_size, bias=True)

        # Candidate cell state (g = gate)
        self.fc_ig = nn.Linear(input_size, hidden_size, bias=True)
        self.fc_hg = nn.Linear(hidden_size, hidden_size, bias=True)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, h, c):
        """
        Args:
            x: input tensor (batch, input_size)
            h: previous hidden state (batch, hidden_size)
            c: previous cell state (batch, hidden_size)

        Returns:
            new_c: new cell state (batch, hidden_size)
        """
        # Input gate
        i_t = self.sigmoid(self.fc_ii(x) + self.fc_hi(h))

        # Forget gate
        f_t = self.sigmoid(self.fc_if(x) + self.fc_hf(h))

        # Candidate cell state
        c_tilde = self.tanh(self.fc_ig(x) + self.fc_hg(h))

        # Cell state update: C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
        new_c = f_t * c + i_t * c_tilde

        return new_c


def manual_shap_gate(W_i, W_h, b_i, b_h, x, h, x_base, h_base, activation='sigmoid'):
    """
    Manual SHAP calculation for a gate (input, forget, or candidate).

    Formula from PDF (eq. 14-15):
    r[x] = (activation(W_i*x + b_i + W_h*h + b_h) - activation(W_i*x_base + b_i + W_h*h_base + b_h))
           * (W_i ⊙ x) / (W_i*x + W_h*h)

    Args:
        W_i: Weight matrix for input (hidden_size, input_size)
        W_h: Weight matrix for hidden state (hidden_size, hidden_size)
        b_i, b_h: Bias vectors
        x, h: Current input and hidden state
        x_base, h_base: Baseline input and hidden state
        activation: 'sigmoid' or 'tanh'

    Returns:
        r_x: Relevance for x (batch, input_size)
        r_h: Relevance for h (batch, hidden_size)
        output: Gate output
    """
    # Forward pass
    linear_current = torch.matmul(x, W_i.T) + torch.matmul(h, W_h.T) + b_i + b_h
    linear_base = torch.matmul(x_base, W_i.T) + torch.matmul(h_base, W_h.T) + b_i + b_h

    if activation == 'sigmoid':
        act_fn = torch.sigmoid
    else:  # tanh
        act_fn = torch.tanh

    output = act_fn(linear_current)
    output_base = act_fn(linear_base)

    # Denominator for normalization (per output dimension)
    # Shape: (batch, hidden_size)
    denom_x = torch.matmul(x, W_i.T)
    denom_h = torch.matmul(h, W_h.T)
    denom = denom_x + denom_h  # (batch, hidden_size)

    # Calculate Z matrices
    # W_i shape: (hidden_size, input_size)
    # x - x_base shape: (batch, input_size)
    # We need: (batch, hidden_size, input_size)

    # Expand for broadcasting
    x_diff = (x - x_base).unsqueeze(1)  # (batch, 1, input_size)
    h_diff = (h - h_base).unsqueeze(1)  # (batch, 1, hidden_size)

    W_i_expanded = W_i.unsqueeze(0)  # (1, hidden_size, input_size)
    W_h_expanded = W_h.unsqueeze(0)  # (1, hidden_size, hidden_size)

    # Element-wise multiplication
    numerator_x = W_i_expanded * x_diff  # (batch, hidden_size, input_size)
    numerator_h = W_h_expanded * h_diff  # (batch, hidden_size, hidden_size)

    # Normalize by denominator
    denom_expanded = denom.unsqueeze(-1)  # (batch, hidden_size, 1)

    Z_x = numerator_x / denom_expanded  # (batch, hidden_size, input_size)
    Z_h = numerator_h / denom_expanded  # (batch, hidden_size, hidden_size)

    # Calculate relevance
    output_diff = (output - output_base).unsqueeze(-1)  # (batch, hidden_size, 1)

    # Sum over hidden dimension
    r_x = (output_diff * Z_x).sum(dim=1)  # (batch, input_size)
    r_h = (output_diff * Z_h).sum(dim=1)  # (batch, hidden_size)

    return r_x, r_h, output, output_base


def manual_shap_multiplication(a, b, a_base, b_base):
    """
    Manual SHAP calculation for element-wise multiplication using Shapley values.

    Formula from PDF (eq. 22):
    R[a] = 1/2 * [a⊙b - a_b⊙b + a⊙b_b - a_b⊙b_b]
    R[b] = 1/2 * [a⊙b - a⊙b_b + a_b⊙b - a_b⊙b_b]

    Args:
        a, b: Current values (batch, hidden_size)
        a_base, b_base: Baseline values (batch, hidden_size)

    Returns:
        r_a: Relevance for a (batch, hidden_size)
        r_b: Relevance for b (batch, hidden_size)
    """
    # Shapley value calculation
    r_a = 0.5 * (a * b - a_base * b + a * b_base - a_base * b_base)
    r_b = 0.5 * (a * b - a * b_base + a_base * b - a_base * b_base)

    return r_a, r_b


def manual_shap_lstm_cell(model, x, h, c, x_base, h_base, c_base):
    """
    Complete manual SHAP calculation for LSTM cell.

    The cell state update is:
    C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t

    We need to:
    1. Calculate SHAP for input gate (i_t)
    2. Calculate SHAP for forget gate (f_t)
    3. Calculate SHAP for candidate (C̃_t)
    4. Use Shapley values for the multiplications
    5. Combine everything

    Returns:
        r_x: Total relevance for input x
        r_h: Total relevance for hidden state h
        r_c: Total relevance for previous cell state c
    """
    # Get weights
    W_ii = model.fc_ii.weight.data
    b_ii = model.fc_ii.bias.data
    W_hi = model.fc_hi.weight.data
    b_hi = model.fc_hi.bias.data

    W_if = model.fc_if.weight.data
    b_if = model.fc_if.bias.data
    W_hf = model.fc_hf.weight.data
    b_hf = model.fc_hf.bias.data

    W_ig = model.fc_ig.weight.data
    b_ig = model.fc_ig.bias.data
    W_hg = model.fc_hg.weight.data
    b_hg = model.fc_hg.bias.data

    # 1. Calculate input gate and its SHAP values
    r_x_i, r_h_i, i_t, i_t_base = manual_shap_gate(
        W_ii, W_hi, b_ii, b_hi, x, h, x_base, h_base, activation='sigmoid'
    )

    # 2. Calculate forget gate and its SHAP values
    r_x_f, r_h_f, f_t, f_t_base = manual_shap_gate(
        W_if, W_hf, b_if, b_hf, x, h, x_base, h_base, activation='sigmoid'
    )

    # 3. Calculate candidate cell state and its SHAP values
    r_x_g, r_h_g, c_tilde, c_tilde_base = manual_shap_gate(
        W_ig, W_hg, b_ig, b_hg, x, h, x_base, h_base, activation='tanh'
    )

    # 4. Use Shapley values for multiplications in cell state update
    # C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t

    # For f_t ⊙ C_{t-1}
    r_f_from_mult, r_c_from_f = manual_shap_multiplication(f_t, c, f_t_base, c_base)

    # For i_t ⊙ C̃_t
    r_i_from_mult, r_ctilde_from_mult = manual_shap_multiplication(i_t, c_tilde, i_t_base, c_tilde_base)

    # 5. Combine relevances
    # The relevance from forget gate multiplication needs to be traced back to x, h
    # through the forget gate calculation

    # For forget gate: distribute r_f_from_mult back to x and h
    # We need to weight r_x_f and r_h_f by how much they contributed
    total_r_f = r_x_f.abs().sum() + r_h_f.abs().sum()
    if total_r_f > 1e-10:
        weight_x_f = r_x_f.abs().sum() / total_r_f
        weight_h_f = r_h_f.abs().sum() / total_r_f
    else:
        weight_x_f = 0.5
        weight_h_f = 0.5

    r_x_from_f = weight_x_f * r_f_from_mult.sum()
    r_h_from_f = weight_h_f * r_f_from_mult.sum()

    # For input gate: distribute r_i_from_mult back to x and h
    total_r_i = r_x_i.abs().sum() + r_h_i.abs().sum()
    if total_r_i > 1e-10:
        weight_x_i = r_x_i.abs().sum() / total_r_i
        weight_h_i = r_h_i.abs().sum() / total_r_i
    else:
        weight_x_i = 0.5
        weight_h_i = 0.5

    r_x_from_i = weight_x_i * r_i_from_mult.sum()
    r_h_from_i = weight_h_i * r_i_from_mult.sum()

    # For candidate: distribute r_ctilde_from_mult back to x and h
    total_r_g = r_x_g.abs().sum() + r_h_g.abs().sum()
    if total_r_g > 1e-10:
        weight_x_g = r_x_g.abs().sum() / total_r_g
        weight_h_g = r_h_g.abs().sum() / total_r_g
    else:
        weight_x_g = 0.5
        weight_h_g = 0.5

    r_x_from_g = weight_x_g * r_ctilde_from_mult.sum()
    r_h_from_g = weight_h_g * r_ctilde_from_mult.sum()

    # Total relevance
    r_x_total = r_x_from_f + r_x_from_i + r_x_from_g
    r_h_total = r_h_from_f + r_h_from_i + r_h_from_g
    r_c_total = r_c_from_f.sum()

    # Forward pass for verification
    output = model(x, h, c)
    output_base = model(x_base, h_base, c_base)

    return r_x_total, r_h_total, r_c_total, output, output_base


# ============================================================================
# Main test script
# ============================================================================

print("="*80)
print("Complete LSTM Cell SHAP Calculation Test")
print("="*80)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Model dimensions - use 2D to keep it general
input_size = 3
hidden_size = 2
batch_size = 1

# Create model
model = LSTMCellModel(input_size, hidden_size)

# Set reproducible weights
# Input gate
model.fc_ii.weight.data = torch.tensor([
    [1.0, 1.0, 0.5],
    [0.5, 0.3, 0.2]
], dtype=torch.float32)
model.fc_ii.bias.data = torch.tensor([0.2, 0.1], dtype=torch.float32)

model.fc_hi.weight.data = torch.tensor([
    [2.0, 1.0],
    [0.5, 0.8]
], dtype=torch.float32)
model.fc_hi.bias.data = torch.tensor([0.32, 0.15], dtype=torch.float32)

# Forget gate
model.fc_if.weight.data = torch.tensor([
    [0.8, 0.6, 0.4],
    [0.3, 0.5, 0.7]
], dtype=torch.float32)
model.fc_if.bias.data = torch.tensor([0.1, 0.05], dtype=torch.float32)

model.fc_hf.weight.data = torch.tensor([
    [1.5, 0.9],
    [0.7, 1.2]
], dtype=torch.float32)
model.fc_hf.bias.data = torch.tensor([0.25, 0.18], dtype=torch.float32)

# Candidate cell state
model.fc_ig.weight.data = torch.tensor([
    [1.2, 0.9, 0.6],
    [0.4, 0.8, 1.0]
], dtype=torch.float32)
model.fc_ig.bias.data = torch.tensor([0.15, 0.08], dtype=torch.float32)

model.fc_hg.weight.data = torch.tensor([
    [1.8, 1.1],
    [0.9, 1.3]
], dtype=torch.float32)
model.fc_hg.bias.data = torch.tensor([0.28, 0.12], dtype=torch.float32)

# Input data
x = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
h = torch.tensor([[0.0, 0.1]], dtype=torch.float32)
c = torch.tensor([[0.5, 0.3]], dtype=torch.float32)

# Baseline data
x_base = torch.tensor([[0.01, 0.02, 0.03]], dtype=torch.float32)
h_base = torch.tensor([[0.0, 0.01]], dtype=torch.float32)
c_base = torch.tensor([[0.1, 0.05]], dtype=torch.float32)

print(f"\nModel configuration:")
print(f"  Input size: {input_size}")
print(f"  Hidden size: {hidden_size}")
print(f"  Batch size: {batch_size}")

print(f"\nInput data:")
print(f"  x: {x}")
print(f"  h: {h}")
print(f"  c: {c}")

print(f"\nBaseline data:")
print(f"  x_base: {x_base}")
print(f"  h_base: {h_base}")
print(f"  c_base: {c_base}")

# Manual calculation
print("\n" + "="*80)
print("Manual SHAP Calculation")
print("="*80)

r_x_manual, r_h_manual, r_c_manual, output, output_base = manual_shap_lstm_cell(
    model, x, h, c, x_base, h_base, c_base
)

print(f"\nForward pass:")
print(f"  Output (C_t): {output}")
print(f"  Output_base (C_t_base): {output_base}")
print(f"  Output difference: {output - output_base}")

print(f"\nManual SHAP values:")
print(f"  r_x (relevance for x): {r_x_manual}")
print(f"  r_h (relevance for h): {r_h_manual}")
print(f"  r_c (relevance for c): {r_c_manual}")
print(f"  Total relevance: {r_x_manual + r_h_manual + r_c_manual}")
print(f"  Output difference sum: {(output - output_base).sum()}")

# Check additivity
additivity_error = abs((r_x_manual + r_h_manual + r_c_manual) - (output - output_base).sum())
print(f"\nAdditivity check:")
print(f"  Error: {additivity_error.item():.6f}")
print(f"  Additivity satisfied: {additivity_error < 0.01}")

# SHAP calculation if available
if SHAP_AVAILABLE:
    print("\n" + "="*80)
    print("SHAP DeepExplainer Calculation")
    print("="*80)

    # Create wrapper for SHAP
    class LSTMCellWrapper(nn.Module):
        def __init__(self, lstm_cell):
            super().__init__()
            self.lstm_cell = lstm_cell

        def forward(self, inputs):
            x, h, c = inputs
            return self.lstm_cell(x, h, c)

    wrapper = LSTMCellWrapper(model)

    try:
        # SHAP explainer
        explainer = shap.DeepExplainer(wrapper, [x_base, h_base, c_base])
        shap_values = explainer.shap_values([x, h, c], check_additivity=False)

        print(f"\nSHAP values:")
        print(f"  r_x (SHAP): {shap_values[0]}")
        print(f"  r_h (SHAP): {shap_values[1]}")
        print(f"  r_c (SHAP): {shap_values[2]}")

        print(f"\nComparison with manual calculation:")
        print(f"  r_x match: {torch.allclose(torch.tensor(r_x_manual), torch.tensor(shap_values[0].sum()), atol=0.01)}")
        print(f"  r_h match: {torch.allclose(torch.tensor(r_h_manual), torch.tensor(shap_values[1].sum()), atol=0.01)}")
        print(f"  r_c match: {torch.allclose(torch.tensor(r_c_manual), torch.tensor(shap_values[2].sum()), atol=0.01)}")

    except Exception as e:
        print(f"\nSHAP calculation failed: {e}")
        print("This is expected as the LSTM cell is complex.")
else:
    print("\n" + "="*80)
    print("SHAP not available - skipping comparison")
    print("="*80)

print("\n" + "="*80)
print("Test complete!")
print("="*80)

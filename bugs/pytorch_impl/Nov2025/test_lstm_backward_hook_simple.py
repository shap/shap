"""
Simple test for LSTM backward hook with manual SHAP calculation

This test:
1. Uses the proven manual SHAP calculation from lstm_cell_complete_pytorch.py
2. Tests it with LSTMCell
3. Verifies additivity
"""

import torch
import torch.nn as nn
import numpy as np


def manual_shap_gate(W_i, W_h, b_i, b_h, x, h, x_base, h_base, activation='sigmoid'):
    """
    Manual SHAP calculation for a single LSTM gate (from lstm_cell_complete_pytorch.py).
    """
    linear_current = torch.matmul(x, W_i.T) + torch.matmul(h, W_h.T) + b_i + b_h
    linear_base = torch.matmul(x_base, W_i.T) + torch.matmul(h_base, W_h.T) + b_i + b_h

    if activation == 'sigmoid':
        act_fn = torch.sigmoid
    else:  # tanh
        act_fn = torch.tanh

    output = act_fn(linear_current)
    output_base = act_fn(linear_base)

    # Denominator for normalization
    denom = torch.matmul(x, W_i.T) + torch.matmul(h, W_h.T)  # (batch, hidden_size)

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
    """
    r_a = 0.5 * (a * b - a_base * b + a * b_base - a_base * b_base)
    r_b = 0.5 * (a * b - a * b_base + a_base * b - a_base * b_base)
    return r_a, r_b


def manual_shap_lstmcell(lstm_cell, x, h, c, x_base, h_base, c_base):
    """
    Complete manual SHAP calculation for LSTMCell (built-in PyTorch).
    """
    # Extract weights from LSTMCell
    # weight_ih: [4*hidden_size, input_size] - concatenated [W_ii, W_if, W_ig, W_io]
    # weight_hh: [4*hidden_size, hidden_size] - concatenated [W_hi, W_hf, W_hg, W_ho]

    hidden_size = lstm_cell.hidden_size
    W_ii, W_if, W_ig, W_io = torch.chunk(lstm_cell.weight_ih, 4, dim=0)
    W_hi, W_hf, W_hg, W_ho = torch.chunk(lstm_cell.weight_hh, 4, dim=0)

    if lstm_cell.bias:
        b_ii, b_if, b_ig, b_io = torch.chunk(lstm_cell.bias_ih, 4, dim=0)
        b_hi, b_hf, b_hg, b_ho = torch.chunk(lstm_cell.bias_hh, 4, dim=0)
    else:
        zeros = torch.zeros(hidden_size)
        b_ii = b_if = b_ig = b_io = zeros
        b_hi = b_hf = b_hg = b_ho = zeros

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

    # 4. Cell state update: c_new = f_t ⊙ c + i_t ⊙ c_tilde
    # Use Shapley values for multiplications

    # For f_t ⊙ c
    r_f_from_mult, r_c_from_f = manual_shap_multiplication(f_t, c, f_t_base, c_base)

    # For i_t ⊙ c_tilde
    r_i_from_mult, r_ctilde_from_mult = manual_shap_multiplication(i_t, c_tilde, i_t_base, c_tilde_base)

    # 5. Combine relevances
    # The multiplication relevances need to be distributed back to x, h
    # Use gate SHAP values as distribution weights

    # For forget gate: distribute r_f_from_mult back to x and h
    total_r_f = r_x_f.abs().sum() + r_h_f.abs().sum()
    if total_r_f > 1e-10:
        weight_x_f = r_x_f.abs().sum() / total_r_f
        weight_h_f = r_h_f.abs().sum() / total_r_f
    else:
        weight_x_f = 0.5
        weight_h_f = 0.5

    r_x_from_f = weight_x_f * r_f_from_mult.sum()  # SCALAR
    r_h_from_f = weight_h_f * r_f_from_mult.sum()  # SCALAR

    # For input gate: distribute r_i_from_mult back to x and h
    total_r_i = r_x_i.abs().sum() + r_h_i.abs().sum()
    if total_r_i > 1e-10:
        weight_x_i = r_x_i.abs().sum() / total_r_i
        weight_h_i = r_h_i.abs().sum() / total_r_i
    else:
        weight_x_i = 0.5
        weight_h_i = 0.5

    r_x_from_i = weight_x_i * r_i_from_mult.sum()  # SCALAR
    r_h_from_i = weight_h_i * r_i_from_mult.sum()  # SCALAR

    # For candidate: distribute r_ctilde_from_mult back to x and h
    total_r_g = r_x_g.abs().sum() + r_h_g.abs().sum()
    if total_r_g > 1e-10:
        weight_x_g = r_x_g.abs().sum() / total_r_g
        weight_h_g = r_h_g.abs().sum() / total_r_g
    else:
        weight_x_g = 0.5
        weight_h_g = 0.5

    r_x_from_g = weight_x_g * r_ctilde_from_mult.sum()  # SCALAR
    r_h_from_g = weight_h_g * r_ctilde_from_mult.sum()  # SCALAR

    # Total SHAP values (SCALARS)
    shap_x = r_x_from_f + r_x_from_i + r_x_from_g
    shap_h = r_h_from_f + r_h_from_i + r_h_from_g
    shap_c = r_c_from_f.sum()  # Sum to scalar

    return shap_x, shap_h, shap_c


def test_manual_shap_with_lstmcell():
    """
    Test manual SHAP calculation with PyTorch LSTMCell
    """
    print("="*80)
    print("LSTM Manual SHAP Test with LSTMCell")
    print("="*80)

    # Set seed
    torch.manual_seed(42)

    # Dimensions
    input_size = 3
    hidden_size = 4
    batch_size = 2

    print(f"\nDimensions:")
    print(f"  Input size: {input_size}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Batch size: {batch_size}")

    # Create LSTMCell
    lstm_cell = nn.LSTMCell(input_size, hidden_size)

    # Test inputs
    x = torch.randn(batch_size, input_size)
    h = torch.randn(batch_size, hidden_size)
    c = torch.randn(batch_size, hidden_size)

    # Baseline (zeros)
    x_base = torch.zeros_like(x)
    h_base = torch.zeros_like(h)
    c_base = torch.zeros_like(c)

    # Forward pass
    h_new, c_new = lstm_cell(x, (h, c))
    h_new_base, c_new_base = lstm_cell(x_base, (h_base, c_base))

    # Calculate expected differences
    delta_c = (c_new - c_new_base).sum().item()
    delta_h = (h_new - h_new_base).sum().item()

    print(f"\nExpected output differences:")
    print(f"  Δc (sum): {delta_c:.6f}")
    print(f"  Δh (sum): {delta_h:.6f}")

    # Calculate manual SHAP (returns scalars)
    shap_x, shap_h, shap_c = manual_shap_lstmcell(lstm_cell, x, h, c, x_base, h_base, c_base)

    # Check additivity
    if isinstance(shap_x, torch.Tensor):
        shap_x = shap_x.item()
    if isinstance(shap_h, torch.Tensor):
        shap_h = shap_h.item()
    if isinstance(shap_c, torch.Tensor):
        shap_c = shap_c.item()

    shap_total = shap_x + shap_h + shap_c

    print(f"\nManual SHAP values:")
    print(f"  SHAP(x): {shap_x:.6f}")
    print(f"  SHAP(h): {shap_h:.6f}")
    print(f"  SHAP(c): {shap_c:.6f}")
    print(f"  SHAP total: {shap_total:.6f}")

    print(f"\nAdditivity check (for cell state c):")
    error_c = abs(shap_total - delta_c)
    print(f"  |SHAP_total - Δc|: {error_c:.6f}")
    print(f"  Relative error: {error_c / (abs(delta_c) + 1e-10) * 100:.2f}%")

    if error_c < 0.01:
        print(f"\n✓ Additivity test PASSED (error < 0.01)")
        return True
    else:
        print(f"\n✗ Additivity test FAILED (error = {error_c:.6f})")
        return False


if __name__ == "__main__":
    test_manual_shap_with_lstmcell()

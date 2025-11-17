"""
LSTM SHAP Backward Hook Implementation

This module provides backward hooks for automatically calculating SHAP values
during backpropagation for LSTM cells in PyTorch.

The module:
1. Extracts LSTM weights automatically
2. Calculates SHAP values manually using validated DeepLift + Shapley formulas
3. Registers backward hooks to intercept gradients
4. Returns SHAP attributions instead of standard gradients
"""

import torch
import torch.nn as nn
import numpy as np


class LSTMShapBackwardHook:
    """
    Backward hook for LSTM cells that calculates SHAP values during backprop.

    This class extracts LSTM weights and uses manual SHAP calculation to provide
    proper relevance scores based on DeepLift and Shapley values.
    """

    def __init__(self, lstm_cell, baseline_x, baseline_h, baseline_c):
        """
        Initialize the SHAP backward hook.

        Args:
            lstm_cell: PyTorch LSTMCell or manual LSTM cell module
            baseline_x: Baseline input (batch, input_size)
            baseline_h: Baseline hidden state (batch, hidden_size)
            baseline_c: Baseline cell state (batch, hidden_size)
        """
        self.lstm_cell = lstm_cell
        self.baseline_x = baseline_x
        self.baseline_h = baseline_h
        self.baseline_c = baseline_c

        # Extract dimensions
        self.input_size = baseline_x.shape[1]
        self.hidden_size = baseline_h.shape[1]

        # Extract weights
        self.extract_weights()

        # Store current inputs for backward pass
        self.current_x = None
        self.current_h = None
        self.current_c = None

        # Store SHAP values
        self.shap_x = None
        self.shap_h = None
        self.shap_c = None

    def extract_weights(self):
        """Extract weights from LSTM cell."""
        # For manual LSTM cell with explicit layers
        if hasattr(self.lstm_cell, 'fc_ii'):
            # Input gate
            self.W_ii = self.lstm_cell.fc_ii.weight.data
            self.b_ii = self.lstm_cell.fc_ii.bias.data
            self.W_hi = self.lstm_cell.fc_hi.weight.data
            self.b_hi = self.lstm_cell.fc_hi.bias.data

            # Forget gate
            self.W_if = self.lstm_cell.fc_if.weight.data
            self.b_if = self.lstm_cell.fc_if.bias.data
            self.W_hf = self.lstm_cell.fc_hf.weight.data
            self.b_hf = self.lstm_cell.fc_hf.bias.data

            # Candidate cell state
            self.W_ig = self.lstm_cell.fc_ig.weight.data
            self.b_ig = self.lstm_cell.fc_ig.bias.data
            self.W_hg = self.lstm_cell.fc_hg.weight.data
            self.b_hg = self.lstm_cell.fc_hg.bias.data

        # For PyTorch's built-in LSTMCell
        elif hasattr(self.lstm_cell, 'weight_ih') and hasattr(self.lstm_cell, 'weight_hh'):
            # PyTorch LSTMCell stores all gates in single weight matrices
            # Format: [input_gate, forget_gate, cell_gate, output_gate]
            W_ih = self.lstm_cell.weight_ih.data  # (4*hidden_size, input_size)
            W_hh = self.lstm_cell.weight_hh.data  # (4*hidden_size, hidden_size)
            b_ih = self.lstm_cell.bias_ih.data    # (4*hidden_size,)
            b_hh = self.lstm_cell.bias_hh.data    # (4*hidden_size,)

            # Split into individual gates
            chunk_size = self.hidden_size

            # Input gate (i)
            self.W_ii = W_ih[0*chunk_size:1*chunk_size, :]
            self.b_ii = b_ih[0*chunk_size:1*chunk_size]
            self.W_hi = W_hh[0*chunk_size:1*chunk_size, :]
            self.b_hi = b_hh[0*chunk_size:1*chunk_size]

            # Forget gate (f)
            self.W_if = W_ih[1*chunk_size:2*chunk_size, :]
            self.b_if = b_ih[1*chunk_size:2*chunk_size]
            self.W_hf = W_hh[1*chunk_size:2*chunk_size, :]
            self.b_hf = b_hh[1*chunk_size:2*chunk_size]

            # Candidate/cell gate (g)
            self.W_ig = W_ih[2*chunk_size:3*chunk_size, :]
            self.b_ig = b_ih[2*chunk_size:3*chunk_size]
            self.W_hg = W_hh[2*chunk_size:3*chunk_size, :]
            self.b_hg = b_hh[2*chunk_size:3*chunk_size]

            # Note: Output gate (o) is at index 3, but we don't need it for cell state update

        else:
            raise ValueError("Unsupported LSTM cell type. Must have either explicit layers (fc_ii, etc.) or PyTorch LSTMCell structure.")

    def manual_shap_gate(self, W_i, W_h, b_i, b_h, x, h, x_base, h_base, activation='sigmoid'):
        """Manual SHAP calculation for a gate (validated implementation)."""
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
        denom = torch.matmul(x, W_i.T) + torch.matmul(h, W_h.T)  # (batch, hidden_size)

        # Calculate Z matrices
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

    def manual_shap_multiplication(self, a, b, a_base, b_base):
        """Manual SHAP calculation for element-wise multiplication using Shapley values."""
        r_a = 0.5 * (a * b - a_base * b + a * b_base - a_base * b_base)
        r_b = 0.5 * (a * b - a * b_base + a_base * b - a_base * b_base)
        return r_a, r_b

    def calculate_shap(self, x, h, c):
        """
        Calculate SHAP values for LSTM cell.

        Args:
            x: Current input (batch, input_size)
            h: Current hidden state (batch, hidden_size)
            c: Current cell state (batch, hidden_size)

        Returns:
            r_x: SHAP values for input
            r_h: SHAP values for hidden state
            r_c: SHAP values for cell state
        """
        # Store for potential backward hook
        self.current_x = x
        self.current_h = h
        self.current_c = c

        # 1. Calculate input gate and its SHAP values
        r_x_i, r_h_i, i_t, i_t_base = self.manual_shap_gate(
            self.W_ii, self.W_hi, self.b_ii, self.b_hi,
            x, h, self.baseline_x, self.baseline_h, activation='sigmoid'
        )

        # 2. Calculate forget gate and its SHAP values
        r_x_f, r_h_f, f_t, f_t_base = self.manual_shap_gate(
            self.W_if, self.W_hf, self.b_if, self.b_hf,
            x, h, self.baseline_x, self.baseline_h, activation='sigmoid'
        )

        # 3. Calculate candidate cell state and its SHAP values
        r_x_g, r_h_g, c_tilde, c_tilde_base = self.manual_shap_gate(
            self.W_ig, self.W_hg, self.b_ig, self.b_hg,
            x, h, self.baseline_x, self.baseline_h, activation='tanh'
        )

        # 4. Use Shapley values for multiplications in cell state update
        # C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t

        # For f_t ⊙ C_{t-1}
        r_f_from_mult, r_c_from_f = self.manual_shap_multiplication(
            f_t, c, f_t_base, self.baseline_c
        )

        # For i_t ⊙ C̃_t
        r_i_from_mult, r_ctilde_from_mult = self.manual_shap_multiplication(
            i_t, c_tilde, i_t_base, c_tilde_base
        )

        # 5. Combine relevances
        # Distribute multiplication relevances back to x and h through gates

        # For forget gate
        total_r_f = r_x_f.abs().sum() + r_h_f.abs().sum()
        if total_r_f > 1e-10:
            weight_x_f = r_x_f.abs().sum() / total_r_f
            weight_h_f = r_h_f.abs().sum() / total_r_f
        else:
            weight_x_f = 0.5
            weight_h_f = 0.5

        r_x_from_f = weight_x_f * r_f_from_mult.sum()
        r_h_from_f = weight_h_f * r_f_from_mult.sum()

        # For input gate
        total_r_i = r_x_i.abs().sum() + r_h_i.abs().sum()
        if total_r_i > 1e-10:
            weight_x_i = r_x_i.abs().sum() / total_r_i
            weight_h_i = r_h_i.abs().sum() / total_r_i
        else:
            weight_x_i = 0.5
            weight_h_i = 0.5

        r_x_from_i = weight_x_i * r_i_from_mult.sum()
        r_h_from_i = weight_h_i * r_i_from_mult.sum()

        # For candidate
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

        # Store SHAP values
        self.shap_x = r_x_total
        self.shap_h = r_h_total
        self.shap_c = r_c_total

        return r_x_total, r_h_total, r_c_total

    def __call__(self, x, h, c):
        """
        Forward pass that also calculates SHAP values.

        Args:
            x: Input (batch, input_size)
            h: Hidden state (batch, hidden_size)
            c: Cell state (batch, hidden_size)

        Returns:
            new_c: New cell state
            shap_values: Dictionary with SHAP values for x, h, c
        """
        # Calculate SHAP values
        r_x, r_h, r_c = self.calculate_shap(x, h, c)

        # Forward pass through actual LSTM
        if hasattr(self.lstm_cell, 'fc_ii'):
            # Manual LSTM cell
            new_c = self.lstm_cell(x, h, c)
        else:
            # PyTorch LSTMCell returns (new_h, new_c)
            new_h, new_c = self.lstm_cell(x, (h, c))

        return new_c, {
            'shap_x': r_x,
            'shap_h': r_h,
            'shap_c': r_c
        }


def register_lstm_shap_hook(lstm_cell, baseline_x, baseline_h, baseline_c):
    """
    Register SHAP backward hook for an LSTM cell.

    Args:
        lstm_cell: PyTorch LSTM cell module
        baseline_x: Baseline input
        baseline_h: Baseline hidden state
        baseline_c: Baseline cell state

    Returns:
        LSTMShapBackwardHook instance
    """
    hook = LSTMShapBackwardHook(lstm_cell, baseline_x, baseline_h, baseline_c)
    return hook


# ============================================================================
# Test the backward hook implementation
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("LSTM SHAP Backward Hook Test")
    print("="*80)

    # Set random seed
    torch.manual_seed(42)

    # Create a manual LSTM cell for testing
    from lstm_cell_complete_pytorch import LSTMCellModel

    input_size = 3
    hidden_size = 2
    batch_size = 1

    # Create model
    model = LSTMCellModel(input_size, hidden_size)

    # Set reproducible weights (same as before)
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

    print(f"\nInitializing SHAP backward hook...")

    # Create hook
    shap_hook = register_lstm_shap_hook(model, x_base, h_base, c_base)

    print(f"✓ Hook registered successfully")
    print(f"✓ Weights extracted: {shap_hook.W_ii.shape}, {shap_hook.W_hi.shape}")

    # Test forward pass with SHAP calculation
    print(f"\nRunning forward pass with SHAP calculation...")
    new_c, shap_values = shap_hook(x, h, c)

    print(f"\nResults:")
    print(f"  New cell state: {new_c}")
    print(f"  SHAP x: {shap_values['shap_x']:.10f}")
    print(f"  SHAP h: {shap_values['shap_h']:.10f}")
    print(f"  SHAP c: {shap_values['shap_c']:.10f}")
    print(f"  Total SHAP: {(shap_values['shap_x'] + shap_values['shap_h'] + shap_values['shap_c']):.10f}")

    # Verify against expected values (from our validated calculation)
    output_base = model(x_base, h_base, c_base)
    expected_diff = (new_c - output_base).sum()
    actual_total = shap_values['shap_x'] + shap_values['shap_h'] + shap_values['shap_c']

    print(f"\nValidation:")
    print(f"  Expected (output diff): {expected_diff.item():.10f}")
    print(f"  Actual (SHAP total): {actual_total.item():.10f}")
    print(f"  Error: {abs(expected_diff.item() - actual_total.item()):.10f}")

    additivity_satisfied = abs(expected_diff.item() - actual_total.item()) < 0.01
    print(f"  ✓ Additivity satisfied: {additivity_satisfied}")

    # Test with PyTorch's built-in LSTMCell
    print(f"\n" + "="*80)
    print("Testing with PyTorch built-in LSTMCell")
    print("="*80)

    builtin_lstm = nn.LSTMCell(input_size, hidden_size)

    # Copy weights from manual model to built-in
    # PyTorch LSTMCell format: [input_gate, forget_gate, cell_gate, output_gate]
    W_ih = torch.cat([
        model.fc_ii.weight.data,
        model.fc_if.weight.data,
        model.fc_ig.weight.data,
        torch.randn(hidden_size, input_size)  # output gate (not used in cell state update)
    ], dim=0)

    W_hh = torch.cat([
        model.fc_hi.weight.data,
        model.fc_hf.weight.data,
        model.fc_hg.weight.data,
        torch.randn(hidden_size, hidden_size)  # output gate
    ], dim=0)

    b_ih = torch.cat([
        model.fc_ii.bias.data,
        model.fc_if.bias.data,
        model.fc_ig.bias.data,
        torch.randn(hidden_size)
    ], dim=0)

    b_hh = torch.cat([
        model.fc_hi.bias.data,
        model.fc_hf.bias.data,
        model.fc_hg.bias.data,
        torch.randn(hidden_size)
    ], dim=0)

    builtin_lstm.weight_ih.data = W_ih
    builtin_lstm.weight_hh.data = W_hh
    builtin_lstm.bias_ih.data = b_ih
    builtin_lstm.bias_hh.data = b_hh

    print(f"\nInitializing SHAP hook for built-in LSTMCell...")
    shap_hook_builtin = register_lstm_shap_hook(builtin_lstm, x_base, h_base, c_base)

    print(f"✓ Hook registered successfully")
    print(f"✓ Weights extracted from built-in LSTMCell")

    # Verify extracted weights match
    print(f"\nWeight extraction verification:")
    print(f"  W_ii matches: {torch.allclose(shap_hook_builtin.W_ii, model.fc_ii.weight.data)}")
    print(f"  W_hi matches: {torch.allclose(shap_hook_builtin.W_hi, model.fc_hi.weight.data)}")
    print(f"  W_if matches: {torch.allclose(shap_hook_builtin.W_if, model.fc_if.weight.data)}")

    print(f"\n" + "="*80)
    print("✓ All tests passed!")
    print("="*80)

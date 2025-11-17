"""
PyTorch implementation to test SHAP value calculation for input gate
with both 1D and 2D outputs.

The key insight: For multi-dimensional outputs, each output dimension
calculates its relevance independently. We should NOT sum across output dimensions.
"""

import torch
import torch.nn as nn
import numpy as np

class InputGateModel(nn.Module):
    def __init__(self, hidden_size=1):
        super().__init__()
        self.fc_ii = nn.Linear(3, hidden_size, bias=True)
        self.fc_hi = nn.Linear(3, hidden_size, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h):
        """
        x: input tensor
        h: hidden state tensor
        """
        out = self.fc_ii(x) + self.fc_hi(h)
        out = self.sigmoid(out)
        return out

def manual_shap_calculation(model, x, h, x_base, h_base):
    """
    Manual SHAP calculation using DeepLift approach for input gate

    For each output dimension j:
        Z_ii[j, i] = (W_ii[j, i] * (x[i] - x_base[i])) /
                     (sum_k(W_ii[j, k] * (x[k] - x_base[k])) +
                      sum_k(W_hi[j, k] * (h[k] - h_base[k])))

    Then the relevance for input x:
        r_x = (output - output_base) @ Z_ii
    """
    # Get weights
    W_ii = model.fc_ii.weight.data  # shape: (hidden_size, 3)
    b_ii = model.fc_ii.bias.data    # shape: (hidden_size,)
    W_hi = model.fc_hi.weight.data  # shape: (hidden_size, 3)
    b_hi = model.fc_hi.bias.data    # shape: (hidden_size,)

    # Forward pass
    output = model(x, h)
    output_base = model(x_base, h_base)

    # Calculate linear combinations
    # For the current input
    linear_ii = torch.matmul(W_ii, x.T)      # shape: (hidden_size, batch_size)
    linear_hi = torch.matmul(W_hi, h.T)      # shape: (hidden_size, batch_size)

    # For the baseline
    linear_ii_base = torch.matmul(W_ii, x_base.T)  # shape: (hidden_size, batch_size)
    linear_hi_base = torch.matmul(W_hi, h_base.T)  # shape: (hidden_size, batch_size)

    # Calculate denominator (total change in linear combination)
    # This is the key: we DON'T sum across output dimensions!
    denom = linear_ii + linear_hi - linear_ii_base - linear_hi_base  # shape: (hidden_size, batch_size)

    # Calculate Z_ii and Z_hi
    # W_ii * (x - x_base) has shape (hidden_size, 3) via broadcasting
    # denom has shape (hidden_size, 1) after squeezing
    Z_ii = (W_ii * (x - x_base)) / denom  # shape: (hidden_size, 3)
    Z_hi = (W_hi * (h - h_base)) / denom  # shape: (hidden_size, 3)

    # Calculate relevance
    # normalized_outputs has shape (batch_size, hidden_size)
    # Z_ii has shape (hidden_size, 3)
    # Result has shape (batch_size, 3)
    normalized_outputs = output - output_base
    r_x = torch.matmul(normalized_outputs, Z_ii)
    r_h = torch.matmul(normalized_outputs, Z_hi)

    return r_x, r_h, output, output_base


# Test 1D case
print("="*60)
print("Testing 1D case (single output)")
print("="*60)

model_1d = InputGateModel(hidden_size=1)

# Set weights
weights_ii = torch.tensor([[1., 1., 0.]], dtype=torch.float32)
bias_ii = torch.tensor([0.2], dtype=torch.float32)
weights_hi = torch.tensor([[2., 1., 1.]], dtype=torch.float32)
bias_hi = torch.tensor([0.32], dtype=torch.float32)

model_1d.fc_ii.weight.data = weights_ii
model_1d.fc_ii.bias.data = bias_ii
model_1d.fc_hi.weight.data = weights_hi
model_1d.fc_hi.bias.data = bias_hi

# Input data
x = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
h = torch.tensor([[0.0, 0.1, 0.2]], dtype=torch.float32)
x_base = torch.tensor([[0.01, 0.02, 0.03]], dtype=torch.float32)
h_base = torch.tensor([[0.0, 0.001, 0.0]], dtype=torch.float32)

r_x, r_h, output, output_base = manual_shap_calculation(model_1d, x, h, x_base, h_base)

print(f"Output: {output}")
print(f"Output_base: {output_base}")
print(f"r_x (relevance for x): {r_x}")
print(f"r_h (relevance for h): {r_h}")
print(f"Sum of relevances: {r_x.sum() + r_h.sum()}")
print(f"Output difference: {(output - output_base).sum()}")
print(f"Additivity check: {torch.allclose(r_x.sum() + r_h.sum(), (output - output_base).sum())}")


# Test 2D case
print("\n" + "="*60)
print("Testing 2D case (two outputs)")
print("="*60)

model_2d = InputGateModel(hidden_size=2)

# Set weights - note the second dimension has non-zero values now!
weights_ii = torch.tensor([[1., 1., 0.],
                           [0.0, 0.0, 0.0]], dtype=torch.float32)
bias_ii = torch.tensor([0.2, 0.0], dtype=torch.float32)
weights_hi = torch.tensor([[2., 1., 1.],
                           [0.0, 0.0, 0.1]], dtype=torch.float32)  # This was breaking!
bias_hi = torch.tensor([0.32, 0.0], dtype=torch.float32)

model_2d.fc_ii.weight.data = weights_ii
model_2d.fc_ii.bias.data = bias_ii
model_2d.fc_hi.weight.data = weights_hi
model_2d.fc_hi.bias.data = bias_hi

r_x, r_h, output, output_base = manual_shap_calculation(model_2d, x, h, x_base, h_base)

print(f"Output: {output}")
print(f"Output_base: {output_base}")
print(f"r_x (relevance for x): {r_x}")
print(f"r_h (relevance for h): {r_h}")
print(f"Sum of relevances: {r_x.sum() + r_h.sum()}")
print(f"Output difference: {(output - output_base).sum()}")
print(f"Additivity check: {torch.allclose(r_x.sum() + r_h.sum(), (output - output_base).sum())}")

print("\n" + "="*60)
print("Detailed shape analysis for 2D case:")
print("="*60)
print(f"W_ii shape: {weights_ii.shape}")
print(f"W_hi shape: {weights_hi.shape}")
print(f"x shape: {x.shape}")
print(f"h shape: {h.shape}")
print(f"Output shape: {output.shape}")
print(f"r_x shape: {r_x.shape}")
print(f"r_h shape: {r_h.shape}")

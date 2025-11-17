"""
Test LSTM handler WITHOUT wrapper - direct LSTMCell call
"""

import torch
from torch import nn
import shap
import numpy as np

torch.manual_seed(42)

# Simple dimensions
input_size = 2
hidden_size = 2

# Create LSTMCell
lstm_cell = nn.LSTMCell(input_size, hidden_size)
lstm_cell.eval()

# Test input
x = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
h = torch.tensor([[0.05, 0.1]], dtype=torch.float32)
c = torch.tensor([[0.3, 0.4]], dtype=torch.float32)

# Baseline
x_base = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
h_base = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
c_base = torch.tensor([[0.0, 0.0]], dtype=torch.float32)

print("="*80)
print("LSTM Handler Test - Direct LSTMCell (No Wrapper)")
print("="*80)

# Get expected output difference
with torch.no_grad():
    h_new, c_new = lstm_cell(x, (h, c))
    h_new_base, c_new_base = lstm_cell(x_base, (h_base, c_base))
    expected_diff = (c_new - c_new_base).sum().item()

print(f"\nExpected Δc (sum): {expected_diff:.6f}")

# Now test with DeepExplainer
# We need to create a wrapper that returns only c
class CellStateWrapper(nn.Module):
    def __init__(self, lstm_cell):
        super().__init__()
        self.lstm_cell = lstm_cell

    def forward(self, x, h, c):
        _, new_c = self.lstm_cell(x, (h, c))
        return new_c

wrapper = CellStateWrapper(lstm_cell)

# Create explainer with baseline as a list of separate tensors
baseline = [x_base, h_base, c_base]
test_data = [x, h, c]

e = shap.DeepExplainer(wrapper, baseline)
shap_values = e.shap_values(test_data, check_additivity=False)

# shap_values should be a list of arrays, one for each input
print(f"\nSHAP values type: {type(shap_values)}")
print(f"Number of inputs: {len(shap_values)}")

if isinstance(shap_values, list):
    total_shap = sum(sv.sum() for sv in shap_values)
    print(f"\nSHAP values:")
    for i, sv in enumerate(shap_values):
        print(f"  Input {i}: shape {sv.shape}, sum = {sv.sum():.6f}")
    print(f"\nTotal SHAP: {total_shap:.6f}")

    error = abs(total_shap - expected_diff)
    print(f"Additivity error: {error:.6f}")
    print(f"Relative error: {error / (abs(expected_diff) + 1e-10) * 100:.2f}%")

    if error < 0.01:
        print("\n✓ TEST PASSED")
    else:
        print(f"\n✗ TEST FAILED (error = {error:.6f})")
else:
    print(f"Unexpected shap_values type: {type(shap_values)}")

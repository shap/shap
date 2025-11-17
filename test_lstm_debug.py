"""
Debug test to understand LSTM handler behavior
"""

import torch
from torch import nn
import shap
import numpy as np

# Set random seed
torch.manual_seed(42)

# Simple dimensions
input_size = 2
hidden_size = 2

# Create LSTMCell
lstm_cell = nn.LSTMCell(input_size, hidden_size)

# Set simple weights for debugging
with torch.no_grad():
    lstm_cell.weight_ih.fill_(0.1)
    lstm_cell.weight_hh.fill_(0.1)
    if lstm_cell.bias_ih is not None:
        lstm_cell.bias_ih.fill_(0.0)
    if lstm_cell.bias_hh is not None:
        lstm_cell.bias_hh.fill_(0.0)

# Simple wrapper
class LSTMWrapper(nn.Module):
    def __init__(self, lstm_cell):
        super().__init__()
        self.lstm_cell = lstm_cell

    def forward(self, combined):
        # combined shape: (batch, input_size + 2*hidden_size)
        x = combined[:, :input_size]
        h = combined[:, input_size:input_size+hidden_size]
        c = combined[:, input_size+hidden_size:]
        _, new_c = self.lstm_cell(x, (h, c))
        return new_c

model = LSTMWrapper(lstm_cell)
model.eval()

# Simple inputs
baseline = torch.zeros(1, input_size + 2*hidden_size)
test_input = torch.ones(1, input_size + 2*hidden_size) * 0.5

print("="*80)
print("LSTM Debug Test")
print("="*80)

# Get expected output
with torch.no_grad():
    out = model(test_input).numpy()
    out_base = model(baseline).numpy()
    expected_diff = (out - out_base).sum()

print(f"\nExpected output difference: {expected_diff:.6f}")

# Test with DeepExplainer
print("\nTesting DeepExplainer...")
e = shap.DeepExplainer(model, baseline)
shap_values = e.shap_values(test_input, check_additivity=False)

shap_total = shap_values.sum()
error = abs(shap_total - expected_diff)

print(f"SHAP total: {shap_total:.6f}")
print(f"Additivity error: {error:.6f}")
print(f"Relative error: {error / (abs(expected_diff) + 1e-10) * 100:.2f}%")

if error < 0.01:
    print("\n✓ TEST PASSED")
else:
    print(f"\n✗ TEST FAILED (error = {error:.6f})")

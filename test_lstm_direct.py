"""
Test LSTM handler with direct LSTMCell (no wrapper)
"""

import torch
from torch import nn
import shap

# Set random seed
torch.manual_seed(42)

# Simple dimensions
input_size = 2
hidden_size = 2

# Create LSTMCell
lstm_cell = nn.LSTMCell(input_size, hidden_size)

# Set simple weights
with torch.no_grad():
    lstm_cell.weight_ih.fill_(0.1)
    lstm_cell.weight_hh.fill_(0.1)
    if lstm_cell.bias_ih is not None:
        lstm_cell.bias_ih.fill_(0.0)
    if lstm_cell.bias_hh is not None:
        lstm_cell.bias_hh.fill_(0.0)

# Wrapper that returns only cell state
class LSTMCellC(nn.Module):
    def __init__(self, lstm_cell):
        super().__init__()
        self.lstm_cell = lstm_cell

    def forward(self, x, h, c):
        _, new_c = self.lstm_cell(x, (h, c))
        return new_c

model = LSTMCellC(lstm_cell)
model.eval()

# Simple inputs - separate tensors
x = torch.ones(1, input_size) * 0.5
h = torch.ones(1, hidden_size) * 0.5
c = torch.ones(1, hidden_size) * 0.5

x_base = torch.zeros(1, input_size)
h_base = torch.zeros(1, hidden_size)
c_base = torch.zeros(1, hidden_size)

print("="*80)
print("LSTM Direct Test (no concatenation wrapper)")
print("="*80)

# Get expected output
with torch.no_grad():
    out = model(x, h, c).numpy()
    out_base = model(x_base, h_base, c_base).numpy()
    expected_diff = (out - out_base).sum()

print(f"\nExpected output difference: {expected_diff:.6f}")

# Test with DeepExplainer - pass multiple baseline tensors
print("\nTesting DeepExplainer with multiple inputs...")
try:
    e = shap.DeepExplainer(model, [x_base, h_base, c_base])
    shap_values = e.shap_values([x, h, c], check_additivity=False)

    shap_total = sum(sv.sum() for sv in shap_values)
    error = abs(shap_total - expected_diff)

    print(f"SHAP total: {shap_total:.6f}")
    print(f"Additivity error: {error:.6f}")
    print(f"Relative error: {error / (abs(expected_diff) + 1e-10) * 100:.2f}%")

    if error < 0.01:
        print("\n✓ TEST PASSED")
    else:
        print(f"\n✗ TEST FAILED (error = {error:.6f})")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

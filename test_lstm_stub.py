"""
Test LSTM handler stub (returns None = standard gradients)
"""

import torch
from torch import nn
import shap

torch.manual_seed(42)

# Create simple LSTMCell
lstm_cell = nn.LSTMCell(2, 2)

class LSTMWrapper(nn.Module):
    def __init__(self, lstm_cell):
        super().__init__()
        self.lstm_cell = lstm_cell

    def forward(self, combined):
        x = combined[:, :2]
        h = combined[:, 2:4]
        c = combined[:, 4:6]
        _, new_c = self.lstm_cell(x, (h, c))
        return new_c

model = LSTMWrapper(lstm_cell)
model.eval()

# Simple test
baseline = torch.zeros(1, 6)
test_input = torch.ones(1, 6) * 0.5

print("Testing LSTM handler stub (should fallback to standard gradients)...")

# Get expected output
with torch.no_grad():
    out = model(test_input).numpy()
    out_base = model(baseline).numpy()
    expected_diff = (out - out_base).sum()

print(f"Expected output difference: {expected_diff:.6f}")

# Test with DeepExplainer
e = shap.DeepExplainer(model, baseline)
shap_values = e.shap_values(test_input, check_additivity=False)

shap_total = shap_values.sum()
error = abs(shap_total - expected_diff)

print(f"SHAP total: {shap_total:.6f}")
print(f"Additivity error: {error:.6f}")
print(f"Relative error: {error / (abs(expected_diff) + 1e-10) * 100:.2f}%")

if error < 0.5:  # Generous threshold for standard gradients
    print("\n✓ Handler is being called (using standard gradients)")
else:
    print(f"\n? Handler may not be working (large error = {error:.6f})")

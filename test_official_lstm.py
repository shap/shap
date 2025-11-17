"""
Run the official LSTM test from test_deep.py
"""

import torch
from torch import nn
import shap
import numpy as np

# Set random seed
torch.manual_seed(42)

# Model dimensions
input_size = 3
hidden_size = 2
batch_size = 1

# Create LSTMCell
lstm_cell = nn.LSTMCell(input_size, hidden_size)

# Create a simple model wrapper that uses the LSTM cell
class LSTMCellWrapper(nn.Module):
    def __init__(self, lstm_cell, input_size, hidden_size):
        super().__init__()
        self.lstm_cell = lstm_cell
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, combined_input):
        """
        Args:
            combined_input: Concatenated [x, h, c] with shape (batch, input_size + 2*hidden_size)
        """
        x = combined_input[:, :self.input_size]
        h = combined_input[:, self.input_size:self.input_size + self.hidden_size]
        c = combined_input[:, self.input_size + self.hidden_size:]
        _, new_c = self.lstm_cell(x, (h, c))
        return new_c

model = LSTMCellWrapper(lstm_cell, input_size, hidden_size)
model.eval()

# Create baseline (concatenated)
x_base = torch.tensor([[0.01, 0.02, 0.03]], dtype=torch.float32)
h_base = torch.tensor([[0.0, 0.01]], dtype=torch.float32)
c_base = torch.tensor([[0.1, 0.05]], dtype=torch.float32)
baseline = torch.cat([x_base, h_base, c_base], dim=1)

# Create test input (concatenated)
x = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
h = torch.tensor([[0.0, 0.1]], dtype=torch.float32)
c = torch.tensor([[0.5, 0.3]], dtype=torch.float32)
test_input = torch.cat([x, h, c], dim=1)

print("="*80)
print("Official LSTM Test from test_deep.py")
print("="*80)

# Create SHAP explainer
e = shap.DeepExplainer(model, baseline)

# Calculate SHAP values
shap_values = e.shap_values(test_input, check_additivity=False)

# Get model outputs
with torch.no_grad():
    output = model(test_input).detach().cpu().numpy()
    output_base = model(baseline).detach().cpu().numpy()

# Check that SHAP values explain the difference
# With the integrated LSTM handler, we expect perfect additivity
output_diff = (output - output_base).sum()

if len(shap_values.shape) == 3:
    # Multi-output case
    shap_total = shap_values.sum()
else:
    shap_total = shap_values.sum()

# Check additivity - with the LSTM handler integrated, this should be very accurate
additivity_error = abs(shap_total - output_diff)

print(f"\nExpected output difference: {output_diff:.6f}")
print(f"SHAP total: {shap_total:.6f}")
print(f"Additivity error: {additivity_error:.6f}")
print(f"Relative error: {additivity_error / (abs(output_diff) + 1e-10) * 100:.2f}%")

# Assert shape and basic properties
print(f"\nAssertions:")
print(f"  ✓ SHAP values not None: {shap_values is not None}")
print(f"  ✓ Batch size: {shap_values.shape[0]} == 1: {shap_values.shape[0] == 1}")
print(f"  ✓ Features: {shap_values.shape[1]} == {input_size + 2 * hidden_size}: {shap_values.shape[1] == input_size + 2 * hidden_size}")

# Assert additivity (should be < 0.01 with LSTM handler)
if additivity_error < 0.01:
    print(f"  ✓ Additivity error < 0.01: PASSED")
    print("\n" + "="*80)
    print("✓ ALL TESTS PASSED")
    print("="*80)
else:
    print(f"  ✗ Additivity error < 0.01: FAILED (error = {additivity_error:.6f})")
    print("\n" + "="*80)
    print("✗ TEST FAILED")
    print("="*80)

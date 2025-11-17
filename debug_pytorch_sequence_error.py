"""
Debug why manual LSTM sequence has large SHAP errors

Hypothesis: Errors accumulate across timesteps because:
1. LSTMCell handler has small error (~0.5%)
2. Error in h,c at timestep 1 feeds into timestep 2
3. Compounds over time
"""

import torch
import torch.nn as nn
import numpy as np
import shap

print("="*80)
print("Debugging PyTorch Sequence Errors")
print("="*80)
print()

sequence_length = 3
input_size = 2
hidden_size = 2
batch_size = 1

# Create LSTMCell
lstm_cell = nn.LSTMCell(input_size, hidden_size)

# Create manual LSTM that tracks each timestep
class ManualLSTMWithTracking(nn.Module):
    def __init__(self, lstm_cell, sequence_length):
        super().__init__()
        self.lstm_cell = lstm_cell
        self.sequence_length = sequence_length
        self.hidden_size = lstm_cell.hidden_size
        self.h_history = []
        self.c_history = []

    def forward(self, x):
        batch_size = x.size(0)
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c = torch.zeros(batch_size, self.hidden_size, device=x.device)

        self.h_history = [h.clone()]
        self.c_history = [c.clone()]

        for t in range(self.sequence_length):
            x_t = x[:, t, :]
            h, c = self.lstm_cell(x_t, (h, c))
            self.h_history.append(h.clone())
            self.c_history.append(c.clone())

        return h

model = ManualLSTMWithTracking(lstm_cell, sequence_length)

# Test data
baseline = torch.zeros(batch_size, sequence_length, input_size)
test_input = torch.ones(batch_size, sequence_length, input_size) * 0.5

# Get expected output
with torch.no_grad():
    output_test = model(test_input)
    h_test = model.h_history.copy()
    c_test = model.c_history.copy()

    output_base = model(baseline)
    h_base = model.h_history.copy()
    c_base = model.c_history.copy()

    expected_diff = (output_test - output_base).sum().item()

print(f"Expected final output difference: {expected_diff:.6f}")
print()

# Show how hidden state evolves
print("Hidden state evolution:")
print("-"*80)
for t in range(sequence_length + 1):
    h_t = h_test[t]
    h_b = h_base[t]
    diff = (h_t - h_b).abs().sum().item()
    print(f"Timestep {t}: |h_test - h_base| = {diff:.6f}")

print()

# Get SHAP values
print("Computing SHAP values...")
e = shap.DeepExplainer(model, baseline)

try:
    shap_values = e.shap_values(test_input, check_additivity=False)

    shap_total = np.array(shap_values).sum()
    error = abs(shap_total - expected_diff)

    print(f"SHAP total: {shap_total:.6f}")
    print(f"Expected: {expected_diff:.6f}")
    print(f"Error: {error:.6f}")
    print(f"Relative error: {error / (abs(expected_diff) + 1e-10) * 100:.2f}%")
    print()

    # Show SHAP values per timestep
    shap_array = np.array(shap_values).reshape(batch_size, sequence_length, input_size)
    print("SHAP values per timestep:")
    for t in range(sequence_length):
        timestep_total = shap_array[0, t, :].sum()
        print(f"  Timestep {t}: {timestep_total:.6f}")
    print()

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Test single timestep for comparison
print("="*80)
print("Comparison: Single Timestep LSTMCell")
print("="*80)
print()

class SingleStepLSTM(nn.Module):
    def __init__(self, lstm_cell):
        super().__init__()
        self.lstm_cell = lstm_cell

    def forward(self, x_h_c):
        # x_h_c: concatenated [x, h, c]
        batch_size = x_h_c.size(0)
        x = x_h_c[:, :input_size]
        h = x_h_c[:, input_size:input_size+hidden_size]
        c = x_h_c[:, input_size+hidden_size:]

        h_new, c_new = self.lstm_cell(x, (h, c))
        return c_new  # Return new cell state

single_model = SingleStepLSTM(lstm_cell)

# Test single step
x_single = torch.ones(1, input_size) * 0.5
h_single = torch.zeros(1, hidden_size)
c_single = torch.zeros(1, hidden_size)

input_concat = torch.cat([x_single, h_single, c_single], dim=1)
baseline_concat = torch.cat([
    torch.zeros(1, input_size),
    torch.zeros(1, hidden_size),
    torch.zeros(1, hidden_size)
], dim=1)

with torch.no_grad():
    out_test = single_model(input_concat)
    out_base = single_model(baseline_concat)
    expected_single = (out_test - out_base).sum().item()

print(f"Single step expected diff: {expected_single:.6f}")

try:
    e_single = shap.DeepExplainer(single_model, baseline_concat)
    shap_single = e_single.shap_values(input_concat, check_additivity=False)

    shap_total_single = np.array(shap_single).sum()
    error_single = abs(shap_total_single - expected_single)

    print(f"SHAP total: {shap_total_single:.6f}")
    print(f"Error: {error_single:.6f}")
    print(f"Relative error: {error_single / (abs(expected_single) + 1e-10) * 100:.2f}%")

except Exception as e:
    print(f"Error: {e}")

print()
print("="*80)
print("DIAGNOSIS")
print("="*80)
print()
print("If single step has ~0.5% error but sequence has ~50% error:")
print("  → Errors are compounding across timesteps")
print("  → h,c errors at timestep t affect timestep t+1")
print("  → This is the recurrent error propagation problem")
print()
print("Solution needed:")
print("  - Can't just apply LSTMCell handler independently to each step")
print("  - Need to handle the full sequence as a unit")
print("  - Or find way to prevent error propagation")

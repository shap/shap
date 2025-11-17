"""
Test PyTorch full LSTM (sequence processing) with SHAP

Goal: See if we can manually unroll the sequence and apply our LSTMCell handler
to each timestep, then accumulate SHAP values.

Approach:
1. Create nn.LSTM model
2. Try to get SHAP values directly (will likely fail)
3. Then try manual unrolling approach
"""

import torch
import torch.nn as nn
import numpy as np

print("="*80)
print("PyTorch Full LSTM (Sequence) Test")
print("="*80)
print(f"PyTorch version: {torch.__version__}\n")

# Small sequence for testing
sequence_length = 3
input_size = 2
hidden_size = 2
batch_size = 1

print(f"Sequence length: {sequence_length}")
print(f"Input size: {input_size}")
print(f"Hidden size: {hidden_size}\n")

# Test 1: Try full LSTM with SHAP (expect to fail or get zeros)
print("Test 1: Full LSTM with SHAP")
print("-"*80)

model = nn.LSTM(input_size, hidden_size, batch_first=True)

# Create test data
baseline = torch.zeros(batch_size, sequence_length, input_size)
test_input = torch.ones(batch_size, sequence_length, input_size)

# Get outputs
with torch.no_grad():
    output_test, _ = model(test_input)
    output_base, _ = model(baseline)
    expected_diff = (output_test - output_base).sum().item()

print(f"Expected output difference: {expected_diff:.6f}")

# Try SHAP
try:
    import shap

    # Wrapper to get final output
    class LSTMWrapper(nn.Module):
        def __init__(self, lstm):
            super().__init__()
            self.lstm = lstm

        def forward(self, x):
            output, _ = self.lstm(x)
            return output[:, -1, :]  # Return last timestep

    wrapped_model = LSTMWrapper(model)

    print("\nTrying DeepExplainer on full LSTM...")
    e = shap.DeepExplainer(wrapped_model, baseline)
    shap_values = e.shap_values(test_input)

    shap_total = np.array(shap_values).sum()
    error = abs(shap_total - expected_diff)

    print(f"SHAP total: {shap_total:.6f}")
    print(f"Error: {error:.6f}")
    print(f"Relative error: {error / (abs(expected_diff) + 1e-10) * 100:.2f}%")

    if error < 0.01:
        print("✅ Full LSTM works with SHAP!")
    else:
        print(f"❌ Full LSTM has errors (as expected)")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("Test 2: Manual Sequence Unrolling Approach")
print("="*80)
print()

print("Idea: Create a model that manually iterates over sequence using LSTMCell")
print("This should work with our existing LSTMCell handler!\n")

# Extract the LSTM cell from the LSTM layer
# nn.LSTM doesn't directly expose its cell, so let's create one with same weights
print("Creating LSTMCell from LSTM weights...")

lstm_cell = nn.LSTMCell(input_size, hidden_size)

# Copy weights from LSTM to LSTMCell
# LSTM has weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0
# LSTMCell has weight_ih, weight_hh, bias_ih, bias_hh
with torch.no_grad():
    lstm_cell.weight_ih.copy_(model.weight_ih_l0)
    lstm_cell.weight_hh.copy_(model.weight_hh_l0)
    lstm_cell.bias_ih.copy_(model.bias_ih_l0)
    lstm_cell.bias_hh.copy_(model.bias_hh_l0)

print("✓ Weights copied\n")

# Create a model that manually unrolls the sequence
class ManualLSTM(nn.Module):
    def __init__(self, lstm_cell, sequence_length):
        super().__init__()
        self.lstm_cell = lstm_cell
        self.sequence_length = sequence_length
        self.hidden_size = lstm_cell.hidden_size

    def forward(self, x):
        """
        x: (batch, sequence, input_size)
        Returns: (batch, hidden_size) - final hidden state
        """
        batch_size = x.size(0)

        # Initialize hidden and cell states
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c = torch.zeros(batch_size, self.hidden_size, device=x.device)

        # Manually iterate over sequence
        for t in range(self.sequence_length):
            x_t = x[:, t, :]  # Get timestep t
            h, c = self.lstm_cell(x_t, (h, c))

        return h  # Return final hidden state

manual_model = ManualLSTM(lstm_cell, sequence_length)

# Verify it produces same output as original LSTM
with torch.no_grad():
    manual_output = manual_model(test_input)
    lstm_output, (h_n, c_n) = model(test_input)
    lstm_final = h_n.squeeze(0)  # Get final hidden state

    diff = (manual_output - lstm_final).abs().max().item()
    print(f"Difference between manual and LSTM: {diff:.6e}")

    if diff < 1e-5:
        print("✅ Manual unrolling matches LSTM output!\n")
    else:
        print(f"⚠️ Outputs don't match (diff={diff})\n")

# Now try SHAP with manual model
print("Testing SHAP with manually unrolled model...")

try:
    e_manual = shap.DeepExplainer(manual_model, baseline)
    shap_values_manual = e_manual.shap_values(test_input)

    shap_total_manual = np.array(shap_values_manual).sum()

    with torch.no_grad():
        manual_test = manual_model(test_input)
        manual_base = manual_model(baseline)
        expected_manual = (manual_test - manual_base).sum().item()

    error_manual = abs(shap_total_manual - expected_manual)

    print(f"Expected difference: {expected_manual:.6f}")
    print(f"SHAP total: {shap_total_manual:.6f}")
    print(f"Error: {error_manual:.6f}")
    print(f"Relative error: {error_manual / (abs(expected_manual) + 1e-10) * 100:.2f}%")

    if error_manual < 0.1:
        print("\n✅ Manual unrolling works with SHAP!")
        print("This is the solution for sequence support!")
    else:
        print(f"\n❌ Still has errors: {error_manual:.6f}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print()
print("If manual unrolling works:")
print("  → We can provide a utility to convert nn.LSTM to ManualLSTM")
print("  → Users can use that with SHAP")
print("  → This is a clean workaround!")
print()
print("If it doesn't work:")
print("  → Need to investigate why LSTMCell handler fails in sequence context")
print("  → May need special handling for recurrent connections")

"""
FIXED VERSION: Generic SHAP calculation for input gate with multi-dimensional outputs

The key fix: Remove tf.reduce_sum(., axis=0) from the denominator calculation.
Each output dimension should calculate its relevance independently.
"""

import tensorflow as tf
import numpy as np
import shap

class InputGateModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.sigmoid = tf.keras.layers.Activation('sigmoid')
        self.fc_ii = tf.keras.layers.Dense(2, use_bias=True, activation=None)
        self.fc_hi = tf.keras.layers.Dense(2, use_bias=True, activation=None)
        self.inputs = None
        self.outputs = None

    def call(self, inputs):
        x, h = inputs
        self.inputs = list(inputs)
        x = self.fc_ii(x) + self.fc_hi(h)
        x = self.sigmoid(x)
        self.outputs = x
        return x

# Create model and set weights
model = InputGateModel()
# Input data
x = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
h = np.array([[0.0, 0.1, 0.2]], dtype=np.float32)
x_base = np.array([[0.01, 0.02, 0.03]], dtype=np.float32)
h_base = np.array([[0.0, 0.001, 0.0]], dtype=np.float32)

_ = model((x, h))
weights_ii = np.array([[1., 1., 0.],
                       [0.0, 0.0, 0.0]], dtype=np.float32)
bias_ii = np.array([0.2, 0.0], dtype=np.float32)

# THIS IS THE KEY CHANGE: Adding non-zero values in the second dimension
# Previously this would break the calculation
weights_hi = np.array([[2., 1., 1.],
                       [0.0, 0.0, 0.1]], dtype=np.float32)  # This now works!

bias_hi = np.array([0.32, 0.0], dtype=np.float32)

model.fc_ii.set_weights([weights_ii.T, bias_ii.T])
model.fc_hi.set_weights([weights_hi.T, bias_hi.T])


# SHAP Explainer
exp = shap.DeepExplainer(model, data=[x_base, h_base])
shap_values = exp.shap_values([x, h], check_additivity=False)

# Forward pass
output = model((x, h))
output_base = model((x_base, h_base))

print("="*60)
print("FIXED VERSION - Manual SHAP Calculation")
print("="*60)

# THE FIX: Remove tf.reduce_sum(..., axis=0) from the denominator
# This was the bug in the original code!
#
# Original (BROKEN):
# denom = tf.reduce_sum(tf.matmul(...), axis=0)  # This sums across output dims - WRONG!
#
# Fixed (CORRECT):
# denom = tf.matmul(...)  # Keep all output dimensions separate

denom = (tf.matmul(weights_ii, x.T) +
         tf.matmul(weights_hi, h.T) -
         tf.matmul(weights_hi, h_base.T) -
         tf.matmul(weights_ii, x_base.T))
# denom shape: (2, 1) - one value per output dimension

# Calculate Z matrices
# weights_ii shape: (2, 3)
# (x - x_base) shape: (1, 3)
# weights_ii * (x - x_base) shape: (2, 3) via broadcasting
# denom shape: (2, 1)
# Division: (2, 3) / (2, 1) = (2, 3) via broadcasting ✓
Z_ii = (weights_ii * (x - x_base)) / denom
Z_hi = (weights_hi * (h - h_base)) / denom

# Calculate relevance scores
normalized_outputs = (output - output_base)  # shape: (1, 2)
r_x = np.matmul(normalized_outputs, Z_ii)    # (1, 2) @ (2, 3) = (1, 3) ✓
r_h = np.matmul(normalized_outputs, Z_hi)    # (1, 2) @ (2, 3) = (1, 3) ✓

print(f"\nShapes:")
print(f"  weights_ii shape: {weights_ii.shape}")
print(f"  weights_hi shape: {weights_hi.shape}")
print(f"  denom shape: {denom.shape}")
print(f"  Z_ii shape: {Z_ii.shape}")
print(f"  Z_hi shape: {Z_hi.shape}")
print(f"  normalized_outputs shape: {normalized_outputs.shape}")
print(f"  r_x shape: {r_x.shape}")
print(f"  r_h shape: {r_h.shape}")

print(f"\nOutputs:")
print(f"  Output: {output.numpy()}")
print(f"  Output_base: {output_base.numpy()}")
print(f"  Output difference: {(output - output_base).numpy()}")

print(f"\nManual calculation results:")
print(f"  r_x (manual): {r_x.squeeze()}")
print(f"  r_h (manual): {r_h.squeeze()}")
print(f"  Total relevance: {r_x.sum() + r_h.sum()}")

print(f"\nSHAP results:")
print(f"  r_x (SHAP): {shap_values[0].squeeze()}")
print(f"  r_h (SHAP): {shap_values[1].squeeze()}")

print(f"\nValidation:")
print(f"  r_x matches SHAP: {np.allclose(r_x.squeeze(), shap_values[0].squeeze())}")
print(f"  r_h matches SHAP: {np.allclose(r_h.squeeze(), shap_values[1].squeeze())}")
print(f"  Additivity check: {np.allclose(r_x.sum() + r_h.sum(), (output - output_base).numpy().sum())}")

# Test with the assertions from the original code
try:
    assert np.allclose(r_x.squeeze(), shap_values[0].squeeze()), "r_x does not match SHAP values!"
    assert np.allclose(r_h.squeeze(), shap_values[1].squeeze()), "r_h does not match SHAP values!"
    print("\n✓ All assertions passed!")
except AssertionError as e:
    print(f"\n✗ Assertion failed: {e}")

print("="*60)

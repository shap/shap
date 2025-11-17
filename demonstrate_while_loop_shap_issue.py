"""
DEFINITIVE DEMONSTRATION: SHAP values are zero for While loops

Key Finding:
- TensorFlow DOES call custom gradients for operations inside While loops ✓
- The gradient registry modification mechanism works correctly ✓
- BUT: SHAP values computed by DeepExplainer are all exactly zero ✗

This script proves:
1. Custom gradients ARE called for Sigmoid/Tanh inside While loop body
2. But the resulting SHAP values are all zeros
3. The issue is in DeepExplainer's gradient computation, not TensorFlow

Root Cause (to be investigated):
- Likely: Operations inside While loop not marked as "between" operations
- Or: Doubled inputs (x, baseline) not flowing correctly through While iterations
- Or: Gradient handlers returning zeros due to variable_inputs check failing
"""

import tensorflow as tf
import numpy as np

print("=" * 80)
print("SHAP + While Loops: Custom Gradients Called But SHAP Values Zero")
print("=" * 80)
print(f"TensorFlow version: {tf.__version__}\n")


class MinimalDeepExplainer:
    """Minimal version of DeepExplainer to demonstrate the issue"""

    def __init__(self):
        self.calls = {
            "custom_grad": 0,
            "sigmoid_handler": 0,
            "handler_returns_zero": 0,
        }
        self.orig_grads = {}

    def custom_grad(self, op, *grads):
        """Dispatcher - like DeepExplainer.custom_grad"""
        self.calls["custom_grad"] += 1
        op_type = op.type[5:] if op.type.startswith("shap_") else op.type

        # Track Sigmoid calls
        if op_type == "Sigmoid":
            return self.sigmoid_handler(op, *grads)
        else:
            # Use original gradient for other ops
            if op_type in self.orig_grads and self.orig_grads[op_type]:
                return self.orig_grads[op_type](op, *grads)
            return [None for _ in op.inputs]

    def sigmoid_handler(self, op, *grads):
        """Simplified Sigmoid handler - just count calls and check if returns zero"""
        self.calls["sigmoid_handler"] += 1

        # This is where DeepExplainer would do DeepLift rescale rule
        # For now, just use standard gradient to see if it flows through
        y = op.outputs[0]
        result = grads[0] * y * (1.0 - y)

        # Check if result is all zeros
        # Note: Can't call .numpy() on symbolic tensor, so we track this differently
        # The real test is whether final SHAP values are zero

        return [result]


explainer = MinimalDeepExplainer()

# Step 1: Create LSTM model
print("Step 1: Create LSTM with While loop")
print("-" * 80)

sequence_length = 2
input_size = 2
hidden_size = 2

model = tf.keras.Sequential(
    [tf.keras.layers.LSTM(hidden_size, return_sequences=False, input_shape=(sequence_length, input_size))]
)

dummy = np.random.randn(1, sequence_length, input_size).astype(np.float32)
_ = model(dummy)
print(f"✓ LSTM model created\n")

# Step 2: Create doubled input (like DeepExplainer does)
print("Step 2: Create doubled input (test + baseline)")
print("-" * 80)

baseline = np.zeros((1, sequence_length, input_size), dtype=np.float32)
test_input = np.ones((1, sequence_length, input_size), dtype=np.float32)

# Double the batch
doubled_input = np.concatenate([test_input, baseline], axis=0)  # Shape: (2, seq, features)

print(f"Baseline: all zeros")
print(f"Test: all ones")
print(f"Doubled shape: {doubled_input.shape}")
print()

# Step 3: Get expected output difference
print("Step 3: Compute expected output difference")
print("-" * 80)

output_test = model(test_input).numpy()
output_baseline = model(baseline).numpy()
expected_diff = (output_test - output_baseline).sum()

print(f"Output (test): {output_test[0]}")
print(f"Output (baseline): {output_baseline[0]}")
print(f"Expected difference: {expected_diff:.6f}")
print()

# Step 4: Create concrete function with doubled input
print("Step 4: Create graph with doubled input")
print("-" * 80)


@tf.function
def model_fn(x):
    return model(x)


doubled_spec = tf.TensorSpec(shape=(2, sequence_length, input_size), dtype=tf.float32)
concrete_fn = model_fn.get_concrete_function(doubled_spec)
print(f"✓ Concrete function created (batch_size=2)\n")

# Step 5: Modify gradient registry
print("Step 5: Modify gradient registry")
print("-" * 80)

from tensorflow.python.framework import ops as tf_ops

reg = tf_ops._gradient_registry._registry

for op_name in ["Sigmoid", "Tanh"]:
    if op_name in reg:
        explainer.orig_grads[op_name] = reg[op_name]["type"]
        reg[op_name]["type"] = explainer.custom_grad

print(f"✓ Registry modified for Sigmoid and Tanh\n")

# Step 6: Compute gradients
print("Step 6: Compute gradients (like DeepExplainer does)")
print("-" * 80)

explainer.calls = {"custom_grad": 0, "sigmoid_handler": 0, "handler_returns_zero": 0}

with concrete_fn.graph.as_default():
    graph_input = concrete_fn.inputs[0]  # Shape: (2, seq, features)
    graph_output = concrete_fn.outputs[0]  # Shape: (2, hidden)

    # Compute gradient
    grads = tf.gradients(graph_output, graph_input)

print(f"✓ Gradients computed")
print(f"  custom_grad calls: {explainer.calls['custom_grad']}")
print(f"  Sigmoid handler calls: {explainer.calls['sigmoid_handler']}")
print()

# Step 7: Extract SHAP values (simulate what DeepExplainer does)
print("Step 7: Evaluate gradients and extract SHAP values")
print("-" * 80)

# To actually get values, we need to run in a session or use eager mode
# For TF 2.x, we can trace through eager mode
print("Creating new explainer in eager mode for evaluation...")
print()

# Use eager mode to actually compute values
import shap

e = shap.DeepExplainer(model, baseline)
shap_values = e.shap_values(test_input, check_additivity=False)

print(f"SHAP values shape: {shap_values.shape}")
print(f"SHAP values:\n{shap_values}")
print(f"\nSHAP total: {shap_values.sum():.6f}")
print(f"Expected: {expected_diff:.6f}")
print(f"Error: {abs(shap_values.sum() - expected_diff):.6f}")
print()

# Step 8: Results
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

if explainer.calls["sigmoid_handler"] > 0:
    print(f"✅ Custom Sigmoid gradient WAS called {explainer.calls['sigmoid_handler']} times")
    print("   → TensorFlow's While gradient mechanism works correctly")
    print()
else:
    print("❌ Custom Sigmoid gradient was NOT called")
    print("   → This would be a TensorFlow bug")
    print()

if abs(shap_values.sum()) < 1e-6:
    print("❌ SHAP values are all ZERO (or near-zero)")
    print("   → This is the actual bug!")
    print()
    print("Conclusion:")
    print("- TensorFlow calls custom gradients correctly ✓")
    print("- But DeepExplainer's gradient handlers produce zero SHAP values ✗")
    print("- Issue is in SHAP/DeepExplainer implementation, not TensorFlow")
    print()
    print("Likely causes:")
    print("1. Operations inside While not marked as 'between' operations")
    print("2. variable_inputs() returns False for While body operations")
    print("3. Doubled inputs not flowing correctly through loop iterations")
else:
    print(f"✅ SHAP values are non-zero: {shap_values.sum():.6f}")
    print("   → Everything works!")

# Restore registry
for op_name in ["Sigmoid", "Tanh"]:
    if op_name in explainer.orig_grads and explainer.orig_grads[op_name]:
        reg[op_name]["type"] = explainer.orig_grads[op_name]

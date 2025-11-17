"""
Test using the EXACT approach DeepExplainer uses:
- Directly modify registry dict (not using @tf.RegisterGradient)
- Assign a method as the gradient function
- Use doubled inputs (x and baseline concatenated)
"""

import tensorflow as tf
import numpy as np

print("=" * 80)
print("DeepExplainer Approach to Gradient Modification")
print("=" * 80)
print(f"TensorFlow version: {tf.__version__}\n")


class FakeExplainer:
    """Mimics DeepExplainer's gradient modification approach"""

    def __init__(self):
        self.custom_grad_calls = 0
        self.orig_grads = {}

    def custom_grad(self, op, *grads):
        """This is like DeepExplainer.custom_grad"""
        self.custom_grad_calls += 1
        op_type = op.type
        print(f"  ✓ custom_grad called! (#{self.custom_grad_calls}) type={op_type}, op={op.name[:50]}")

        # For Sigmoid, just compute the standard gradient
        # (DeepExplainer would do DeepLift here, but we're just testing if it's called)
        if op_type == "Sigmoid" or op_type == "shap_Sigmoid":
            y = op.outputs[0]
            return [grads[0] * y * (1.0 - y)]
        else:
            # Fallback to original gradient
            orig_type = op_type[5:] if op_type.startswith("shap_") else op_type
            if orig_type in self.orig_grads and self.orig_grads[orig_type]:
                return self.orig_grads[orig_type](op, *grads)
            else:
                return [None for _ in op.inputs]


explainer = FakeExplainer()

# Create LSTM model
print("Step 1: Create LSTM model")
print("-" * 80)

sequence_length = 3
input_size = 2
hidden_size = 2

model = tf.keras.Sequential(
    [tf.keras.layers.LSTM(hidden_size, return_sequences=False, input_shape=(sequence_length, input_size))]
)

dummy = np.random.randn(1, sequence_length, input_size).astype(np.float32)
_ = model(dummy)
print("✓ Model created\n")

# Get concrete function
print("Step 2: Create concrete function")
print("-" * 80)


@tf.function
def model_fn(x):
    return model(x)


x_shape = (1, sequence_length, input_size)
concrete_fn = model_fn.get_concrete_function(tf.TensorSpec(shape=x_shape, dtype=tf.float32))
print("✓ Concrete function created\n")

# Modify registry using DeepExplainer's approach
print("Step 3: Modify registry (DeepExplainer approach)")
print("-" * 80)

from tensorflow.python.framework import ops as tf_ops

reg = tf_ops._gradient_registry._registry

# Save original gradients
for op_name in ["Sigmoid", "Tanh"]:
    if op_name in reg:
        explainer.orig_grads[op_name] = reg[op_name]["type"]
        # Modify registry to use our custom_grad method
        reg[op_name]["type"] = explainer.custom_grad
        print(f"✓ Modified {op_name} -> explainer.custom_grad")

print()

# Compute gradients
print("Step 4: Compute gradients using tf.gradients()")
print("-" * 80)

explainer.custom_grad_calls = 0

with concrete_fn.graph.as_default():
    graph_input = concrete_fn.inputs[0]
    graph_output = concrete_fn.outputs[0]

    print("Calling tf.gradients()...")
    grads = tf.gradients(graph_output, graph_input)
    print(f"✓ Gradients: {grads}")
    print()

print(f"custom_grad calls: {explainer.custom_grad_calls}")
print()

# Results
print("=" * 80)
print("RESULTS")
print("=" * 80)

if explainer.custom_grad_calls > 0:
    print(f"✅ custom_grad was called {explainer.custom_grad_calls} times")
    print("   Registry modification works with While loops!")
else:
    print("❌ custom_grad was NOT called")
    print("   Registry modification doesn't work with While loops")

print()

# Restore original gradients
for op_name in ["Sigmoid", "Tanh"]:
    if op_name in explainer.orig_grads and explainer.orig_grads[op_name]:
        reg[op_name]["type"] = explainer.orig_grads[op_name]

print("✓ Registry restored")

"""
Minimal example demonstrating that TensorFlow's While loop gradient does NOT use
the gradient registry for operations inside the loop body.

Expected behavior:
- When gradient registry is modified for an operation (e.g., Sigmoid)
- All gradient computations for that operation should use the modified gradient
- This includes operations inside While loop bodies

Actual behavior:
- Modified gradient IS used for Sigmoid operations outside While loops
- Modified gradient IS NOT used for Sigmoid operations inside While loop bodies
- While loop gradient computation bypasses the gradient registry for body operations

Impact:
- Makes it impossible to override gradients for operations inside While loops
- Breaks libraries like SHAP/DeepExplainer that rely on gradient customization
- Inconsistent behavior compared to other operations

Tested with TensorFlow 2.20.0
"""

import tensorflow as tf
import numpy as np

print("=" * 80)
print("TensorFlow While Loop Gradient Registry Bug Demonstration")
print("=" * 80)
print(f"TensorFlow version: {tf.__version__}\n")

# Global counter to track gradient calls
gradient_calls = {"original": 0, "custom": 0}


# Register a custom Sigmoid gradient
@tf.RegisterGradient("CustomSigmoid")
def custom_sigmoid_gradient(op, grad):
    """Custom Sigmoid gradient that tracks when it's called"""
    gradient_calls["custom"] += 1
    print(f"  ✓ CustomSigmoid gradient called! (call #{gradient_calls['custom']})")

    # Compute the same gradient as original, but we've tracked the call
    y = op.outputs[0]
    return grad * y * (1.0 - y)


# Save the original Sigmoid gradient
from tensorflow.python.framework import ops as tf_ops

original_sigmoid_grad = tf_ops._gradient_registry._registry.get("Sigmoid", {}).get("type")


def count_original_gradient(op, grad):
    """Wrapper to count original gradient calls"""
    gradient_calls["original"] += 1
    if original_sigmoid_grad:
        return original_sigmoid_grad(op, grad)
    else:
        # Fallback: standard sigmoid gradient
        y = op.outputs[0]
        return grad * y * (1.0 - y)


print("Step 1: Create a simple model with Sigmoid OUTSIDE a While loop")
print("-" * 80)


@tf.function
def model_without_while(x):
    """Simple model: x -> Sigmoid"""
    return tf.sigmoid(x)


x_test = tf.constant([[1.0, 2.0]], dtype=tf.float32)

# Compute gradient WITHOUT custom gradient
gradient_calls = {"original": 0, "custom": 0}
with tf.GradientTape() as tape:
    tape.watch(x_test)
    y = model_without_while(x_test)
    loss = tf.reduce_sum(y)

grad_normal = tape.gradient(loss, x_test)
print(f"Normal gradient computed: {grad_normal.numpy()}")
print(f"Custom gradient calls: {gradient_calls['custom']}")
print()

# Modify the gradient registry to use our custom Sigmoid gradient
print("Step 2: Modify gradient registry to use CustomSigmoid")
print("-" * 80)
reg = tf_ops._gradient_registry._registry
reg["Sigmoid"]["type"] = custom_sigmoid_gradient
print("✓ Registry modified: Sigmoid -> CustomSigmoid\n")

# Compute gradient WITH custom gradient (outside While loop)
print("Step 3: Compute gradient for Sigmoid OUTSIDE While loop")
print("-" * 80)
gradient_calls = {"original": 0, "custom": 0}

with tf.GradientTape() as tape:
    tape.watch(x_test)
    y = model_without_while(x_test)
    loss = tf.reduce_sum(y)

grad_custom = tape.gradient(loss, x_test)
print(f"Custom gradient computed: {grad_custom.numpy()}")
print(f"Custom gradient calls: {gradient_calls['custom']}")

if gradient_calls["custom"] > 0:
    print("✅ SUCCESS: Custom gradient WAS used for Sigmoid outside While loop\n")
else:
    print("❌ FAIL: Custom gradient NOT used (unexpected)\n")

# Now test with a While loop containing Sigmoid
print("Step 4: Create model with Sigmoid INSIDE a While loop")
print("-" * 80)


def create_while_loop_model():
    """Create an LSTM-like model that uses a While loop with Sigmoid inside"""

    # Simple LSTM layer - uses While loop internally
    sequence_length = 3
    input_size = 2
    hidden_size = 2

    model = tf.keras.Sequential(
        [tf.keras.layers.LSTM(hidden_size, return_sequences=False, input_shape=(sequence_length, input_size))]
    )

    # Build the model
    dummy = np.random.randn(1, sequence_length, input_size).astype(np.float32)
    _ = model(dummy)

    return model


lstm_model = create_while_loop_model()
print("✓ LSTM model created (uses While loop with Sigmoid operations inside)\n")

# Verify the model uses a While loop
print("Step 5: Verify model graph contains While loop with Sigmoid")
print("-" * 80)


@tf.function
def traced_lstm(x):
    return lstm_model(x)


x_seq = tf.constant(np.random.randn(1, 3, 2).astype(np.float32))
concrete_func = traced_lstm.get_concrete_function(x_seq)

# Check for While operation
while_ops = [op for op in concrete_func.graph.get_operations() if op.type == "While"]
print(f"While operations found: {len(while_ops)}")

if while_ops:
    # Check the body for Sigmoid operations
    while_op = while_ops[0]
    body_func = while_op.get_attr("body")
    graph_def = concrete_func.graph.as_graph_def()

    sigmoid_count = 0
    for func_def in graph_def.library.function:
        if func_def.signature.name == body_func.name:
            for node in func_def.node_def:
                if node.op == "Sigmoid":
                    sigmoid_count += 1

    print(f"Sigmoid operations inside While loop body: {sigmoid_count}")
    print()

# Compute gradient through the While loop
print("Step 6: Compute gradient through While loop (with Sigmoid inside)")
print("-" * 80)
print("Registry is STILL modified: Sigmoid -> CustomSigmoid")
print()

gradient_calls = {"original": 0, "custom": 0}

x_lstm = tf.constant(np.random.randn(1, 3, 2).astype(np.float32))

with tf.GradientTape() as tape:
    tape.watch(x_lstm)
    y_lstm = lstm_model(x_lstm)
    loss_lstm = tf.reduce_sum(y_lstm)

grad_lstm = tape.gradient(loss_lstm, x_lstm)
print(f"Gradient through While loop computed")
print(f"Custom gradient calls: {gradient_calls['custom']}")
print()

# Results
print("=" * 80)
print("RESULTS")
print("=" * 80)

if gradient_calls["custom"] == 0:
    print("❌ BUG CONFIRMED:")
    print("   Custom Sigmoid gradient was NOT used inside While loop body")
    print("   Even though gradient registry was modified globally")
    print()
    print("Expected: Custom gradient should be used (consistent with behavior outside While)")
    print("Actual: Custom gradient was bypassed for operations inside While loop body")
    print()
    print("Impact:")
    print("- Makes it impossible to customize gradients for ops inside While loops")
    print("- Breaks SHAP/DeepExplainer and similar tools that rely on gradient overriding")
    print("- Inconsistent behavior: registry works outside While but not inside")
else:
    print(f"✅ Custom gradient WAS used {gradient_calls['custom']} times")
    print("   While loop correctly uses gradient registry")

# Restore original gradient
print()
print("Restoring original gradient registry...")
if original_sigmoid_grad:
    reg["Sigmoid"]["type"] = original_sigmoid_grad
print("✓ Done")

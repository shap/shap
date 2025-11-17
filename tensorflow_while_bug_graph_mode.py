"""
Test gradient registry with While loops in GRAPH MODE (not eager)

This is closer to what DeepExplainer does in non-eager TensorFlow.
Key: Use tf.gradients() in graph mode, not GradientTape in eager mode.
"""

import tensorflow as tf
import numpy as np

print("=" * 80)
print("While Loop Gradient Registry - GRAPH MODE (tf.gradients)")
print("=" * 80)
print(f"TensorFlow version: {tf.__version__}\n")

gradient_calls = {"custom": 0}


@tf.RegisterGradient("CustomSigmoid")
def custom_sigmoid_gradient(op, grad):
    """Custom Sigmoid gradient"""
    gradient_calls["custom"] += 1
    print(f"  ✓ CustomSigmoid called! (#{gradient_calls['custom']}) op={op.name[:50]}")
    y = op.outputs[0]
    return grad * y * (1.0 - y)


# Create model
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

# Get concrete function in graph mode
print("Step 2: Create graph (tf.function + get_concrete_function)")
print("-" * 80)


@tf.function
def model_fn(x):
    return model(x)


x_shape = (1, sequence_length, input_size)
concrete_fn = model_fn.get_concrete_function(tf.TensorSpec(shape=x_shape, dtype=tf.float32))
print(f"✓ Concrete function created")
print(f"  Graph: {concrete_fn.graph}")
print()

# Modify registry AFTER graph creation
print("Step 3: Modify gradient registry")
print("-" * 80)

from tensorflow.python.framework import ops as tf_ops

reg = tf_ops._gradient_registry._registry
original_grad = reg.get("Sigmoid", {}).get("type")
reg["Sigmoid"]["type"] = custom_sigmoid_gradient
print("✓ Registry modified\n")

# Compute gradients using tf.gradients (graph mode)
print("Step 4: Compute gradients using tf.gradients()")
print("-" * 80)
print("Using the EXISTING graph (created before registry modification)")
print()

gradient_calls = {"custom": 0}

# Get input and output from the concrete function's graph
with concrete_fn.graph.as_default():
    # The function takes one input
    graph_input = concrete_fn.inputs[0]
    graph_output = concrete_fn.outputs[0]

    print(f"Graph input: {graph_input}")
    print(f"Graph output: {graph_output}")
    print()

    # Compute gradient using tf.gradients
    print("Calling tf.gradients()...")
    grads = tf.gradients(graph_output, graph_input)
    print(f"✓ Gradients computed: {grads}")
    print()

print(f"Custom gradient calls during tf.gradients(): {gradient_calls['custom']}")
print()

# Compare with eager mode
print("Step 5: Compare with EAGER mode (GradientTape)")
print("-" * 80)

gradient_calls = {"custom": 0}

x_eager = tf.constant(np.random.randn(*x_shape).astype(np.float32))

with tf.GradientTape() as tape:
    tape.watch(x_eager)
    y_eager = model(x_eager)

grad_eager = tape.gradient(y_eager, x_eager)
print(f"Gradient computed: {grad_eager.shape}")
print(f"Custom gradient calls: {gradient_calls['custom']}")
print()

# Results
print("=" * 80)
print("RESULTS")
print("=" * 80)
print()
print("GRAPH MODE (tf.gradients on pre-built graph):")
print(f"  Custom gradient calls: UNKNOWN (need to run graph)")
print()
print("EAGER MODE (GradientTape):")
print(f"  Custom gradient calls: {gradient_calls['custom']}")
print()

print("Note: Graph mode gradients are symbolic - need sess.run() to execute")
print("But the gradient OPERATIONS are created during tf.gradients()")
print("If custom gradient was used, we would see print statements above")

# Restore
if original_grad:
    reg["Sigmoid"]["type"] = original_grad

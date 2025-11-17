"""
Test using gradient_override_map to replace gradients inside While loop
"""
import tensorflow as tf
import numpy as np

print("="*80)
print("Testing gradient_override_map for While loop operations")
print("="*80)

# Create a simple model with While loop
sequence_length = 3
input_size = 2
hidden_size = 2

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(hidden_size, return_sequences=False, input_shape=(sequence_length, input_size))
])

dummy_input = np.random.randn(1, sequence_length, input_size).astype(np.float32)
_ = model(dummy_input)

# Create test inputs
x_test = tf.constant(np.random.randn(1, sequence_length, input_size).astype(np.float32))
x_base = tf.constant(np.random.randn(1, sequence_length, input_size).astype(np.float32) * 0.1)

print(f"\nOriginal model output shapes:")
print(f"  Test output: {model(x_test).shape}")
print(f"  Base output: {model(x_base).shape}")

# Test 1: Normal gradient
print(f"\n{'='*80}")
print("Test 1: Normal gradient computation")
print(f"{'='*80}")

with tf.GradientTape() as tape:
    tape.watch(x_test)
    output = model(x_test)
    loss = tf.reduce_sum(output)

normal_grad = tape.gradient(loss, x_test)
print(f"Normal gradient shape: {normal_grad.shape}")
print(f"Normal gradient sum: {tf.reduce_sum(normal_grad).numpy():.6f}")

# Test 2: Try using gradient_override_map
print(f"\n{'='*80}")
print("Test 2: Using gradient_override_map")
print(f"{'='*80}")

# Define custom Sigmoid gradient
@tf.RegisterGradient("CustomSigmoid")
def custom_sigmoid_grad(op, grad):
    """Custom gradient for Sigmoid - just pass through for testing"""
    print("  ✓ CustomSigmoid gradient called!")
    # Original Sigmoid gradient: y * (1 - y) * grad
    y = op.outputs[0]
    original_grad = grad * y * (1.0 - y)
    print(f"    Original grad sum: {tf.reduce_sum(original_grad).numpy():.6f}")
    # For now, return original gradient to verify it's being called
    return original_grad

# Try to override Sigmoid gradient
try:
    with tf.GradientTape() as tape:
        tape.watch(x_test)
        # Use gradient_override_map context
        with tape.gradient_override_map({"Sigmoid": "CustomSigmoid"}):
            output = model(x_test)
            loss = tf.reduce_sum(output)

    override_grad = tape.gradient(loss, x_test)
    print(f"Override gradient shape: {override_grad.shape}")
    print(f"Override gradient sum: {tf.reduce_sum(override_grad).numpy():.6f}")

    # Check if gradients are different (they shouldn't be since we return original)
    diff = tf.reduce_sum(tf.abs(override_grad - normal_grad)).numpy()
    print(f"\nDifference from normal gradient: {diff:.6f}")

except Exception as e:
    print(f"✗ Error with gradient_override_map: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'='*80}")
print("Test 3: Check if override works at graph construction time")
print(f"{'='*80}")

# Try creating the model inside the override context
try:
    with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess:
            # Create placeholder
            x_placeholder = tf.compat.v1.placeholder(tf.float32, [1, sequence_length, input_size])

            # Create model inside override context
            with tf.compat.v1.get_default_graph().gradient_override_map({"Sigmoid": "CustomSigmoid"}):
                model2 = tf.keras.Sequential([
                    tf.keras.layers.LSTM(hidden_size, return_sequences=False, input_shape=(sequence_length, input_size))
                ])
                output2 = model2(x_placeholder)

            print("✓ Model created with gradient override")

            # Now compute gradient
            grad_op = tf.gradients(output2, x_placeholder)[0]
            print("✓ Gradient operation created")

            # Initialize and run
            sess.run(tf.compat.v1.global_variables_initializer())
            grad_val = sess.run(grad_op, feed_dict={x_placeholder: x_test.numpy()})
            print(f"Gradient shape: {grad_val.shape}")
            print(f"Gradient sum: {np.sum(grad_val):.6f}")

except Exception as e:
    print(f"✗ Error with graph-level override: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'='*80}")
print("KEY FINDINGS")
print(f"{'='*80}")
print("We need to understand:")
print("1. Does gradient_override_map work with While loops?")
print("2. If yes, when should we apply it?")
print("3. If no, we need a different approach")

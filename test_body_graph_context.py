"""
Test to understand why gradient registry isn't used for While loop body operations
"""
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops as tf_ops

print("="*80)
print("Understanding While Loop Body Graph Context")
print("="*80)

# Create a simple LSTM
sequence_length = 3
input_size = 2
hidden_size = 2

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(hidden_size, return_sequences=False, input_shape=(sequence_length, input_size))
])

dummy_input = np.random.randn(1, sequence_length, input_size).astype(np.float32)
_ = model(dummy_input)

# Get the concrete function
@tf.function
def traced_model(x):
    return model(x)

concrete_func = traced_model.get_concrete_function(
    tf.TensorSpec(shape=(1, sequence_length, input_size), dtype=tf.float32)
)

# Find the While operation
while_ops = [op for op in concrete_func.graph.get_operations() if op.type == "While"]
print(f"\nFound {len(while_ops)} While operation(s)")

if while_ops:
    while_op = while_ops[0]

    # Get the body function
    body_func = while_op.get_attr("body")
    print(f"\nBody function: {body_func.name}")

    # Convert to FuncGraph
    from tensorflow.python.framework import function_def_to_graph

    graph_def = concrete_func.graph.as_graph_def()
    body_func_def = None
    for func_def in graph_def.library.function:
        if func_def.signature.name == body_func.name:
            body_func_def = func_def
            break

    if body_func_def:
        body_graph = function_def_to_graph.function_def_to_graph(body_func_def)
        print(f"Body graph type: {type(body_graph)}")

        # Find Sigmoid operations in the body
        sigmoid_ops = [op for op in body_graph.get_operations() if op.type == "Sigmoid"]
        print(f"\nSigmoid operations in body: {len(sigmoid_ops)}")

        # Now test: what happens when we compute gradients for the body?
        print(f"\n{'='*80}")
        print("TEST: Compute gradient for a Sigmoid op in the body")
        print(f"{'='*80}")

        if sigmoid_ops:
            sig_op = sigmoid_ops[0]
            print(f"\nSigmoid op: {sig_op.name}")
            print(f"Input: {sig_op.inputs[0]}")
            print(f"Output: {sig_op.outputs[0]}")

            # Check the gradient registry
            reg = tf_ops._gradient_registry._registry
            print(f"\nCurrent Sigmoid gradient in registry: {reg.get('Sigmoid', {}).get('type')}")

            # Modify the registry
            original_sigmoid_grad = reg.get('Sigmoid', {}).get('type')

            @tf.RegisterGradient("TestSigmoid")
            def test_sigmoid_grad(op, grad):
                print("  ✓✓✓ TEST SIGMOID GRADIENT CALLED! ✓✓✓")
                y = op.outputs[0]
                return grad * y * (1.0 - y)

            print("\nModifying Sigmoid gradient registry...")
            reg['Sigmoid']['type'] = test_sigmoid_grad

            # Now try to compute a gradient that involves this Sigmoid
            print("\nComputing gradient using tf.gradients...")

            # We need to create a simple computation that uses the Sigmoid
            # and then compute its gradient
            try:
                # Get the body's output and input
                body_outputs = list(body_graph.outputs)
                body_inputs = list(body_graph.inputs)

                print(f"Body has {len(body_inputs)} inputs and {len(body_outputs)} outputs")

                # Try to compute gradient of first output w.r.t. first input
                with body_graph.as_default():
                    if body_outputs and body_inputs:
                        grad = tf.gradients(body_outputs[0], body_inputs[0])
                        print(f"Gradient computed: {grad}")

                print("\nDid 'TEST SIGMOID GRADIENT CALLED!' appear above?")
                print("If NO: TensorFlow doesn't use the registry for body function gradients")
                print("If YES: The registry IS used, but something else is wrong")

            except Exception as e:
                print(f"Error computing gradient: {e}")

            finally:
                # Restore original gradient
                if original_sigmoid_grad:
                    reg['Sigmoid']['type'] = original_sigmoid_grad

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("If the test gradient was called: Registry IS used, need different fix")
print("If NOT called: Need to modify body function before While gradient is computed")

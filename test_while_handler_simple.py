"""
Simple test to understand While loop structure and try basic gradient override
"""
import tensorflow as tf
import numpy as np

# Create a simple LSTM to get a While loop
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
print(f"Found {len(while_ops)} While operation(s)")

if while_ops:
    while_op = while_ops[0]
    print(f"\nWhile operation: {while_op.name}")
    print(f"Type: {while_op.type}")

    # Get attributes
    print(f"\nAttributes:")
    for attr_name in while_op.node_def.attr:
        print(f"  {attr_name}")

    # Access the body and cond functions
    body_func = while_op.get_attr("body")
    cond_func = while_op.get_attr("cond")

    print(f"\nBody function: {body_func.name}")
    print(f"Condition function: {cond_func.name}")

    # Get the body graph
    from tensorflow.python.framework import func_graph as func_graph_module

    # The body function is a FuncGraph
    print(f"\n{'='*80}")
    print("OPERATIONS INSIDE WHILE LOOP BODY:")
    print(f"{'='*80}")

    # Access the body's operations
    # Note: This is tricky because the body is a separate function
    print(f"\nBody function type: {type(body_func)}")
    print(f"Body function signature: {body_func.signature if hasattr(body_func, 'signature') else 'N/A'}")

    # Try to get the function definition
    if hasattr(body_func, 'definition'):
        print("\nBody has definition attribute")
        func_def = body_func.definition
        print(f"Function definition: {func_def}")

    # Get the graph def to inspect operations
    graph_def = concrete_func.graph.as_graph_def()

    # Find function definitions
    print(f"\n{'='*80}")
    print("FUNCTION LIBRARY:")
    print(f"{'='*80}")

    if graph_def.library and graph_def.library.function:
        print(f"\nFound {len(graph_def.library.function)} function(s) in library:")
        for func in graph_def.library.function:
            print(f"\n  Function: {func.signature.name}")
            print(f"  Operations in function:")

            # Count operation types
            op_counts = {}
            for node in func.node_def:
                op_type = node.op
                op_counts[op_type] = op_counts.get(op_type, 0) + 1

            for op_type, count in sorted(op_counts.items()):
                print(f"    - {op_type}: {count}")

            # Highlight the ones we care about
            important_ops = ["Sigmoid", "Tanh", "Mul", "MatMul"]
            print(f"\n  Operations needing gradient replacement:")
            for op_type in important_ops:
                if op_type in op_counts:
                    print(f"    ✓ {op_type}: {op_counts[op_type]} operation(s)")

print("\n" + "="*80)
print("KEY INSIGHT:")
print("="*80)
print("The While loop's body is stored as a separate function in the graph library.")
print("To override gradients inside it, we need to:")
print("1. Access the function definition from the library")
print("2. Identify operations inside (Sigmoid, Tanh, Mul, etc.)")
print("3. Apply gradient replacement to those operations")
print("4. This requires modifying the function or intercepting its gradient computation")

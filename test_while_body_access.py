"""
Test accessing and modifying operations inside While loop body function
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
print(f"Found {len(while_ops)} While operation(s)\n")

if while_ops:
    while_op = while_ops[0]
    print(f"While operation: {while_op.name}")

    # Get the body function
    body_func = while_op.get_attr("body")
    print(f"Body function: {body_func}")
    print(f"Body function type: {type(body_func)}")

    # Try to access the graph from the function library
    graph_def = concrete_func.graph.as_graph_def()

    print(f"\n{'='*80}")
    print("ATTEMPTING TO ACCESS BODY FUNCTION OPERATIONS")
    print(f"{'='*80}\n")

    # Find the body function in the library
    body_func_name = body_func.name
    body_func_def = None

    for func_def in graph_def.library.function:
        if func_def.signature.name == body_func_name:
            body_func_def = func_def
            break

    if body_func_def:
        print(f"✓ Found body function definition: {body_func_def.signature.name}")
        print(f"\nOperations in body function:")

        # List all Sigmoid operations
        sigmoid_ops = [node for node in body_func_def.node_def if node.op == "Sigmoid"]
        print(f"\nSigmoid operations: {len(sigmoid_ops)}")
        for i, node in enumerate(sigmoid_ops):
            print(f"  {i+1}. {node.name}")
            print(f"     Input: {node.input}")

        # List all Tanh operations
        tanh_ops = [node for node in body_func_def.node_def if node.op == "Tanh"]
        print(f"\nTanh operations: {len(tanh_ops)}")
        for i, node in enumerate(tanh_ops):
            print(f"  {i+1}. {node.name}")
            print(f"     Input: {node.input}")

        # List all Mul operations
        mul_ops = [node for node in body_func_def.node_def if node.op == "Mul"]
        print(f"\nMul operations: {len(mul_ops)}")
        for i, node in enumerate(mul_ops):
            print(f"  {i+1}. {node.name}")
            print(f"     Inputs: {node.input}")

        print(f"\n{'='*80}")
        print("NEXT STEP: Access body function as FuncGraph")
        print(f"{'='*80}\n")

        # Try to get the actual FuncGraph from the concrete function's graph
        # The body function should be accessible through the graph's functions
        if hasattr(concrete_func.graph, '_functions'):
            print(f"Graph has _functions attribute")
            print(f"Functions: {concrete_func.graph._functions.keys() if hasattr(concrete_func.graph._functions, 'keys') else 'N/A'}")

        # Try another approach: access via library_function
        try:
            from tensorflow.python.framework import function_def_to_graph
            body_graph = function_def_to_graph.function_def_to_graph(body_func_def)
            print(f"\n✓ Successfully created graph from function definition!")
            print(f"Body graph type: {type(body_graph)}")
            print(f"Number of operations: {len(list(body_graph.get_operations()))}")

            # List operations in the graph
            print(f"\nOperations in body graph:")
            sigmoid_count = 0
            tanh_count = 0
            mul_count = 0

            for op in body_graph.get_operations():
                if op.type == "Sigmoid":
                    sigmoid_count += 1
                    print(f"  Sigmoid: {op.name}")
                elif op.type == "Tanh":
                    tanh_count += 1
                    print(f"  Tanh: {op.name}")
                elif op.type == "Mul":
                    mul_count += 1
                    print(f"  Mul: {op.name}")

            print(f"\nCounts: Sigmoid={sigmoid_count}, Tanh={tanh_count}, Mul={mul_count}")

        except Exception as e:
            print(f"\n✗ Error creating graph from function def: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print("KEY INSIGHT")
    print(f"{'='*80}")
    print("We can access the body function's operations through:")
    print("1. body_func_def.node_def - List of NodeDef objects")
    print("2. function_def_to_graph(body_func_def) - Convert to FuncGraph")
    print("\nNow we need to figure out how to replace gradients for these operations!")

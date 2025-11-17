"""
Trace the sequence input from model input to While loop
"""
import tensorflow as tf
import numpy as np

print("="*80)
print("Tracing Sequence Input Path")
print("="*80)

# Create LSTM model
sequence_length = 3
input_size = 2
hidden_size = 2

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(hidden_size, return_sequences=False, input_shape=(sequence_length, input_size))
])

dummy_input = np.random.randn(1, sequence_length, input_size).astype(np.float32)
_ = model(dummy_input)

# Get the concrete function with doubled batch (like DeepExplainer)
@tf.function
def traced_model(x):
    return model(x)

doubled_input = tf.constant(np.random.randn(2, sequence_length, input_size).astype(np.float32))
concrete_func = traced_model.get_concrete_function(doubled_input)

# Find the While loop
while_ops = [op for op in concrete_func.graph.get_operations() if op.type == "While"]
if while_ops:
    while_op = while_ops[0]
    print(f"\nWhile operation: {while_op.name}")

    # Find input 6 (the TensorList)
    tensor_list_input = while_op.inputs[6]
    print(f"\nInput 6 (TensorList): {tensor_list_input.name}")
    print(f"  Shape: {tensor_list_input.shape}")
    print(f"  Dtype: {tensor_list_input.dtype}")
    print(f"  Produced by: {tensor_list_input.op.name}, type={tensor_list_input.op.type}")

    # Trace backwards from TensorList to find the actual sequence input
    current_op = tensor_list_input.op
    print(f"\n{'='*80}")
    print("BACKWARDS TRACE FROM WHILE INPUT 6:")
    print(f"{'='*80}\n")

    visited = set()
    def trace_back(op, depth=0):
        if op.name in visited or depth > 10:
            return
        visited.add(op.name)

        indent = "  " * depth
        print(f"{indent}{op.name} [{op.type}]")

        # Show inputs
        for i, inp in enumerate(op.inputs):
            print(f"{indent}  Input {i}: {inp.name}, shape={inp.shape}, dtype={inp.dtype}")

        # Recursively trace back inputs
        for inp in op.inputs:
            if inp.op.type not in ["Const", "Placeholder", "ReadVariableOp"]:
                trace_back(inp.op, depth + 1)

    trace_back(current_op)

    print(f"\n{'='*80}")
    print("KEY QUESTION:")
    print(f"{'='*80}")
    print("To get gradients w.r.t. the actual sequence input:")
    print("1. We need to find the TensorListFromTensor operation")
    print("2. Get its gradient w.r.t. its input tensor (the actual sequence)")
    print("3. This should give us the SHAP values for the sequence")

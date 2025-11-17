"""
Investigate what operations are inside TensorFlow's While loop for LSTM
"""
import tensorflow as tf
import numpy as np

# Create a simple LSTM model
sequence_length = 3
input_size = 2
hidden_size = 2

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(hidden_size, return_sequences=False, input_shape=(sequence_length, input_size))
])

# Build the model
dummy_input = np.random.randn(1, sequence_length, input_size).astype(np.float32)
_ = model(dummy_input)

# Get the concrete function to see the graph
@tf.function
def traced_model(x):
    return model(x)

concrete_func = traced_model.get_concrete_function(
    tf.TensorSpec(shape=(1, sequence_length, input_size), dtype=tf.float32)
)

# Print the graph operations
print("="*80)
print("Operations in LSTM graph:")
print("="*80)

ops_by_type = {}
for op in concrete_func.graph.get_operations():
    op_type = op.type
    if op_type not in ops_by_type:
        ops_by_type[op_type] = []
    ops_by_type[op_type].append(op.name)

# Print grouped by operation type
for op_type in sorted(ops_by_type.keys()):
    print(f"\n{op_type} ({len(ops_by_type[op_type])} ops):")
    for name in ops_by_type[op_type][:3]:  # Show first 3 examples
        print(f"  - {name}")
    if len(ops_by_type[op_type]) > 3:
        print(f"  ... and {len(ops_by_type[op_type]) - 3} more")

# Highlight the While loop
print("\n" + "="*80)
print("THE KEY OPERATION:")
print("="*80)
if "While" in ops_by_type:
    print(f"\n⚠️  Found While loop(s): {len(ops_by_type['While'])} operation(s)")
    for name in ops_by_type["While"]:
        print(f"  - {name}")

    print("\nThe While loop contains:")
    print("  - Condition function (when to stop)")
    print("  - Body function (LSTM cell computation for each timestep)")
    print("  - Inside the body: MatMul, Sigmoid, Tanh, Mul, Add, etc.")
    print("\nTo support LSTM sequences, we'd need to:")
    print("  1. Intercept the While operation")
    print("  2. Unroll or recursively apply gradient replacement to the body")
    print("  3. Handle state carried between iterations (h, c)")
    print("  4. Properly accumulate SHAP values across timesteps")
else:
    print("No While loop found (unexpected!)")

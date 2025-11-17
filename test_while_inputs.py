"""
Test to understand what inputs the While loop receives
"""
import tensorflow as tf
import numpy as np

print("="*80)
print("Inspecting While Loop Inputs")
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

# Get the concrete function
@tf.function
def traced_model(x):
    return model(x)

# Test with normal input
normal_input = tf.constant(np.random.randn(1, sequence_length, input_size).astype(np.float32))
concrete_func_normal = traced_model.get_concrete_function(normal_input)

# Test with doubled input (like DeepExplainer does)
doubled_input = tf.constant(np.random.randn(2, sequence_length, input_size).astype(np.float32))
concrete_func_doubled = traced_model.get_concrete_function(doubled_input)

print("\n" + "="*80)
print("NORMAL INPUT (batch_size=1)")
print("="*80)

while_ops_normal = [op for op in concrete_func_normal.graph.get_operations() if op.type == "While"]
if while_ops_normal:
    while_op = while_ops_normal[0]
    print(f"\nWhile operation: {while_op.name}")
    print(f"Number of inputs: {len(while_op.inputs)}")
    print("\nInput shapes:")
    for i, inp in enumerate(while_op.inputs):
        print(f"  {i}: {inp.shape} - {inp.name}")

print("\n" + "="*80)
print("DOUBLED INPUT (batch_size=2, like DeepExplainer)")
print("="*80)

while_ops_doubled = [op for op in concrete_func_doubled.graph.get_operations() if op.type == "While"]
if while_ops_doubled:
    while_op = while_ops_doubled[0]
    print(f"\nWhile operation: {while_op.name}")
    print(f"Number of inputs: {len(while_op.inputs)}")
    print("\nInput shapes:")
    for i, inp in enumerate(while_op.inputs):
        print(f"  {i}: {inp.shape} - {inp.name}")

print("\n" + "="*80)
print("KEY QUESTION")
print("="*80)
print("When batch_size goes from 1 to 2:")
print("- Does the sequence input (TensorListFromTensor) double in batch size?")
print("- Do the initial hidden states (h, c) double in batch size?")
print("\nIf YES to both: The hidden states ARE being doubled properly")
print("If NO to hidden states: That's the problem - hidden states need doubling!")

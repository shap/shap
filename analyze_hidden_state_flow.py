"""
Analyze how hidden states flow through the While loop and where errors arise
"""
import tensorflow as tf
import numpy as np

print("="*80)
print("Analyzing Hidden State Flow in While Loop")
print("="*80)

# Create 2-timestep LSTM to keep it simple
seq_len = 2
input_size = 3
hidden_size = 4

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(hidden_size, return_sequences=False,
                         input_shape=(seq_len, input_size))
])

dummy = np.random.randn(1, seq_len, input_size).astype(np.float32)
_ = model(dummy)

print("\nCreated 2-timestep LSTM")
print(f"Input: ({seq_len}, {input_size}) -> Hidden: {hidden_size}")
print()

# Get the concrete function with DOUBLED batch
@tf.function
def model_fn(x):
    return model(x)

x_spec = tf.TensorSpec(shape=(2, seq_len, input_size), dtype=tf.float32)  # Doubled!
concrete_fn = model_fn.get_concrete_function(x_spec)

# Find While operation
while_ops = [op for op in concrete_fn.graph.get_operations() if op.type == "While"]
if not while_ops:
    print("No While operations found!")
    exit(1)

while_op = while_ops[0]
print(f"Found While operation: {while_op.name}")
print(f"Inputs: {len(while_op.inputs)}")
print(f"Outputs: {len(while_op.outputs)}")
print()

# Analyze inputs
print("While Loop Inputs:")
print("-"*80)
for i, inp in enumerate(while_op.inputs):
    print(f"  {i}: {inp.name[:60]}")
    print(f"      Shape: {inp.shape}, Dtype: {inp.dtype}")

print()
print("Key observations:")
print("  Input 4 & 5: Initial h and c states")
print("    - These start as zeros(batch_size, hidden_size)")
print("    - With doubled batch: zeros(2, hidden_size)")
print("    - This is CORRECT - both test and baseline start with zero states")
print()

# Analyze outputs
print("While Loop Outputs:")
print("-"*80)
for i, out in enumerate(while_op.outputs):
    print(f"  {i}: {out.name[:60]}")
    print(f"      Shape: {out.shape}")

print()

# Get body function
body_func = while_op.get_attr("body")
graph_def = concrete_fn.graph.as_graph_def()

body_func_def = None
for func_def in graph_def.library.function:
    if func_def.signature.name == body_func.name:
        body_func_def = func_def
        break

if body_func_def:
    print("While Body Function Structure:")
    print("-"*80)

    # Find operations that involve hidden states
    print("\nOperations involving hidden states (h, c):")

    # Look for placeholders (inputs to body)
    placeholders = [node for node in body_func_def.node_def if 'placeholder' in node.name.lower()]
    print(f"\nBody inputs (placeholders): {len(placeholders)}")
    for i, node in enumerate(placeholders[:8]):  # Show first 8
        print(f"  {i}: {node.name}")

    # Find LSTMCell operations
    print("\nLSTM Cell operations:")
    lstm_ops = [node for node in body_func_def.node_def if 'lstm_cell' in node.name.lower()]

    # Focus on operations that combine x_t with h,c
    matmul_ops = [node for node in body_func_def.node_def if node.op == 'MatMul']
    print(f"\nMatMul operations (x*W and h*U): {len(matmul_ops)}")
    for node in matmul_ops:
        print(f"  {node.name}")
        print(f"    Inputs: {list(node.input)}")

    # Find where h,c are updated
    print(f"\nTotal operations in body: {len(body_func_def.node_def)}")

    # Key question: At what point do h,c get DOUBLED?
    print()
    print("="*80)
    print("KEY QUESTION")
    print("="*80)
    print()
    print("Hidden state flow through iterations:")
    print()
    print("Iteration 0:")
    print("  h0 = zeros(2, hidden_size)  <- doubled batch, both zero")
    print("  c0 = zeros(2, hidden_size)  <- doubled batch, both zero")
    print("  x0 = input[:, 0, :]         <- doubled: [x_test[0], x_base[0]]")
    print()
    print("Iteration 1:")
    print("  h1, c1 = LSTMCell(x1, h0, c0)")
    print("  ├─ h0 and c0 are ZEROS for both test and baseline ✓")
    print("  ├─ x1 is DIFFERENT for test vs baseline ✓")
    print("  └─ Result: h1, c1 are DIFFERENT for test vs baseline ✓")
    print()
    print("Iteration 2:")
    print("  h2, c2 = LSTMCell(x2, h1, c1)")
    print("  ├─ h1, c1 are outputs from iteration 1")
    print("  ├─ Are h1, c1 properly split as [h1_test, h1_base] ???")
    print("  └─ If NOT, errors propagate!")
    print()
    print("The question: Does the While backward pass correctly maintain")
    print("separate gradients for the test and baseline hidden states?")
    print()
    print("If the backward pass mixes test/baseline hidden states,")
    print("that would cause error compounding!")

print()
print("="*80)
print("HYPOTHESIS")
print("="*80)
print()
print("TensorFlow's While gradient computes:")
print("  1. Forward pass: Correctly maintains doubled batch throughout")
print("  2. Backward pass: Creates a reverse While loop")
print("  3. Body gradients: Uses standard TensorFlow ops")
print()
print("Our gradient handlers (Sigmoid, Tanh, etc.) expect:")
print("  - Input to be [x_test, x_base] concatenated")
print("  - They split, compute DeepLift, return gradient")
print()
print("Potential issue:")
print("  - Inside While backward loop, gradients flow through h,c")
print("  - Gradient handlers may not correctly split h,c gradients")
print("  - This causes errors to compound")
print()
print("To verify: Check if h,c tensors in backward pass have correct shapes")
print("Expected: (2*batch, hidden) at each backward iteration")
print("If actual: (batch, hidden), then doubling is lost!")

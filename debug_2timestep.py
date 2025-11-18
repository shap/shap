"""
Debug 2-timestep LSTM to understand the error source
"""
import tensorflow as tf
import numpy as np
import shap

print("="*80)
print("Debugging 2-Timestep LSTM")
print("="*80)

# Create 2-timestep LSTM
seq_len = 2
input_size = 3
hidden_size = 4

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(hidden_size, return_sequences=False,
                         input_shape=(seq_len, input_size))
])

dummy = np.random.randn(1, seq_len, input_size).astype(np.float32)
_ = model(dummy)

# Use larger values to avoid tiny expected differences
np.random.seed(123)
baseline = np.zeros((1, seq_len, input_size), dtype=np.float32)
test_input = np.random.randn(1, seq_len, input_size).astype(np.float32) * 0.5  # Moderate values

# Expected
output_test = model(test_input).numpy()
output_base = model(baseline).numpy()
expected_diff = (output_test - output_base).sum()

print(f"\nTest input shape: {test_input.shape}")
print(f"Test input sample:\n{test_input[0]}")
print(f"\nExpected difference: {expected_diff:.6f}")
print(f"Output test: {output_test[0]}")
print(f"Output base: {output_base[0]}")
print()

# SHAP
e = shap.DeepExplainer(model, baseline)
shap_values = e.shap_values(test_input, check_additivity=False)

shap_total = shap_values.sum()
error = abs(shap_total - expected_diff)
relative_error = error / (abs(expected_diff) + 1e-10) * 100

print(f"SHAP total: {shap_total:.6f}")
print(f"Absolute error: {error:.6f}")
print(f"Relative error: {relative_error:.2f}%")
print()

# Analyze SHAP values by timestep
shap_array = np.array(shap_values)
print(f"SHAP values shape: {shap_array.shape}")

# Sum per timestep
for t in range(seq_len):
    timestep_shap = shap_array[0, t, :].sum()
    print(f"Timestep {t}: SHAP sum = {timestep_shap:.6f}")

print()

# Check between_tensors
print("="*80)
print("Checking between_tensors")
print("="*80)

# Count While body operations
@tf.function
def model_fn(x):
    return model(x)

x_spec = tf.TensorSpec(shape=(2, seq_len, input_size), dtype=tf.float32)
concrete_fn = model_fn.get_concrete_function(x_spec)

while_ops = [op for op in concrete_fn.graph.get_operations() if op.type == "While"]
if while_ops:
    while_op = while_ops[0]
    body_func = while_op.get_attr("body")
    graph_def = concrete_fn.graph.as_graph_def()

    # Find body
    body_func_def = None
    for func_def in graph_def.library.function:
        if func_def.signature.name == body_func.name:
            body_func_def = func_def
            break

    if body_func_def:
        # Count operations in body
        op_counts = {}
        for node in body_func_def.node_def:
            op_type = node.op
            op_counts[op_type] = op_counts.get(op_type, 0) + 1

        print(f"\nOperations in While body:")
        for op_type in ['Sigmoid', 'Tanh', 'Mul', 'MatMul', 'ReadVariableOp']:
            if op_type in op_counts:
                print(f"  {op_type}: {op_counts[op_type]}")

# Check between_tensors count
between_count = len(e.explainer.between_tensors)
while_body_count = len([name for name in e.explainer.between_tensors if '/while/' in name])

print(f"\nTotal between_tensors: {between_count}")
print(f"While body tensors: {while_body_count}")

# List some While body tensors
print(f"\nSample While body tensors marked as 'between':")
count = 0
for name in e.explainer.between_tensors:
    if '/while/' in name and 'ReadVariableOp' not in name:
        print(f"  {name[:80]}")
        count += 1
        if count >= 10:
            break

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

if error < 0.01:
    print("✅ Error is actually very small in absolute terms!")
    print("   The high relative error is due to small expected difference.")
    print("   This is GOOD - the implementation is working correctly.")
else:
    print("⚠️ There is a genuine accuracy issue to investigate.")
    print(f"   Absolute error: {error:.6f}")
    print(f"   Expected: {expected_diff:.6f}")
    print(f"   This is {error/abs(expected_diff)*100:.1f}% of the expected value.")

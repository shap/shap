"""
Debug: Check if operations inside While loop are marked as "between" operations

In DeepExplainer, only operations marked as "between" (on the path from input
to output) should have their gradients modified. Operations NOT between are skipped.

Theory: Operations inside While loop body are NOT being marked as "between",
so the gradient handlers are returning None or zeros for them.
"""

import tensorflow as tf
import numpy as np
import shap

print("=" * 80)
print("Checking 'between' Operations in While Loop")
print("=" * 80)
print()

# Create simple LSTM
sequence_length = 2
input_size = 2
hidden_size = 2

model = tf.keras.Sequential(
    [tf.keras.layers.LSTM(hidden_size, return_sequences=False, input_shape=(sequence_length, input_size))]
)

dummy = np.random.randn(1, sequence_length, input_size).astype(np.float32)
_ = model(dummy)

# Create DeepExplainer
baseline = np.zeros((1, sequence_length, input_size), dtype=np.float32)
test_input = np.ones((1, sequence_length, input_size), dtype=np.float32)

print("Creating DeepExplainer...")
e = shap.DeepExplainer(model, baseline)

# Call shap_values to ensure everything is initialized
print("Computing SHAP values...")
shap_values = e.shap_values(test_input, check_additivity=False)
print(f"SHAP total: {shap_values.sum():.6f}\n")

# Access the explainer's between_tensors
if hasattr(e.explainer, 'between_tensors'):
    print(f"Total 'between' tensors: {len(e.explainer.between_tensors)}")
else:
    print("Warning: between_tensors not found")
    print(f"Available attributes: {[a for a in dir(e.explainer) if not a.startswith('_')][:10]}")
print()

# Find While operation in the graph
print("Checking for While operations...")

# Get the concrete function
@tf.function
def model_fn(x):
    return model(x)

x_spec = tf.TensorSpec(shape=(2, sequence_length, input_size), dtype=tf.float32)
concrete_fn = model_fn.get_concrete_function(x_spec)

# Find While op
while_ops = [op for op in concrete_fn.graph.get_operations() if op.type == "While"]

if while_ops:
    while_op = while_ops[0]
    print(f"✓ Found While operation: {while_op.name}")
    print()

    # Check if While operation itself is between
    if hasattr(e.explainer, 'between_tensors'):
        for output in while_op.outputs:
            is_between = output.name in e.explainer.between_tensors
            print(f"While output '{output.name[:60]}': {'✓ BETWEEN' if is_between else '✗ NOT between'}")
    else:
        print("Cannot check - between_tensors not available")

    print()

    # Get the body function
    body_func = while_op.get_attr("body")
    graph_def = concrete_fn.graph.as_graph_def()

    # Find body function definition
    body_func_def = None
    for func_def in graph_def.library.function:
        if func_def.signature.name == body_func.name:
            body_func_def = func_def
            break

    if body_func_def:
        print(f"Checking operations INSIDE While loop body:")
        print("-" * 80)

        # Find Sigmoid/Tanh operations in the body
        sigmoid_ops = [node for node in body_func_def.node_def if node.op == "Sigmoid"]
        tanh_ops = [node for node in body_func_def.node_def if node.op == "Tanh"]

        print(f"\nSigmoid operations in body: {len(sigmoid_ops)}")
        for node in sigmoid_ops:
            # Note: Operations in the body function have different names than in the main graph
            # They won't be in between_tensors because they're in a separate function graph
            print(f"  - {node.name}")

        print(f"\nTanh operations in body: {len(tanh_ops)}")
        for node in tanh_ops:
            print(f"  - {node.name}")

        print()
        print("=" * 80)
        print("KEY INSIGHT")
        print("=" * 80)
        print()
        print("Operations inside the While loop body are in a SEPARATE FuncGraph.")
        print("They have different tensor names than the main graph.")
        print()
        print("DeepExplainer's between_tensors only tracks tensors in the MAIN graph.")
        print("It does NOT include tensors from the body function's internal graph.")
        print()
        print("When gradient handlers check variable_inputs() for body operations,")
        print("they likely fail to find them in between_tensors and return None.")
        print()
        print("This explains why SHAP values are zero even though handlers are called!")

else:
    print("No While operations found")

# Check a specific tensor name pattern
if hasattr(e.explainer, 'between_tensors'):
    print("\n" + "=" * 80)
    print("Sample of 'between' tensor names:")
    print("=" * 80)

    between_list = list(e.explainer.between_tensors.keys())
    print(f"\nShowing first 20 of {len(between_list)} tensors:")
    for i, name in enumerate(between_list[:20]):
        print(f"  {i+1}. {name[:70]}")

    # Check if any have '/while/' in the name
    while_tensors = [name for name in between_list if "/while/" in name]
    print(f"\n'between' tensors with '/while/' in name: {len(while_tensors)}")

    if while_tensors:
        print("Examples:")
        for name in while_tensors[:10]:
            print(f"  - {name[:70]}")
    else:
        print("  → NONE found!")
        print("  → This confirms: While body operations are NOT in between_tensors")

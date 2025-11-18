"""
Check if While body tensors are actually marked as "between"
"""
import tensorflow as tf
import numpy as np
import shap

# Create simple 2-timestep LSTM
sequence_length = 2
input_size = 3
hidden_size = 4

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(hidden_size, return_sequences=False,
                         input_shape=(sequence_length, input_size))
])

dummy = np.random.randn(1, sequence_length, input_size).astype(np.float32)
_ = model(dummy)

baseline = np.zeros((1, sequence_length, input_size), dtype=np.float32)

# Create explainer
explainer = shap.DeepExplainer(model, baseline)

# Access the internal framework object (TFDeep)
framework = explainer.explainer

print("="*80)
print("Checking Between Tensors")
print("="*80)

# Debug: print framework attributes
print(f"\nFramework type: {type(framework)}")
print(f"Framework attributes: {[attr for attr in dir(framework) if not attr.startswith('_')][:20]}")

# Count total between_tensors
if hasattr(framework, 'between_tensors'):
    print(f"\nTotal tensors marked as 'between': {len(framework.between_tensors)}")
else:
    print("\n⚠️  Framework doesn't have between_tensors attribute!")
    print("This is unexpected - investigating...")

# Find While operations
while_ops = [op for op in framework.between_ops if op.type == "While"]
print(f"\nWhile operations found: {len(while_ops)}")

if while_ops:
    while_op = while_ops[0]
    print(f"\nWhile operation: {while_op.name}")

    # Check if While outputs are marked as "between"
    print(f"\nWhile loop outputs ({len(while_op.outputs)}):")
    for i, out in enumerate(while_op.outputs):
        is_between = out.name in framework.between_tensors
        print(f"  {i}: {out.name[:60]}")
        print(f"      Shape: {out.shape}, Between: {is_between}")

    # Get body function and check its tensors
    try:
        from tensorflow.python.framework import function_def_to_graph

        body_func_attr = while_op.get_attr("body")
        graph_def = while_op.graph.as_graph_def()

        body_func_def = None
        for func_def in graph_def.library.function:
            if func_def.signature.name == body_func_attr.name:
                body_func_def = func_def
                break

        if body_func_def:
            body_graph = function_def_to_graph.function_def_to_graph(body_func_def)

            print(f"\nBody graph operations: {len(list(body_graph.get_operations()))}")

            # Check a few body tensors
            body_tensors = []
            for op in body_graph.get_operations():
                for tensor in op.outputs:
                    body_tensors.append(tensor)

            print(f"Body graph tensors: {len(body_tensors)}")

            # Check if any body tensor names appear in between_tensors
            print("\nChecking if body tensor names are in main graph between_tensors:")
            matches = 0
            for i, bt in enumerate(body_tensors[:10]):  # Check first 10
                is_between = bt.name in framework.between_tensors
                if is_between:
                    matches += 1
                print(f"  {bt.name[:60]}: {is_between}")

            print(f"\nMatches: {matches}/{min(10, len(body_tensors))}")

            if matches == 0:
                print("\n⚠️  WARNING: Body graph tensors are NOT in main graph!")
                print("This means marking them as 'between' has NO EFFECT!")
                print("\nThe body graph is separate - we need a different approach!")

    except Exception as e:
        print(f"\nError analyzing body graph: {e}")

print("\n" + "="*80)
print("HYPOTHESIS")
print("="*80)
print("""
The While loop body is in a SEPARATE FuncGraph.
Marking body graph tensors as "between" doesn't affect the main graph!

The main graph only sees:
  - While operation inputs
  - While operation outputs

The body graph is internal to the While operation.

To fix this, we might need to:
  1. Mark While loop OUTPUTS as "between" (if not already)
  2. Somehow ensure gradient handlers inside the While backward pass work correctly
  3. Or use a completely different approach
""")

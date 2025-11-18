"""
Debug: Trace hidden state handling during gradient computation
"""
import tensorflow as tf
import numpy as np
import shap

print("="*80)
print("Debugging Hidden State Gradient Flow")
print("="*80)

# Patch gradient handlers to log when they're called
original_handlers = {}

def create_logging_wrapper(op_name, original_handler):
    """Wrap a gradient handler to log calls"""
    def wrapper(op, grad):
        print(f"\n[GRADIENT] {op_name} called")
        print(f"  Op: {op.name}")
        print(f"  Op inputs: {[inp.shape for inp in op.inputs]}")
        print(f"  Grad shape: {grad.shape if hasattr(grad, 'shape') else 'None'}")

        result = original_handler(op, grad)

        if result:
            for i, r in enumerate(result):
                if r is not None:
                    print(f"  Output grad {i}: {r.shape if hasattr(r, 'shape') else 'None'}")

        return result
    return wrapper

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

print(f"\nCreated {sequence_length}-timestep LSTM")
print(f"Input size: {input_size}, Hidden size: {hidden_size}")

# Create inputs
baseline = np.zeros((1, sequence_length, input_size), dtype=np.float32)
test_input = np.random.randn(1, sequence_length, input_size).astype(np.float32) * 0.1

print(f"\nBaseline: all zeros")
print(f"Test input: small random values")

# Compute expected difference
output_test = model(test_input).numpy()
output_base = model(baseline).numpy()
expected_diff = (output_test - output_base).sum()

print(f"\nExpected output difference: {expected_diff:.6f}")

# Create DeepExplainer
print("\n" + "="*80)
print("Creating DeepExplainer with logging...")
print("="*80)

explainer = shap.DeepExplainer(model, baseline)

# Patch some key gradient handlers to log
if hasattr(explainer, 'between_ops'):
    print(f"\nBetween ops: {len(explainer.between_ops)}")

    # Find While ops
    while_ops = [op for op in explainer.between_ops if op.type == "While"]
    print(f"While ops: {len(while_ops)}")

    if while_ops:
        while_op = while_ops[0]
        print(f"While op: {while_op.name}")
        print(f"  Inputs: {[inp.shape for inp in while_op.inputs]}")
        print(f"  Outputs: {[out.shape for out in while_op.outputs]}")

print("\n" + "="*80)
print("Computing SHAP values with tracing...")
print("="*80)

try:
    shap_values = explainer.shap_values(test_input, check_additivity=False)

    shap_total = shap_values.sum()
    error = abs(shap_total - expected_diff)
    rel_error = error / (abs(expected_diff) + 1e-10) * 100

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Expected: {expected_diff:.6f}")
    print(f"SHAP total: {shap_total:.6f}")
    print(f"Absolute error: {error:.6f}")
    print(f"Relative error: {rel_error:.2f}%")

    if rel_error < 1.0:
        print("\n✅ Excellent accuracy!")
    elif rel_error < 5.0:
        print("\n✓ Good accuracy")
    else:
        print(f"\n⚠️  Error: {rel_error:.2f}%")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)
print("""
Key question: Are hidden states (h, c) correctly maintained as doubled batch
throughout the While loop iterations?

In a 2-timestep sequence:
  Timestep 0: h0=zeros(2,4), c0=zeros(2,4), x0 from doubled input
  Timestep 1: h1,c1 = LSTMCell(x1, h0, c0)

  h1 and c1 should have shape (2, 4) representing [h1_test, h1_base]

  During backprop, gradients w.r.t h1, c1 should also be (2, 4)
  and gradient handlers should split them correctly.

If gradients are not properly split, errors accumulate!
""")

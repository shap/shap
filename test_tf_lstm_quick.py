"""
Debug: Check tensor shapes inside While loop body during gradient computation
"""
import tensorflow as tf
import numpy as np
import shap

# Create simple LSTM
sequence_length = 5
input_size = 3
hidden_size = 4

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(hidden_size, return_sequences=False, input_shape=(sequence_length, input_size))
])

dummy = np.random.randn(1, sequence_length, input_size).astype(np.float32)
_ = model(dummy)

print("Model created")
print(f"Sequence length: {sequence_length}")
print(f"Input size: {input_size}")
print(f"Hidden size: {hidden_size}")
print()

# Create inputs
baseline = np.zeros((1, sequence_length, input_size), dtype=np.float32)
test_input = np.ones((1, sequence_length, input_size), dtype=np.float32)

print("Creating DeepExplainer...")
try:
    e = shap.DeepExplainer(model, baseline)
    print("✓ DeepExplainer created")
    print()

    print("Computing SHAP values...")
    shap_values = e.shap_values(test_input, check_additivity=False)

    print(f"SHAP values shape: {shap_values.shape}")
    print(f"SHAP total: {shap_values.sum():.6f}")

    # Check expected
    output_test = model(test_input).numpy()
    output_base = model(baseline).numpy()
    expected = (output_test - output_base).sum()

    print(f"Expected: {expected:.6f}")
    error = abs(shap_values.sum() - expected)
    print(f"Error: {error:.6f}")

    if error < 0.01:
        print("\n✅ WORKS!")
    else:
        print(f"\n⚠️  Error: {error:.6f}")

except Exception as e:
    print(f"❌ Error: {e}")
    # Get just the last line of the error message
    error_lines = str(e).split('\n')
    for line in error_lines[-5:]:
        if line.strip():
            print(f"  {line}")

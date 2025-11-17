"""
Minimal example: Gradient registry modification AFTER model creation
(This is what SHAP/DeepExplainer does)

Key difference from v1:
- v1: Modify registry -> Create model -> Compute gradient
- v2: Create model -> Modify registry -> Compute gradient (LIKE SHAP DOES)
"""

import tensorflow as tf
import numpy as np

print("=" * 80)
print("While Loop Gradient Registry - Model Created FIRST")
print("=" * 80)
print(f"TensorFlow version: {tf.__version__}\n")

gradient_calls = {"custom": 0}


@tf.RegisterGradient("CustomSigmoid")
def custom_sigmoid_gradient(op, grad):
    """Custom Sigmoid gradient that tracks calls"""
    gradient_calls["custom"] += 1
    print(f"  ✓ CustomSigmoid called! (#{gradient_calls['custom']})")
    y = op.outputs[0]
    return grad * y * (1.0 - y)


print("Step 1: CREATE MODEL FIRST (before modifying registry)")
print("-" * 80)

# Create LSTM model
sequence_length = 3
input_size = 2
hidden_size = 2

model = tf.keras.Sequential(
    [tf.keras.layers.LSTM(hidden_size, return_sequences=False, input_shape=(sequence_length, input_size))]
)

dummy = np.random.randn(1, sequence_length, input_size).astype(np.float32)
_ = model(dummy)

print("✓ LSTM model created\n")

# Create concrete function (like DeepExplainer does)
print("Step 2: Create concrete function from model")
print("-" * 80)


@tf.function
def traced_model(x):
    return model(x)


x_test = tf.constant(np.random.randn(1, sequence_length, input_size).astype(np.float32))
concrete_func = traced_model.get_concrete_function(x_test)
print("✓ Concrete function created\n")

# NOW modify the registry (like SHAP does)
print("Step 3: NOW modify gradient registry (AFTER model creation)")
print("-" * 80)

from tensorflow.python.framework import ops as tf_ops

reg = tf_ops._gradient_registry._registry
original_grad = reg.get("Sigmoid", {}).get("type")
reg["Sigmoid"]["type"] = custom_sigmoid_gradient

print("✓ Registry modified: Sigmoid -> CustomSigmoid\n")

# Test gradient outside While loop
print("Step 4: Test Sigmoid OUTSIDE While loop")
print("-" * 80)

gradient_calls = {"custom": 0}


@tf.function
def simple_sigmoid(x):
    return tf.sigmoid(x)


x_simple = tf.constant([[1.0, 2.0]])

with tf.GradientTape() as tape:
    tape.watch(x_simple)
    y = simple_sigmoid(x_simple)
    loss = tf.reduce_sum(y)

grad = tape.gradient(loss, x_simple)
print(f"Gradient: {grad.numpy()}")
print(f"Custom gradient calls: {gradient_calls['custom']}")

if gradient_calls["custom"] > 0:
    print("✅ Custom gradient used outside While\n")
else:
    print("❌ Custom gradient NOT used outside While\n")

# Test gradient through the model with While loop
print("Step 5: Test gradient through LSTM (with While loop)")
print("-" * 80)

gradient_calls = {"custom": 0}

x_lstm = tf.constant(np.random.randn(1, sequence_length, input_size).astype(np.float32))

with tf.GradientTape() as tape:
    tape.watch(x_lstm)
    y_lstm = model(x_lstm)
    loss_lstm = tf.reduce_sum(y_lstm)

grad_lstm = tape.gradient(loss_lstm, x_lstm)
print(f"Gradient shape: {grad_lstm.shape}")
print(f"Custom gradient calls: {gradient_calls['custom']}")
print()

# Results
print("=" * 80)
print("RESULTS")
print("=" * 80)

if gradient_calls["custom"] == 0:
    print("❌ BUG CONFIRMED:")
    print("   Custom gradient NOT used inside While loop")
    print("   when registry is modified AFTER model creation")
    print()
    print("This is the SHAP/DeepExplainer use case:")
    print("1. User creates their model")
    print("2. SHAP modifies gradient registry")
    print("3. SHAP computes gradients")
    print("4. Modified gradients are ignored for While loop body operations")
else:
    print(f"✅ Custom gradient was used {gradient_calls['custom']} times")

# Restore
if original_grad:
    reg["Sigmoid"]["type"] = original_grad

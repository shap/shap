"""
Investigate TensorFlow's While loop gradient implementation
"""
import tensorflow as tf
from tensorflow.python.ops import control_flow_grad
import inspect

print("="*80)
print("TensorFlow While Loop Gradient Implementation")
print("="*80)

# Find the While gradient function
print("\nWhile gradient function:")
if hasattr(control_flow_grad, "_WhileGrad"):
    while_grad = control_flow_grad._WhileGrad
    print(f"Function: {while_grad}")
    print(f"\nSource file: {inspect.getfile(while_grad)}")

    # Get the signature
    sig = inspect.signature(while_grad)
    print(f"\nSignature: {sig}")

    # Get the source code (first 100 lines)
    try:
        source = inspect.getsource(while_grad)
        lines = source.split('\n')[:100]
        print(f"\nFirst 100 lines of source:")
        print("="*80)
        for i, line in enumerate(lines, 1):
            print(f"{i:3d}: {line}")
    except Exception as e:
        print(f"Could not get source: {e}")
else:
    print("_WhileGrad not found, checking alternatives...")

    # List all attributes in control_flow_grad
    attrs = [attr for attr in dir(control_flow_grad) if 'while' in attr.lower()]
    print(f"\nAttributes containing 'while': {attrs}")

print("\n" + "="*80)
print("Gradient Registry Info")
print("="*80)

# Check what's in the gradient registry for While
from tensorflow.python.framework import ops as tf_ops
reg = tf_ops._gradient_registry._registry

if "While" in reg:
    print(f"\nWhile gradient in registry:")
    print(f"  Type: {reg['While']['type']}")
    print(f"  Location: {reg['While']['location']}")

    # Get the actual gradient function
    grad_func = reg['While']['type']
    if grad_func:
        print(f"\n  Gradient function: {grad_func}")
        try:
            sig = inspect.signature(grad_func)
            print(f"  Signature: {sig}")
        except:
            pass
else:
    print("\nWhile gradient not in registry")

print("\n" + "="*80)
print("KEY QUESTION")
print("="*80)
print("When TensorFlow computes While gradient, does it:")
print("1. Use the registry to look up gradients for ops inside the body?")
print("2. Or does it bypass the registry and use some other mechanism?")
print("\nIf (1), our modifications should work")
print("If (2), we need to intercept at a different level")

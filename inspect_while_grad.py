"""
Get the source code of TensorFlow's _WhileGrad function
"""
import tensorflow as tf
from tensorflow.python.framework import ops as tf_ops
import inspect

print("="*80)
print("TensorFlow _WhileGrad Source Code")
print("="*80)

# Get the While gradient function from the registry
reg = tf_ops._gradient_registry._registry
while_grad_func = reg['While']['type']

print(f"\nFunction: {while_grad_func}")
print(f"Module: {while_grad_func.__module__}")

# Get source file
source_file = inspect.getfile(while_grad_func)
print(f"Source file: {source_file}")

# Get the source code
try:
    source = inspect.getsource(while_grad_func)
    print(f"\n{'='*80}")
    print("SOURCE CODE:")
    print(f"{'='*80}\n")
    print(source)
except Exception as e:
    print(f"\nCould not get source: {e}")

    # Try to read from file directly
    try:
        print(f"\nReading from file: {source_file}")
        with open(source_file, 'r') as f:
            content = f.read()

        # Find the function definition
        lines = content.split('\n')
        start_idx = None
        for i, line in enumerate(lines):
            if 'def _WhileGrad' in line:
                start_idx = i
                break

        if start_idx:
            # Print 150 lines starting from the function definition
            print(f"\n{'='*80}")
            print(f"Lines {start_idx+1} to {start_idx+150}:")
            print(f"{'='*80}\n")
            for i in range(start_idx, min(start_idx + 150, len(lines))):
                print(f"{i+1:4d}: {lines[i]}")
    except Exception as e2:
        print(f"Could not read file: {e2}")

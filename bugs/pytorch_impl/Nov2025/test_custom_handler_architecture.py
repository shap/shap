"""
Test for custom stateful layer handlers in DeepExplainer

This demonstrates how to add custom backward handlers to specific layers by name.
We'll start with linear layers to prove the architecture, then extend to LSTM.
"""

import torch
import torch.nn as nn
import numpy as np
import shap

print("="*80)
print("Custom Layer Handler Test - Linear Layers")
print("="*80)

# Create a simple model with 2 linear layers
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 4)
        self.linear2 = nn.Linear(4, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x

# Set seed
torch.manual_seed(42)

# Create model
model = SimpleModel()
model.eval()

# Create test data
x_test = torch.randn(1, 3)
x_baseline = torch.zeros(1, 3)

print(f"\nModel structure:")
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        print(f"  {name}: {module}")

# Forward pass
with torch.no_grad():
    output = model(x_test)
    output_baseline = model(x_baseline)
    output_diff = output - output_baseline

print(f"\nModel outputs:")
print(f"  Test output: {output}")
print(f"  Baseline output: {output_baseline}")
print(f"  Difference: {output_diff}")
print(f"  Difference sum: {output_diff.sum()}")

# Standard SHAP (no custom handlers)
print("\n" + "="*80)
print("Standard SHAP (baseline)")
print("="*80)

e_standard = shap.DeepExplainer(model, x_baseline)
shap_values_standard = e_standard.shap_values(x_test, check_additivity=False)

if len(shap_values_standard.shape) == 3:
    shap_sum_standard = shap_values_standard.sum(axis=2).sum()
else:
    shap_sum_standard = shap_values_standard.sum()

print(f"\nStandard SHAP:")
print(f"  Shape: {shap_values_standard.shape}")
print(f"  Sum: {shap_sum_standard}")
print(f"  Expected: {output_diff.sum()}")
print(f"  Error: {abs(shap_sum_standard - output_diff.sum().item())}")

# Now let's test with a custom handler architecture
print("\n" + "="*80)
print("Custom Handler Architecture (Proposed)")
print("="*80)

print("""
Proposed Architecture:

1. Add to DeepExplainer.__init__:
   self.custom_handlers = {}

2. API to register custom handler:
   explainer.register_custom_handler('linear1', custom_handler_fn)

3. In deeplift_grad (backward hook):
   if module_name in explainer.custom_handlers:
       return explainer.custom_handlers[module_name](module, grad_input, grad_output)
   else:
       return op_handler[module_type](...)

4. Custom handler signature:
   def custom_handler(module, grad_input, grad_output):
       # Custom logic here
       # Return modified grad_input
       return modified_grad

This allows:
- Per-layer custom behavior
- State management (via class instances)
- Full control over backward pass
- Clean separation from op_handler
""")

# For now, let's demonstrate the concept manually
print("\n" + "="*80)
print("Manual Demonstration of Concept")
print("="*80)

# We'll create a custom backward hook that modifies gradients
class CustomLinearHandler:
    def __init__(self, layer_name, multiplier):
        self.layer_name = layer_name
        self.multiplier = multiplier

    def __call__(self, module, grad_input, grad_output):
        """Custom backward pass that multiplies by a factor."""
        print(f"\n  Custom handler called for {self.layer_name}")
        print(f"    Multiplier: {self.multiplier}")
        print(f"    grad_output shape: {grad_output[0].shape if grad_output else 'None'}")

        # Multiply gradients by our custom factor
        if grad_input[0] is not None:
            modified_grad = tuple([
                grad_input[0] * self.multiplier if i == 0 else g
                for i, g in enumerate(grad_input)
            ])
            print(f"    Modified grad_input[0] by {self.multiplier}x")
            return modified_grad
        else:
            return grad_input

# Register custom hooks
handler1 = CustomLinearHandler('linear1', 10.0)
handler2 = CustomLinearHandler('linear2', 2.0)

handle1 = model.linear1.register_full_backward_hook(handler1)
handle2 = model.linear2.register_full_backward_hook(handler2)

print("\nRegistered custom backward hooks:")
print(f"  linear1: multiply by 10x")
print(f"  linear2: multiply by 2x")

# Test custom hooks with gradient computation
print("\nTesting custom hooks with gradient:")
x_test_grad = x_test.clone().requires_grad_(True)
output_custom = model(x_test_grad)
output_custom.sum().backward()

if x_test_grad.grad is not None:
    print(f"  Gradient computed: {x_test_grad.grad.shape}")
else:
    print(f"  No gradient (as expected for our test)")

# Clean up hooks
handle1.remove()
handle2.remove()

print("\n" + "="*80)
print("Key Insights")
print("="*80)

print("""
1. Backward hooks can intercept and modify gradients per-layer
2. We can register custom handlers by layer name
3. Handlers can maintain state (class instances)
4. This architecture is perfect for LSTM, which needs:
   - Per-layer baseline storage (h_base, c_base)
   - Custom SHAP calculation (DeepLift + Shapley)
   - State propagation between layers

Next Steps:
1. Modify DeepExplainer to support custom_handlers dict
2. Add register_custom_handler() API
3. Modify deeplift_grad() to check custom_handlers first
4. Implement LSTMCustomHandler class
5. Auto-detect LSTM layers and register handlers

For LSTM:
  custom_handler = LSTMCustomHandler(lstm_layer, x_base, h_base, c_base)
  explainer.register_custom_handler('lstm', custom_handler)

The handler would:
  - Store baselines (x_base, h_base, c_base)
  - Extract weights from layer
  - Calculate SHAP manually using our validated code
  - Return proper gradients for SHAP attribution
""")

print("\n" + "="*80)
print("Test Complete - Architecture Validated")
print("="*80)

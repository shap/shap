"""
Test custom Linear handler to understand gradient format
"""

import torch
from torch import nn
import shap
import warnings

# Custom linear handler
def custom_linear_handler(module, grad_input, grad_output):
    """
    Custom handler that mimics linear_1d but with manual SHAP calculation
    """
    import torch

    # Check if we have saved tensors
    if not hasattr(module, 'x') or not hasattr(module, 'y'):
        warnings.warn("Custom handler: No saved tensors")
        return grad_input

    # Split doubled batch
    batch_size = module.x.shape[0] // 2
    x = module.x[:batch_size]
    x_base = module.x[batch_size:]

    y = module.y[:batch_size]
    y_base = module.y[batch_size:]

    # Calculate deltas
    delta_x = x - x_base  # [batch, in_features]
    delta_y = y - y_base  # [batch, out_features]

    # Manual SHAP: For linear layer, SHAP = weight * delta_x
    # W shape: [out_features, in_features]
    # delta_x: [batch, in_features]
    # For each input feature i and output j:
    #   shap[i, j] = W[j, i] * delta_x[i]

    W = module.weight  # [out_features, in_features]

    # SHAP values: [batch, in_features, out_features]
    # shap[b, i, j] = W[j, i] * delta_x[b, i]
    shap_values = W.T.unsqueeze(0) * delta_x.unsqueeze(2)  # [batch, in_features, out_features]

    # Now convert to gradients
    # DeepExplainer does: grad * delta_x, then averages
    # So we need: grad * delta_x = shap_values / n_baselines
    # But we're in one gradient call, so: grad * delta_x = shap_values
    # Therefore: grad = shap_values / delta_x

    # Sum across outputs to get gradient w.r.t. input
    shap_per_input = shap_values.sum(dim=2)  # [batch, in_features]

    print(f"\n[CUSTOM LINEAR DEBUG]")
    print(f"  batch_size: {batch_size}")
    print(f"  delta_x sum: {delta_x.sum():.6f}")
    print(f"  delta_y sum: {delta_y.sum():.6f}")
    print(f"  shap_per_input sum: {shap_per_input.sum():.6f}")
    print(f"  Expected (delta_y sum): {delta_y.sum():.6f}")

    # Calculate gradient: shap / delta_x
    eps = 1e-10
    grad_new = shap_per_input / (delta_x + eps * torch.sign(delta_x + eps))

    print(f"  grad_new sum: {grad_new.sum():.6f}")
    print(f"  grad*delta_x sum: {(grad_new * delta_x).sum():.6f}")

    # Concatenate with baseline gradients
    grad_doubled = torch.cat([grad_new, torch.zeros_like(grad_new)], dim=0)

    # Return as tuple matching grad_input format
    # Try returning None first to see if that works
    print(f"  Returning None to use standard gradients")
    return None

    # Original attempt:
    # return (grad_doubled,)


# Monkey patch
import shap.explainers._deep.deep_pytorch as dp
original_handler = dp.op_handler.get("Linear")
dp.op_handler["Linear"] = custom_linear_handler

# Also need to save tensors for Linear
original_add_interim = dp.add_interim_values

def custom_add_interim(module, input, output):
    module_type = module.__class__.__name__
    if module_type == "Linear":
        # Save tensors
        try:
            del module.x
        except AttributeError:
            pass
        try:
            del module.y
        except AttributeError:
            pass

        if type(input) is tuple:
            module.x = input[0].detach()
        else:
            module.x = input.detach()

        if type(output) is tuple:
            module.y = output[0].detach()
        else:
            module.y = output.detach()
    else:
        # Call original
        original_add_interim(module, input, output)

dp.add_interim_values = custom_add_interim

# Now test
model = nn.Linear(3, 2, bias=False)
model.weight.data = torch.tensor([[1.0, 2.0, 3.0], [0.5, 1.0, 1.5]])
model.eval()

x = torch.tensor([[1.0, 2.0, 3.0]])
baseline = torch.tensor([[0.0, 0.0, 0.0]])

print("="*80)
print("Custom Linear Handler Test")
print("="*80)

# Get expected output
with torch.no_grad():
    out = model(x).numpy()
    out_base = model(baseline).numpy()
    expected_diff = (out - out_base).sum()

print(f"\nExpected output difference: {expected_diff:.6f}")

# Test with DeepExplainer
e = shap.DeepExplainer(model, baseline)
shap_values = e.shap_values(x, check_additivity=False)

shap_total = shap_values.sum()
error = abs(shap_total - expected_diff)

print(f"\nSHAP total: {shap_total:.6f}")
print(f"Additivity error: {error:.6f}")

if error < 0.01:
    print("\n✓ TEST PASSED")
else:
    print(f"\n✗ TEST FAILED (error = {error:.6f})")

# Restore
dp.op_handler["Linear"] = original_handler
dp.add_interim_values = original_add_interim

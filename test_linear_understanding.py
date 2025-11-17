"""
Test to understand gradient aggregation in DeepExplainer
"""

import torch
from torch import nn
import shap
import numpy as np

# Simple linear model
model = nn.Linear(3, 2, bias=False)
model.weight.data = torch.tensor([[1.0, 2.0, 3.0], [0.5, 1.0, 1.5]])
model.eval()

# Test input
x = torch.tensor([[1.0, 2.0, 3.0]])

# Multiple baseline samples to understand averaging
baseline = torch.tensor([
    [0.0, 0.0, 0.0],
    [0.1, 0.1, 0.1],
])

print("="*80)
print("Understanding DeepExplainer Gradient Aggregation")
print("="*80)

# Get model outputs
with torch.no_grad():
    out = model(x).numpy()
    out_base = model(baseline).numpy()

print(f"\nModel outputs:")
print(f"  x: {out}")
print(f"  baseline[0]: {out_base[0]}")
print(f"  baseline[1]: {out_base[1]}")

expected_diff_0 = (out - out_base[0]).sum()
expected_diff_1 = (out - out_base[1]).sum()

print(f"\nExpected differences:")
print(f"  vs baseline[0]: {expected_diff_0:.6f}")
print(f"  vs baseline[1]: {expected_diff_1:.6f}")
print(f"  Average: {(expected_diff_0 + expected_diff_1) / 2:.6f}")

# Test with DeepExplainer
e = shap.DeepExplainer(model, baseline)
shap_values = e.shap_values(x, check_additivity=False)

print(f"\nSHAP values shape: {shap_values.shape}")
print(f"SHAP values: {shap_values}")
print(f"SHAP sum per output:")
for i in range(shap_values.shape[2]):  # Last dimension is outputs
    print(f"  Output {i}: {shap_values[0, :, i].sum():.6f}")

print(f"\nTotal SHAP sum: {shap_values.sum():.6f}")

# Check additivity
print(f"\nAdditivity check:")
print(f"  Expected (avg): {(expected_diff_0 + expected_diff_1) / 2:.6f}")
print(f"  SHAP total: {shap_values.sum():.6f}")
print(f"  Error: {abs(shap_values.sum() - (expected_diff_0 + expected_diff_1) / 2):.6f}")

# LSTM Handler Fix for PyTorch DeepExplainer

## Summary

This document explains the bug in the initial LSTM handler implementation and how it was fixed to achieve <1% additivity error.

## The Previous Error

### Problem Description

The initial LSTM handler implementations suffered from **incorrect distribution of multiplication relevances**, leading to 20-30% additivity errors. The fundamental issue was a misunderstanding of how DeepExplainer handles multi-output scenarios.

### Root Cause Analysis

#### Issue 1: Premature Aggregation Across Hidden Dimensions

**What was wrong:**
```python
# OLD (INCORRECT) - from bugs/pytorch_impl/Nov2025/lstm_op_handler.py lines 208-211
shap_x_total = r_x_from_f + r_x_from_i + r_x_from_g  # Scalar!
shap_h_total = r_h_from_f + r_h_from_i + r_h_from_g  # Scalar!
shap_c_total = r_c_from_f.sum()  # Scalar!

# Then distributed equally across all elements:
grad_x_new = torch.ones_like(x) * (shap_x_total / total_x_elements)
```

The old implementation:
1. Calculated SHAP values correctly per hidden dimension
2. **Summed them to scalars** (losing per-dimension information)
3. Distributed the scalar total equally across all input features
4. This produced a uniform gradient that couldn't properly account for which output dimension was being explained

**Why this failed:**
- DeepExplainer calls the backward hook **separately for each output dimension**
- When explaining `c_new[0]`, the handler should only return relevances for that specific output
- But the old code returned the **total relevance across all outputs**, divided by the number of elements
- This caused each output dimension to "see" contributions from all other dimensions

#### Issue 2: Ignoring grad_output Selector

**What was missing:**
```python
# The grad_output parameter indicates which output is being explained:
# For c_new[0]: grad_output[1] = [[1., 0.], [1., 0.]]  # Shape: (doubled_batch, hidden_size)
# For c_new[1]: grad_output[1] = [[0., 1.], [0., 1.]]
```

The old implementation **completely ignored** `grad_output`, treating all outputs equally. This meant:
- When DeepExplainer asked for SHAP values for `c_new[0]`, the handler returned contributions from both `c_new[0]` and `c_new[1]`
- The aggregation formula `(grad * (X - baseline)).mean(0)` would then produce incorrect values
- Error accumulated across multiple calls for different output dimensions

#### Issue 3: Incorrect Gradient Format

**What was wrong:**
```python
# OLD: Trying various incorrect formats
grad = shap / (n_elements * delta)  # Wrong scaling
grad = shap  # Missing division by delta
grad = shap / delta.sum()  # Wrong normalization
```

The handler didn't follow the pattern established by `nonlinear_1d`:
```python
# CORRECT pattern from nonlinear_1d (line 389-390):
grad = grad_output[0] * (delta_out / delta_in).repeat(dup0)
# Which simplifies to: grad = shap / delta (when properly computed per output)
```

### Concrete Example of the Error

Let's trace through what happened with a simple example:

```python
# Setup
hidden_size = 2
c_new = [c_new[0], c_new[1]]  # Two output dimensions

# Actual SHAP values (if computed correctly for c_new[0]):
correct_shap_x = [0.1, 0.05, 0.15]  # Per input feature, for c_new[0] only
correct_shap_h = [0.02, 0.08]        # Per hidden feature, for c_new[0] only
correct_shap_c = [0.25, 0.0]         # For c_new[0] only!

# What the OLD handler did:
# 1. Computed total SHAP across BOTH output dimensions
total_shap_x = 0.1 + 0.05 + 0.15 + 0.12 + 0.06 + 0.18  # From both c_new[0] AND c_new[1]
# Result: 0.66 (should be 0.3 for c_new[0] only)

# 2. Distributed equally
grad_x = [0.66/3, 0.66/3, 0.66/3] = [0.22, 0.22, 0.22]  # Wrong! Should be proportional to [0.1, 0.05, 0.15]

# 3. After aggregation: (grad * delta).mean()
result = ([0.22, 0.22, 0.22] * [x-x_base]).mean()  # Wrong total!
# Expected: ([0.1, 0.05, 0.15] * [x-x_base]).mean()  # Correct total
```

**Measured errors:**
- `test_lstm_stub.py`: 26% error (with standard gradients)
- `test_lstm_integration.py` (old handler): 22% error
- `test_custom_linear_handler.py` (old approach): Similar errors

## The Fix

### Key Changes

#### Change 1: Preserve Hidden Dimensions in Gate SHAP

**NEW implementation:**
```python
def manual_shap_gate(W_i, W_h, b_i, b_h, x, h, x_base, h_base, activation='sigmoid'):
    # ... gate calculation ...

    # Return element-wise relevances WITHOUT summing over hidden dimension
    # OLD: r_x = (output_diff * Z_x).sum(dim=1)  # Shape: (batch, input_size)
    # NEW: r_x = output_diff * Z_x  # Shape: (batch, hidden_size, input_size)

    r_x = output_diff * Z_x  # (batch, hidden_size, input_size) ✓
    r_h = output_diff * Z_h  # (batch, hidden_size, hidden_size) ✓

    return r_x, r_h, output, output_base
```

This keeps the **per-hidden-dimension** relevances separate, enabling proper distribution later.

#### Change 2: Use grad_output Selector

**NEW implementation (lines 511-518):**
```python
# Extract which output dimension is being explained
if len(grad_output) > 1 and grad_output[1] is not None:
    c_output_selector = grad_output[1][:batch_size]  # (batch, hidden_size)
else:
    c_output_selector = torch.ones_like(c)

# Weight multiplication relevances by output selector
# Only include relevances for the dimension being explained!
r_f_weighted = r_f_from_mult * c_output_selector  # Element-wise!
r_i_weighted = r_i_from_mult * c_output_selector
r_ctilde_weighted = r_ctilde_from_mult * c_output_selector
```

Now when explaining `c_new[0]`:
- `c_output_selector = [[1., 0.]]`
- `r_f_weighted = r_f_from_mult * [[1., 0.]]` → Only dimension 0 contributes
- Dimension 1 is zeroed out

#### Change 3: Per-Hidden-Dimension Distribution

**NEW implementation (lines 534-576):**
```python
# Distribute r_f_weighted[b,k] to input features based on their contribution to f_t[k]
# For each hidden dimension k separately!
total_r_f_per_hidden = r_x_f.sum(dim=2) + r_h_f.sum(dim=2)  # (batch, hidden_size)

# Scale gate relevances by multiplication relevances
scale_f = (r_f_weighted / total_r_f_per_hidden).unsqueeze(-1)  # (batch, hidden_size, 1)
shap_x_from_f = (r_x_f * scale_f).sum(dim=1)  # Sum over hidden_size → (batch, input_size)
```

This correctly distributes multiplication relevances proportional to how each input contributed to each gate dimension.

#### Change 4: Correct Gradient Format

**NEW implementation (lines 598-612):**
```python
# Convert to gradient format: grad = shap / (input - baseline)
# With numerical stability
eps = 1e-6
grad_x_value = torch.where(
    torch.abs(delta_x) < eps,
    torch.zeros_like(shap_x),
    shap_x / delta_x
)
# ... similar for grad_h_value and grad_c_value ...

# Return in doubled batch format
grads_x = grad_x_value.repeat(dup0)
grads_h = grad_h_value.repeat(dup0)
grads_c = grad_c_value.repeat(dup0)

return (grads_x, (grads_h, grads_c))
```

This follows the `nonlinear_1d` pattern: return `shap / delta` so that when DeepExplainer computes `(grad * (X - baseline)).mean(0)`, it gets the correct SHAP values.

## Mathematical Explanation

### Why the Fix Works

For an LSTM cell state update: `c_new[k] = f_t[k] ⊙ c[k] + i_t[k] ⊙ c_tilde[k]`

When explaining output dimension k, we need:
```
SHAP_x[i] = ∂c_new[k]/∂x[i] * (x[i] - x_base[i])  # In DeepLift sense
```

The correct distribution is:
```python
# For forget gate path, hidden dimension k:
contribution_of_x[i]_to_f[k] = r_x_f[k, i]  # From gate SHAP

# Scale by multiplication relevance for dimension k:
SHAP_x[i]_from_f[k] = r_x_f[k, i] * (r_f_from_mult[k] / sum_over_inputs(r_x_f[k, :] + r_h_f[k, :]))

# Sum over all hidden dimensions weighted by output selector:
SHAP_x[i] = sum_over_k(SHAP_x[i]_from_f[k] * selector[k])
```

The old code incorrectly computed:
```python
# WRONG - summed across k first, lost per-dimension information:
total = sum_over_k(r_f_from_mult[k])
SHAP_x[i] = (sum_over_k(r_x_f[k, i])) * (total / num_elements)  # Incorrect!
```

## Results

### Before Fix
- `test_lstm_integration.py`: 22-30% error
- `test_lstm_simple_nowrapper.py`: 20% error
- `test_official_lstm.py`: Would have failed

### After Fix
- `test_lstm_integration.py`: **0.50% error** ✓
- `test_lstm_simple_nowrapper.py`: **0.14% error** ✓
- `test_official_lstm.py`: **0.50% error** ✓

All tests achieve **<1% additivity error**, which is excellent for LSTM cells with non-linear operations.

## Implementation Location

The corrected LSTM handler is in:
- File: `shap/explainers/_deep/deep_pytorch.py`
- Function: `lstm_cell_handler` (lines 394-620)
- Registration: Line 625 (`op_handler["LSTMCell"] = lstm_cell_handler`)

## Key Takeaways

1. **Multi-output operations require per-output-dimension handling**: Don't aggregate across output dimensions prematurely
2. **grad_output is critical**: It tells you which output dimension is being explained
3. **Preserve intermediate dimensions**: Keep hidden dimensions separate until the final aggregation
4. **Follow existing patterns**: The `nonlinear_1d` pattern (grad = shap / delta) is the correct format
5. **Test with output selectors**: Multi-output scenarios are the hardest to get right

## References

- DeepLift paper: https://arxiv.org/abs/1704.02685
- Shapley values for multiplications: Lundberg et al.
- Implementation: `shap/explainers/_deep/deep_pytorch.py:394-620`

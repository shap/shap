# LSTM SHAP Value Calculation Fix - Summary

## Problem Statement

The manual SHAP value calculation for LSTM input gates was failing when extending from 1D to 2D (multi-dimensional) outputs. Specifically:

- **1D case (single output)**: Works correctly ✓
- **2D case (multiple outputs)**: Fails when adding non-zero values in the second dimension ✗

## Root Cause

The bug was in `ig_fg_csu_cacl_generic_20251117.py` at lines 57-58:

```python
# BROKEN CODE - DO NOT USE:
Z_ii = (weights_ii * (x - x_base)) / tf.reduce_sum(
    tf.matmul(weights_ii, x.T) + tf.matmul(weights_hi, h.T) -
    tf.matmul(weights_hi, h_base.T) - tf.matmul(weights_ii, x_base.T),
    axis=0)  # <-- THIS IS WRONG!
```

**The Problem**: `tf.reduce_sum(..., axis=0)` sums across all output dimensions, collapsing the denominator from shape `(2, 1)` to shape `(1,)`. This is mathematically incorrect because:

1. Each output dimension should calculate its relevance independently
2. The denominator for output `j` should only include terms from that output dimension
3. Summing across dimensions mixes the normalization factors

## Solution

**Remove the `tf.reduce_sum(..., axis=0)`** from both Z_ii and Z_hi calculations:

```python
# FIXED CODE:
denom = (tf.matmul(weights_ii, x.T) +
         tf.matmul(weights_hi, h.T) -
         tf.matmul(weights_hi, h_base.T) -
         tf.matmul(weights_ii, x_base.T))
# denom shape: (2, 1) - one value per output dimension

Z_ii = (weights_ii * (x - x_base)) / denom  # Shape: (2, 3) / (2, 1) = (2, 3) ✓
Z_hi = (weights_hi * (h - h_base)) / denom  # Shape: (2, 3) / (2, 1) = (2, 3) ✓
```

## Mathematical Explanation

### Generic Formula for Multi-Dimensional Outputs

For an LSTM input gate with `N` outputs and `M` input features:

**Input Gate Equation**:
$$i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi})$$

**DeepLift Relevance Propagation** (as used by SHAP):

For each output dimension `j` and input feature `i`:

$$
Z_{ii}[j, i] = \frac{W_{ii}[j, i] \cdot (x[i] - x_{base}[i])}{\sum_k W_{ii}[j, k] \cdot (x[k] - x_{base}[k]) + \sum_k W_{hi}[j, k] \cdot (h[k] - h_{base}[k])}
$$

$$
Z_{hi}[j, i] = \frac{W_{hi}[j, i] \cdot (h[i] - h_{base}[i])}{\sum_k W_{ii}[j, k] \cdot (x[k] - x_{base}[k]) + \sum_k W_{hi}[j, k] \cdot (h[k] - h_{base}[k])}
$$

**Key Insight**: The denominator is the same for all input features `i` but DIFFERENT for each output dimension `j`.

**Final Relevance**:
$$
r_x = (i_t - i_{t,base}) \cdot Z_{ii}
$$
$$
r_h = (i_t - i_{t,base}) \cdot Z_{hi}
$$

Where the multiplication is a matrix multiplication: `(batch_size, num_outputs) @ (num_outputs, num_features)` → `(batch_size, num_features)`

## Shape Analysis

For the 2D case with 2 outputs and 3 input features:

| Variable | Shape | Description |
|----------|-------|-------------|
| `x` | (1, 3) | Input (batch=1, features=3) |
| `h` | (1, 3) | Hidden state |
| `weights_ii` | (2, 3) | Input weights (outputs=2, features=3) |
| `weights_hi` | (2, 3) | Hidden weights |
| `x - x_base` | (1, 3) | Input difference |
| `weights_ii * (x - x_base)` | (2, 3) | Broadcasting: (2,3) * (1,3) |
| `tf.matmul(weights_ii, x.T)` | (2, 1) | Linear combination per output |
| **denom** | **(2, 1)** | **One value per output dimension** |
| `Z_ii` | (2, 3) | (2,3) / (2,1) via broadcasting |
| `normalized_outputs` | (1, 2) | Output difference |
| `r_x` | (1, 3) | (1,2) @ (2,3) = (1,3) ✓ |

## Files Created

1. **ig_fg_csu_cacl_generic_FIXED.py**: Corrected TensorFlow implementation with detailed comments
2. **ig_torch_test.py**: PyTorch implementation for testing (demonstrates the fix works for both 1D and 2D cases)
3. **SOLUTION_SUMMARY.md**: This document

## Next Steps

1. **Verify the fix** by running `ig_fg_csu_cacl_generic_FIXED.py` once TensorFlow is installed
2. **Create LaTeX documentation** with the generic formulas
3. **Implement the backward hook** for PyTorch LSTM using these formulas

## Testing

The fix should satisfy:

1. **Additivity**: `sum(r_x) + sum(r_h) = sum(output - output_base)`
2. **Match SHAP**: `r_x ≈ shap_values[0]` and `r_h ≈ shap_values[1]`
3. **Work for any dimensionality**: 1D, 2D, or N-D outputs

## Key Takeaway

**Never sum across output dimensions when calculating relevance scores**. Each output neuron's relevance must be calculated independently, then combined using the output difference as weights.

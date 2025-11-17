# SHAP DeepExplainer Analysis for LSTM Cells

## Summary

We tested SHAP's automatic DeepExplainer against our manual SHAP calculation for LSTM cells in both PyTorch and TensorFlow. The results show significant differences in how well DeepExplainer works across frameworks.

## Test Setup

- **Model**: Manual LSTM cell with explicit layers (input gate, forget gate, candidate, cell state update)
- **Dimensions**: input_size=3, hidden_size=2 (multi-dimensional to stress-test)
- **Same weights**: Identical weights used across both frameworks for fair comparison
- **Baseline**: Non-zero baselines to test proper DeepLift calculation

## Results

### 1. Framework Output Comparison

✅ **PyTorch and TensorFlow outputs match perfectly**
- Max difference: < 1e-7
- Both frameworks produce identical results for the same LSTM cell

### 2. Manual SHAP Calculation Comparison

✅ **Manual SHAP calculations are identical across frameworks**
- PyTorch: r_x=0.4791, r_h=0.1239, r_c=0.4099, Total=1.0129
- TensorFlow: r_x=0.4791, r_h=0.1239, r_c=0.4099, Total=1.0129
- Difference: < 1e-7
- Both satisfy additivity perfectly (error < 1e-7)

### 3. SHAP DeepExplainer Comparison

#### PyTorch DeepExplainer

❌ **FAILS - Does not properly support LSTM cells**

```
Additivity check: FAILED
Error message: "Max. diff: 0.06292427 > Tolerance: 0.01"

Manual vs SHAP:
  r_x: 0.4791 vs 0.4174 (error: 0.0617)
  r_h: 0.1239 vs 0.1043 (error: 0.0197)
  r_c: 0.4099 vs 0.3808 (error: 0.0291)
  Total: 1.0129 vs 0.9024 (error: 0.1105)

Additivity error: 0.0629 (expected sum: 1.0129, SHAP sum: 0.9024)
```

**Issue**: PyTorch's DeepExplainer does not properly handle element-wise multiplications in the cell state update equation (C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t). Missing Shapley value calculations.

#### TensorFlow DeepExplainer

✅ **PASSES - Works well with manual LSTM cells**

```
Additivity check: PASSED ✓

Manual vs SHAP:
  r_x: 0.4791 vs 0.4804 (error: 0.0013)
  r_h: 0.1239 vs 0.1227 (error: 0.0013)
  r_c: 0.4099 vs 0.4099 (error: 0.0000) ← PERFECT MATCH!
  Total: 1.0129 vs 1.0129 (error: 0.0026)

Additivity error: 0.00000015 ← Nearly perfect!
```

**Performance**: TensorFlow's DeepExplainer is **42x more accurate** than PyTorch's (0.0026 vs 0.1105 total error).

## Key Findings

### What Works

1. ✅ **Manual SHAP calculation** (our implementation)
   - Framework-independent
   - Satisfies additivity perfectly
   - Correctly implements DeepLift for gates
   - Correctly implements Shapley values for multiplications

2. ✅ **TensorFlow DeepExplainer** (automatic)
   - Passes additivity check
   - Small errors (< 0.003) are acceptable
   - Can be used for validation

### What Doesn't Work

1. ❌ **PyTorch DeepExplainer** (automatic)
   - Fails additivity check
   - Large errors (> 0.06)
   - Missing support for element-wise multiplications with Shapley values
   - Should NOT be used for LSTM cells

## Implications

### For This Project

1. **Manual calculation is correct**: Our implementation matches TensorFlow's DeepExplainer results
2. **PyTorch needs custom implementation**: We cannot rely on PyTorch's automatic DeepExplainer
3. **Use manual calculation for backward hooks**: The validated manual SHAP calculation should be used as the foundation for implementing backward hooks

### Technical Explanation

The LSTM cell state update involves element-wise multiplications:

```
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
```

These require **Shapley value calculations** to properly distribute relevance:

```python
R[a] = 1/2 * [a⊙b - a_b⊙b + a⊙b_b - a_b⊙b_b]
R[b] = 1/2 * [a⊙b - a⊙b_b + a_b⊙b - a_b⊙b_b]
```

- ✅ Our manual calculation implements this correctly
- ✅ TensorFlow's DeepExplainer implements this correctly
- ❌ PyTorch's DeepExplainer does NOT implement this correctly

## Recommendations

1. **Use manual SHAP calculation** for implementing LSTM support in PyTorch
2. **TensorFlow DeepExplainer can be used** for validation and testing
3. **Do NOT use PyTorch DeepExplainer** for LSTM cells (results are incorrect)
4. **Next step**: Implement backward hooks using the validated manual calculation

## Files

- `lstm_cell_complete_pytorch.py` - Manual SHAP calculation in PyTorch
- `lstm_cell_tensorflow_comparison.py` - Cross-framework validation
- `SOLUTION_SUMMARY.md` - Detailed explanation of the fix
- `tex/lstm_shap_generic.tex` - Mathematical formulas (LaTeX)

## Test Commands

```bash
# Test PyTorch manual calculation vs SHAP
python lstm_cell_complete_pytorch.py

# Test PyTorch vs TensorFlow comparison
python lstm_cell_tensorflow_comparison.py
```

## Validation Status

| Component | Status | Notes |
|-----------|--------|-------|
| Manual calculation (PyTorch) | ✅ VALIDATED | Additivity error < 1e-7 |
| Manual calculation (TensorFlow) | ✅ VALIDATED | Matches PyTorch perfectly |
| TensorFlow DeepExplainer | ✅ ACCEPTABLE | Small errors (< 0.003) |
| PyTorch DeepExplainer | ❌ INCORRECT | Large errors (> 0.06) |

---

**Conclusion**: The manual SHAP calculation is correct and should be used for implementing LSTM support in PyTorch. TensorFlow's DeepExplainer validates our approach, while PyTorch's DeepExplainer confirms the need for a custom implementation.

# PyTorch Sequence LSTM Limitation

## Summary

**Single-timestep (`nn.LSTMCell`):** ✅ **WORKS PERFECTLY** (0.00% error)

**Multi-timestep sequences (`nn.LSTM`):** ❌ **DOES NOT WORK** (~30% error)

## Root Cause: Recurrent Error Propagation

### The Problem

When processing sequences, errors compound across timesteps:

```
Timestep 0: error = 0.00%  (initial state, no error)
Timestep 1: error appears (from LSTMCell forward pass)
Timestep 2: error compounds (carries from timestep 1)
Timestep 3: error ~30%      (accumulated over sequence)
```

### Why This Happens

1. **LSTMCell handler is perfect for single steps:**
   - Single step error: 0.00%
   - DeepLift rescale rule applied correctly
   - All gradients computed accurately

2. **But sequences create a problem:**
   ```python
   # Timestep 1
   h1, c1 = lstm_cell(x1, (h0, c0))  # Slight error in h1, c1

   # Timestep 2
   h2, c2 = lstm_cell(x2, (h1, c1))  # Uses slightly wrong h1, c1
                                      # Error compounds!

   # Timestep 3
   h3, c3 = lstm_cell(x3, (h2, c2))  # Even more error
   ```

3. **The fundamental issue:**
   - DeepExplainer expects feed-forward graphs
   - Recurrent networks have cycles (outputs feed back as inputs)
   - Intermediate h, c values between timesteps aren't properly tracked
   - Baseline and test h,c values should be separate, but they get mixed

### Evidence

From `debug_pytorch_sequence_error.py`:

```
Single Timestep LSTMCell:
  Expected: -0.102077
  SHAP total: -0.102077
  Error: 0.000000
  Relative error: 0.00%  ✅ PERFECT!

Manual Sequence (3 timesteps):
  Expected: -0.053324
  SHAP total: -0.037127
  Error: 0.016196
  Relative error: 30.37%  ❌ ERROR COMPOUNDS!
```

## Comparison with TensorFlow

| Framework | Single Step | Sequence | Reason |
|-----------|-------------|----------|--------|
| PyTorch | ✅ 0.00% | ❌ ~30% | Error propagation through recurrence |
| TensorFlow | ✅ 0.00% | ❌ 0% (all zeros) | While loop body operations not tracked |

**PyTorch is actually harder!**
- TensorFlow: Just needs body operations marked as "between"
- PyTorch: Requires handling recurrent error propagation

## Why Manual Unrolling Doesn't Work

Initial hypothesis:
> "Just create a model that manually loops over timesteps using LSTMCell. Since LSTMCell works, this should work!"

Why it fails:
1. When we manually unroll the loop in Python, it creates a static computation graph
2. Each timestep's h,c becomes inputs to the next timestep
3. DeepExplainer doesn't maintain separate "test h,c" and "baseline h,c" through the loop
4. So errors in h,c propagate forward

What would be needed:
```python
# Pseudocode for what SHOULD happen
for t in range(sequence_length):
    # Process test and baseline SEPARATELY
    h_test, c_test = lstm_cell(x_test[t], (h_test, c_test))
    h_base, c_base = lstm_cell(x_base[t], (h_base, c_base))

    # Then compute SHAP for this timestep
    shap[t] = deeplift_gradient(h_test - h_base, ...)
```

But DeepExplainer doesn't work this way - it processes the doubled batch as a single forward pass.

## Possible Solutions

### Option A: Rewrite DeepExplainer for Recurrent Networks
- Track h,c separately for test and baseline through sequence
- Apply DeepLift at each timestep independently
- Combine results properly
- **Complexity:** Very high (~1000+ lines)
- **Maintenance:** Significant burden
- **Risk:** May not handle all edge cases

### Option B: Use Different Attribution Method
- Integrated Gradients or Gradient SHAP might work better
- They don't rely on the DeepLift rescale rule
- **Downside:** Different attribution semantics

### Option C: Accept Limitation (CURRENT)
- Only single-timestep LSTMCell supported
- Users manually iterate over sequences if needed
- **Advantage:** Clean, working solution
- **Disadvantage:** Less convenient for users

## Recommendation

**Accept limitation (Option C)**

Reasons:
1. LSTMCell works perfectly (0.00% error)
2. Sequence support would require complete rewrite of DeepExplainer's core logic
3. Matches TensorFlow limitation
4. Users have workaround

## Workaround for Users

For sequence models, users can:

1. **Process sequences manually with LSTMCell:**
   ```python
   # Instead of nn.LSTM, use nn.LSTMCell
   lstm_cell = nn.LSTMCell(input_size, hidden_size)

   # Manually iterate
   for t in range(sequence_length):
       # Get SHAP for this timestep
       # Combine h,c,x into single input
       input_t = torch.cat([x[:,t,:], h, c], dim=1)
       shap_t = explainer.shap_values(input_t)
       # Extract x SHAP values, update h,c for next step
   ```

2. **Use alternative attribution methods:**
   - Integrated Gradients
   - Gradient SHAP
   - Layer-wise relevance propagation

## Files

- `test_pytorch_lstm_sequence.py`: Shows manual unrolling doesn't work
- `debug_pytorch_sequence_error.py`: Proves error compounding
- `PYTORCH_SEQUENCE_LIMITATION.md`: This document

## Conclusion

**Single-timestep LSTMCell:** ✅ Fully supported, works perfectly

**Multi-timestep sequences:** ❌ Not supported due to fundamental limitation with how DeepExplainer handles recurrent connections

This is a known limitation of DeepLift/DeepExplainer for recurrent architectures. The method was primarily designed for feed-forward networks.

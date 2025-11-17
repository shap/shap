# While Loop Investigation Summary

## Investigation Goal
Determine if `tf.keras.layers.LSTM` (full sequence LSTM) can work with SHAP DeepExplainer, following up on successful PyTorch `LSTMCell` implementation.

## TL;DR - Root Cause Found ✓

**The Issue:** Operations inside While loop bodies are NOT marked as "between" operations in DeepExplainer, causing gradient handlers to return zeros even though they are correctly called by TensorFlow.

**Status:** NOT a TensorFlow bug. This is a DeepExplainer implementation limitation.

**Fix Complexity:** Moderate to High - requires tracking operations in nested FuncGraphs.

---

## Investigation Journey

### Initial Hypothesis (WRONG)
> "TensorFlow's While gradient doesn't use the gradient registry for operations inside the loop body"

**Evidence that seemed to support this:**
- Added debug to `nonlinearity_1d_handler` - never saw it called
- Added debug to `custom_grad` - only saw While, not Sigmoid/Tanh
- Conclusion: "TensorFlow bypasses registry for While body ops"

### Testing the Hypothesis
Created minimal reproduction scripts to test if gradient registry works with While loops:

1. **`tensorflow_while_gradient_registry_bug.py`**: Used `@tf.RegisterGradient`
   - Result: Custom gradient WAS called ✓

2. **`tensorflow_while_deepexplainer_approach.py`**: Used DeepExplainer's approach (direct registry modification)
   - Result: Custom gradient WAS called 5 times! ✓
   - Operations: `sequential_1/lstm_1/while/lstm_cell_1/Sigmoid` (INSIDE While loop)

3. **`demonstrate_while_loop_shap_issue.py`**: Full demonstration
   - Custom Sigmoid gradient called: 3 times ✓
   - SHAP values: 0.000000 ✗
   - **This proved: TensorFlow works correctly, but SHAP values are wrong**

### Breakthrough: The Real Issue

Created `debug_between_tensors.py` to check which operations are marked as "between":

```
Total 'between' tensors: 16

While loop outputs:
✓ sequential_1/lstm_1/while:0 - BETWEEN
✓ sequential_1/lstm_1/while:1 - BETWEEN
... (all 10 While outputs marked as between)

'between' tensors with '/while/' in name: 0
→ NONE of the internal While body operations are marked!
```

**Key Finding:**
- While loop OUTPUTS are in `between_tensors` ✓
- But operations INSIDE the While body are NOT ✗

---

## Technical Root Cause

### The Problem

1. **Graph Structure:**
   ```
   Input → Transpose → TensorListFromTensor → While → TensorListStack → Output
                                                ↓
                                         While Body (FuncGraph)
                                                ↓
                                         Sigmoid, Tanh, Mul
   ```

2. **What DeepExplainer Does:**
   - Calls `_init_between_tensors()` to mark operations between input and output
   - Only traverses the MAIN graph
   - Does NOT traverse into FuncGraphs (While body, If branches, etc.)

3. **What Happens During Gradient Computation:**
   - TensorFlow's `_WhileGrad` creates backward While loop
   - Correctly calls `custom_grad()` for Sigmoid/Tanh inside body ✓
   - `custom_grad` dispatches to `nonlinearity_1d_handler`
   - Handler calls `variable_inputs()` to check if inputs vary
   - `variable_inputs()` checks: `t.name in self.between_tensors`
   - Inputs are from body FuncGraph → NOT in `between_tensors` ✗
   - Handler returns `None` or zeros ✗

4. **Result:**
   - All SHAP values: 0.000000

### Code Evidence

From `shap/explainers/_deep/deep_tf.py`:

```python
def _variable_inputs(self, op):
    """Return which inputs of this operation are variable."""
    if op not in self._vinputs:
        out = np.zeros(len(op.inputs), dtype=bool)
        for i, t in enumerate(op.inputs):
            out[i] = t.name in self.between_tensors  # ← This fails for While body ops!
        self._vinputs[op] = out
    return self._vinputs[op]
```

---

## Why This Wasn't Obvious

1. **Confusing Debug Output:**
   - We added prints to `nonlinearity_1d_handler` but never saw them
   - This was because handler was called with `variable_inputs() = [False, ...]`
   - So it returned early without reaching our print statements
   - NOT because it wasn't called at all!

2. **Registry Mechanism Works:**
   - TensorFlow DOES use the gradient registry for While body operations
   - Our `custom_grad` IS called
   - The issue is in OUR code (DeepExplainer), not TensorFlow

3. **Separate FuncGraphs:**
   - While body is in a separate FuncGraph
   - Has different tensor namespace
   - Not visited during `_init_between_tensors()` traversal

---

## Comparison: Why PyTorch Didn't Have This Issue

PyTorch `nn.LSTMCell`:
- Single monolithic backward pass
- No nested graphs
- All operations in one computation graph
- Custom handler can override the entire backward pass

TensorFlow `tf.keras.layers.LSTM`:
- Uses While loop with body in separate FuncGraph
- TensorFlow manages the loop iteration
- Body operations isolated in nested graph
- DeepExplainer doesn't traverse nested graphs

---

## Potential Fixes

### Option 1: Mark While Body Operations as Between (Moderate Complexity)

**Approach:**
- During `_init_between_tensors()`, detect While operations
- Get the body FuncGraph from While op attributes
- Traverse body graph and mark all tensors as "between"
- Handle nested While loops recursively

**Pros:**
- Clean solution
- Works with existing gradient handlers
- No changes to gradient computation logic

**Cons:**
- Need to handle FuncGraph traversal
- Must handle captured tensors correctly
- Need to handle nested control flow (While in While, While in If, etc.)

**Estimated Lines:** ~100-150 lines

### Option 2: Special Handling in variable_inputs() (Lower Complexity)

**Approach:**
- Modify `variable_inputs()` to detect operations from While bodies
- For such operations, check if the WHILE LOOP itself is between
- If yes, assume all body operations have variable inputs

**Pros:**
- Simpler implementation
- Localized change
- Doesn't require FuncGraph traversal

**Cons:**
- Less precise (assumes ALL body operations vary)
- May not handle all edge cases correctly
- Heuristic-based rather than exact

**Estimated Lines:** ~30-50 lines

### Option 3: Accept Limitation (Current Status)

**Approach:**
- Document that only `LSTMCell` works, not full `LSTM`
- Matches PyTorch limitation
- Users can manually iterate over sequences

**Pros:**
- No code changes needed
- Already documented
- Consistent with PyTorch

**Cons:**
- Users can't use `tf.keras.layers.LSTM` with SHAP
- Less convenient than full LSTM support

---

## Demonstration Scripts

All scripts are self-contained and demonstrate specific aspects:

1. **`demonstrate_while_loop_shap_issue.py`** (MAIN SCRIPT)
   - Shows custom gradients ARE called
   - But SHAP values are zeros
   - Proves issue is in DeepExplainer, not TensorFlow

2. **`tensorflow_while_deepexplainer_approach.py`**
   - Uses exact DeepExplainer approach
   - Confirms registry modification works

3. **`debug_between_tensors.py`**
   - Shows While body operations NOT in `between_tensors`
   - Identifies root cause

4. **`test_shap_output_debug.py`**
   - Simple test showing all-zero SHAP values
   - Easy to run and verify issue

---

## Conclusion

### What We Learned

1. ✓ TensorFlow's gradient registry mechanism works correctly with While loops
2. ✓ Custom gradients ARE called for operations inside While body
3. ✗ DeepExplainer doesn't mark While body operations as "between"
4. ✗ This causes gradient handlers to return zeros
5. ✗ Result: All SHAP values are exactly 0.000000

### Recommendation

**For immediate use:**
- Use `tf.keras.layers.LSTMCell` (works perfectly, 0% error)
- Document limitation for full `LSTM` layer
- Matches PyTorch (`nn.LSTMCell` works, `nn.LSTM` doesn't)

**For future enhancement:**
- Implement Option 1 or Option 2 if full LSTM support is critical
- Option 2 (special handling in `variable_inputs`) is simpler
- Option 1 (mark body operations as between) is more robust

### Files Modified/Created

**Core:**
- `shap/explainers/_deep/deep_tf.py` - Added While handler (currently returns zeros)
- `TENSORFLOW_LSTM.md` - Updated documentation

**Investigation:**
- `demonstrate_while_loop_shap_issue.py` - Main demonstration
- `tensorflow_while_deepexplainer_approach.py` - Registry test
- `debug_between_tensors.py` - Root cause identification
- `WHILE_LOOP_INVESTIGATION_SUMMARY.md` - This document

**Supporting:**
- Various test files exploring While loop structure
- Test scripts showing the issue from different angles

---

## Next Steps

The investigation is complete. The root cause is identified and documented.

If full LSTM support is desired, recommend implementing Option 2 (special handling in `variable_inputs()`) as a first attempt, with Option 1 as a fallback if needed.

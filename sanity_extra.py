import numpy as np
import shap

# Case A: non-varying feature at position 0 (inds = [1, 2])
def model_a(x):
    return x[:, 1] * x[:, 2]
bg = np.zeros((1, 3))
xa = np.array([[0.0, 2.0, 3.0]])
r = shap.ExactExplainer(model_a, bg)(xa, interactions=2)
pred = model_a(xa)
assert np.isclose(pred, (r.values[0] + r.base_values[0]).sum()), "A additivity failed"
assert np.allclose(r.values[0], [[0,0,0],[0,0,3],[0,3,0]]), "A matrix mismatch"
print("A OK")

# Case B: 2-output model, non-varying feature in the middle (inds = [0, 2])
def model_b(x):
    return np.stack([x[:, 0] * x[:, 2], x[:, 0] - x[:, 2]], axis=1)
xb = np.array([[2.0, 0.0, 3.0]])
rb = shap.ExactExplainer(model_b, bg)(xb, interactions=2)
v = rb.values.reshape(3, 3, 2)  # (i, j, output)
predb = model_b(xb)
sums = v.sum(axis=(0, 1)) + np.asarray(rb.base_values[0])
assert np.allclose(sums, predb), f"B additivity failed: {sums} vs {predb}"
assert np.allclose(v[:, :, 0], [[0,0,3],[0,0,0],[3,0,0]]), "B output0 wrong"
assert np.allclose(v[:, :, 1], [[2,0,0],[0,0,0],[0,0,-3]]), "B output1 wrong"
print("B OK")

# Case C: all varying (inds = arange) still works
def model_c(x):
    return x[:, 0] * x[:, 2] + x[:, 1]
xc = np.array([[2.0, 5.0, 3.0]])
rc = shap.ExactExplainer(model_c, bg)(xc, interactions=2)
predc = model_c(xc)
assert np.isclose(predc, (rc.values[0] + rc.base_values[0]).sum()), "C additivity failed"
print("C values:\n", rc.values[0])
print("C OK")
print("ALL EXTRA SANITY PASSED")

import numpy as np
import shap

def model(x):
    return x[:, 0] * x[:, 2]

background = np.zeros((1, 3))
x = np.array([[2.0, 0.0, 3.0]])

result = shap.ExactExplainer(model, background)(x, interactions=2)
pred = model(x)
print("shap version:", shap.__version__)
print("values:\n", result.values[0])
print("base_values:", result.base_values[0])
print("pred:", pred)
print("sum(values)+base:", (result.values[0] + result.base_values[0]).sum())
print("additivity ok:", np.isclose(pred, (result.values[0] + result.base_values[0]).sum()))
assert np.isclose(pred, (result.values[0] + result.base_values[0]).sum())
print("MRE PASSED")

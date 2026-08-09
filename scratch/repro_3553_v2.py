import traceback

import numpy as np

import shap

# Mock data: 10 SHAP values but only 2 feature names
shap_values = np.random.randn(10)
expected_value = 0
features = np.random.randn(10)
feature_names = ["Feature 1", "Feature 2"]  # Too short!

# Also test modern waterfall
# (Requires creating an Explanation object)
print("\nTesting waterfall with short feature_names...")
try:
    exp = shap.Explanation(values=shap_values, base_values=expected_value, data=features, feature_names=feature_names)
    shap.plots.waterfall(exp, show=False)
    print("SUCCESS: Waterfall executed without crash!")
except Exception:
    traceback.print_exc()

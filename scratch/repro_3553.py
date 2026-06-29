import numpy as np

import shap

# Mock data: 10 SHAP values but only 2 feature names
shap_values = np.random.randn(10)
expected_value = 0
features = np.random.randn(10)
feature_names = ["Feature 1", "Feature 2"]  # Too short!

print("Testing waterfall_legacy with short feature_names...")
try:
    shap.plots.waterfall_legacy(expected_value, shap_values, features, feature_names=feature_names, show=False)
    print("SUCCESS: No crash (but how?)")
except IndexError as e:
    print(f"ERROR: Caught expected IndexError: {e}")
except Exception as e:
    print(f"ERROR: Caught unexpected exception: {type(e).__name__}: {e}")

# Also test modern waterfall
# (Requires creating an Explanation object)
print("\nTesting waterfall with short feature_names...")
try:
    exp = shap.Explanation(values=shap_values, base_values=expected_value, data=features, feature_names=feature_names)
    shap.plots.waterfall(exp, show=False)
except IndexError as e:
    print(f"ERROR: Caught expected IndexError in modern waterfall: {e}")
except Exception as e:
    print(f"ERROR: Caught unexpected exception: {type(e).__name__}: {e}")

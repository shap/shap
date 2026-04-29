import xgboost
import shap
import numpy as np

# train XGBoost model
X, y = shap.datasets.adult()
model = xgboost.XGBClassifier().fit(X.values, y)

def custom_masker(mask, x):
    # in this simple example we just zero out the features we are masking
    return (x * mask).reshape(1, len(x))

# compute SHAP values
explainer = shap.Explainer(model.predict_proba, custom_masker)
shap_values = explainer(X[:10])
print("Successfully computed SHAP values!")

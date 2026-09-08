import xgboost

import shap

# train XGBoost model
X, y = shap.datasets.adult()
model = xgboost.XGBClassifier(n_estimators=100, max_depth=2).fit(X, y)

# compute SHAP values
explainer = shap.Explainer(model, X)
shap_values = explainer(X[:10])

# check if decision plot supports Explanation
try:
    shap.plots.decision(shap_values.base_values[0], shap_values.values, show=False)
    print("Raw values worked!")
    shap.plots.decision(shap_values, show=False)
    print("Explanation object worked directly!")
except Exception as e:
    print("Error:", e)

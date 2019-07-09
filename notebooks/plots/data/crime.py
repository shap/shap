# Create 'crime.pickle'

from sklearn.model_selection import train_test_split
import lightgbm as lgb
import pickle
import shap

random_state = 1203344

# Load data and train model
X, y = shap.datasets.communitiesandcrime()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=random_state)
model = lgb.LGBMRegressor(random_state=random_state)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# Calculate and plot SHAP values
explainer = shap.TreeExplainer(model)
idx = 13
shap_values = explainer.shap_values(X_test.iloc[[idx]], y_test[idx])

# Dump to pickle
o = (explainer.expected_value, shap_values, X_test.iloc[0])
with open('./crime.pickle', 'wb') as fl:
    pickle.dump(o, fl)


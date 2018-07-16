import matplotlib
import numpy as np

matplotlib.use('Agg')
import shap


def test_front_page_xgboost():
    try:
        import xgboost
    except Exception as e:
        print("Skipping test_front_page_xgboost!")
        return
    import shap

    # load JS visualization code to notebook
    shap.initjs()

    # train XGBoost model
    X, y = shap.datasets.boston()
    model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)

    # explain the model's predictions using SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # visualize the first prediction's explaination
    shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :])

    # visualize the training set predictions
    shap.force_plot(explainer.expected_value, shap_values, X)

    # create a SHAP dependence plot to show the effect of a single feature across the whole dataset
    shap.dependence_plot(5, shap_values, X, show=False)
    shap.dependence_plot("RM", shap_values, X, show=False)

    # summarize the effects of all the features
    shap.summary_plot(shap_values, X, show=False)

def test_front_page_sklearn():
    import sklearn.ensemble
    import shap

    # load JS visualization code to notebook
    shap.initjs()

    # train model
    X, y = shap.datasets.boston()
    model = sklearn.ensemble.RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    # explain the model's predictions using SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # visualize the first prediction's explaination
    shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :])

    # visualize the training set predictions
    shap.force_plot(explainer.expected_value, shap_values, X)

    # create a SHAP dependence plot to show the effect of a single feature across the whole dataset
    shap.dependence_plot(5, shap_values, X, show=False)
    shap.dependence_plot("RM", shap_values, X, show=False)

    # summarize the effects of all the features
    shap.summary_plot(shap_values, X, show=False)

def test_xgboost_multiclass():
    try:
        import xgboost
    except Exception as e:
        print("Skipping test_xgboost_multiclass!")
        return
    import shap

    # train XGBoost model
    X, Y = shap.datasets.iris()
    model = xgboost.XGBClassifier(objective="binary:logistic", max_depth=4)
    model.fit(X, Y)

    # explain the model's predictions using SHAP values (use pred_contrib in LightGBM)
    shap_values = shap.TreeExplainer(model).shap_values(X)

    # ensure plot works for first class
    shap.dependence_plot(0, shap_values[0], X, show=False)

def test_xgboost_mixed_types():
    try:
        import xgboost
    except Exception as e:
        print("Skipping test_xgboost_mixed_types!")
        return
    import shap
    import numpy as np

    X,y = shap.datasets.boston()
    X["LSTAT"] = X["LSTAT"].astype(np.int64)
    X["B"] = X["B"].astype(np.bool)
    bst = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 1000)
    shap_values = shap.TreeExplainer(bst).shap_values(X)
    shap.dependence_plot(0, shap_values, X, show=False)

def test_sklearn_random_forest_multiclass():
    import shap
    from sklearn.ensemble import RandomForestClassifier

    X, y = shap.datasets.iris()
    y[y == 2] = 1
    model = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    assert np.abs(shap_values[0][0,0] - 0.05) < 1e-3
    assert np.abs(shap_values[1][0,0] + 0.05) < 1e-3

def test_sklearn_decision_tree_multiclass():
    import shap
    from sklearn.tree import DecisionTreeClassifier

    X, y = shap.datasets.iris()
    y[y == 2] = 1
    model = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    assert np.abs(shap_values[0][0,0] - 0.05) < 1e-1
    assert np.abs(shap_values[1][0,0] + 0.05) < 1e-1

def test_lightgbm():
    try:
        import lightgbm
    except Exception as e:
        print("Skipping test_lightgbm!")
        return
    import shap

    # train XGBoost model
    X, y = shap.datasets.boston()
    model = lightgbm.sklearn.LGBMRegressor()
    model.fit(X, y)

    # explain the model's predictions using SHAP values
    shap_values = shap.TreeExplainer(model).shap_values(X)

def test_lightgbm_multiclass():
    try:
        import lightgbm
    except Exception as e:
        print("Skipping test_lightgbm_multiclass!")
        return
    import shap

    # train XGBoost model
    X, Y = shap.datasets.iris()
    model = lightgbm.sklearn.LGBMClassifier()
    model.fit(X, Y)

    # explain the model's predictions using SHAP values
    shap_values = shap.TreeExplainer(model).shap_values(X)

    # ensure plot works for first class
    shap.dependence_plot(0, shap_values[0], X, show=False)

# TODO: Test tree_limit argument

def test_sklearn_interaction():
    import sklearn
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    # train a simple sklean RF model on the iris dataset
    X, y = shap.datasets.iris()
    X_train,X_test,Y_train,Y_test = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)
    rforest = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0)
    model = rforest.fit(X_train, Y_train)

    # verify symmetry of the interaction values (this typically breaks if anything is wrong)
    interaction_vals = shap.TreeExplainer(model).shap_interaction_values(X)
    for i in range(len(interaction_vals)):
        for j in range(len(interaction_vals[i])):
            for k in range(len(interaction_vals[i][j])):
                for l in range(len(interaction_vals[i][j][k])):
                    assert abs(interaction_vals[i][j][k][l] - interaction_vals[i][j][l][k]) < 0.0000001

def test_sum_match_random_forest():
    import shap
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    import sklearn

    X_train,X_test,Y_train,Y_test = train_test_split(*shap.datasets.adult(), test_size=0.2, random_state=0)
    clf = RandomForestClassifier(random_state=202, n_estimators=10, max_depth=10)
    clf.fit(X_train, Y_train)
    predicted = clf.predict_proba(X_test)
    ex = shap.TreeExplainer(clf)
    shap_values = ex.shap_values(X_test)
    assert np.abs(shap_values[0].sum(1) + ex.expected_value[0] - predicted[:,0]).max() < 1e-6, \
        "SHAP values don't sum to model output!"

def test_single_row_random_forest():
    import shap
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    import sklearn

    X_train,X_test,Y_train,Y_test = train_test_split(*shap.datasets.adult(), test_size=0.2, random_state=0)
    clf = RandomForestClassifier(random_state=202, n_estimators=10, max_depth=10)
    clf.fit(X_train, Y_train)
    predicted = clf.predict_proba(X_test)
    ex = shap.TreeExplainer(clf)
    shap_values = ex.shap_values(X_test.iloc[0,:])
    assert np.abs(shap_values[0].sum() + ex.expected_value[0] - predicted[0,0]) < 1e-6, \
        "SHAP values don't sum to model output!"

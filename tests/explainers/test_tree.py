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
    models = [
        sklearn.ensemble.RandomForestRegressor(n_estimators=100),
        sklearn.ensemble.ExtraTreesRegressor(n_estimators=100),
    ]
    for model in models:
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
    except:
        print("Skipping test_lightgbm!")
        return
    import shap

    # train XGBoost model
    X, y = shap.datasets.boston()
    model = lightgbm.sklearn.LGBMRegressor(categorical_feature=[8])
    model.fit(X, y)

    # explain the model's predictions using SHAP values
    shap_values = shap.TreeExplainer(model).shap_values(X)

def test_lightgbm_multiclass():
    try:
        import lightgbm
    except:
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
                    assert abs(interaction_vals[i][j][k][l] - interaction_vals[i][j][l][k]) < 1e-6

    # ensure the interaction plot works
    shap.summary_plot(interaction_vals[0], X, show=False)

def test_lightgbm_interaction():
    try:
        import lightgbm
    except Exception as e:
        print("Skipping test_lightgbm_interaction!")
        return
    import shap

    # train XGBoost model
    X, y = shap.datasets.boston()
    model = lightgbm.sklearn.LGBMRegressor()
    model.fit(X, y)

    # verify symmetry of the interaction values (this typically breaks if anything is wrong)
    interaction_vals = shap.TreeExplainer(model).shap_interaction_values(X)
    for j in range(len(interaction_vals)):
        for k in range(len(interaction_vals[j])):
            for l in range(len(interaction_vals[j][k])):
                assert abs(interaction_vals[j][k][l] - interaction_vals[j][l][k]) < 1e-6

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
    
def test_sum_match_extra_trees():
    import shap
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import ExtraTreesRegressor
    import sklearn

    X_train,X_test,Y_train,Y_test = train_test_split(*shap.datasets.adult(), test_size=0.2, random_state=0)
    clf = ExtraTreesRegressor(random_state=202, n_estimators=10, max_depth=10)
    clf.fit(X_train, Y_train)
    predicted = clf.predict(X_test)
    ex = shap.TreeExplainer(clf)
    shap_values = ex.shap_values(X_test)
    assert np.abs(shap_values.sum(1) + ex.expected_value - predicted).max() < 1e-6, \
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

def test_sum_match_gradient_boosting_classifier():
    import shap
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingClassifier
    import sklearn

    X_train,X_test,Y_train,Y_test = train_test_split(*shap.datasets.adult(), test_size=0.2, random_state=0)
    clf = GradientBoostingClassifier(random_state=202, n_estimators=10, max_depth=10)
    clf.fit(X_train, Y_train)

    # Use decision function to get prediction before it is mapped to a probability
    predicted = clf.decision_function(X_test)
    ex = shap.TreeExplainer(clf)
    shap_values = ex.shap_values(X_test)
    assert np.abs(shap_values.sum(1) + ex.expected_value - predicted).max() < 1e-6, \
        "SHAP values don't sum to model output!"

def test_single_row_gradient_boosting_classifier():
    import shap
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingClassifier
    import sklearn

    X_train,X_test,Y_train,Y_test = train_test_split(*shap.datasets.adult(), test_size=0.2, random_state=0)
    clf = GradientBoostingClassifier(random_state=202, n_estimators=10, max_depth=10)
    clf.fit(X_train, Y_train)
    predicted = clf.decision_function(X_test)
    ex = shap.TreeExplainer(clf)
    shap_values = ex.shap_values(X_test.iloc[0,:])
    assert np.abs(shap_values.sum() + ex.expected_value - predicted[0]) < 1e-6, \
        "SHAP values don't sum to model output!"

def test_sum_match_gradient_boosting_regressor():
    import shap
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingRegressor
    import sklearn

    X_train,X_test,Y_train,Y_test = train_test_split(*shap.datasets.adult(), test_size=0.2, random_state=0)
    clf = GradientBoostingRegressor(random_state=202, n_estimators=10, max_depth=10)
    clf.fit(X_train, Y_train)

    predicted = clf.predict(X_test)
    ex = shap.TreeExplainer(clf)
    shap_values = ex.shap_values(X_test)
    assert np.abs(shap_values.sum(1) + ex.expected_value - predicted).max() < 1e-6, \
        "SHAP values don't sum to model output!"

def test_single_row_gradient_boosting_regressor():
    import shap
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingRegressor
    import sklearn

    X_train,X_test,Y_train,Y_test = train_test_split(*shap.datasets.adult(), test_size=0.2, random_state=0)
    clf = GradientBoostingRegressor(random_state=202, n_estimators=10, max_depth=10)
    clf.fit(X_train, Y_train)
    
    predicted = clf.predict(X_test)
    ex = shap.TreeExplainer(clf)
    shap_values = ex.shap_values(X_test.iloc[0,:])
    assert np.abs(shap_values.sum() + ex.expected_value - predicted[0]) < 1e-6, \
        "SHAP values don't sum to model output!"

def test_single_tree_compare_with_kernel_shap():
    """ Compare with Kernel SHAP, which makes the same independence assumptions
    as Independent Tree SHAP.  Namely, they both assume independence between the 
    set being conditioned on, and the remainder set.
    """
    try:
        import xgboost
    except Exception as e:
        print("Skipping test_single_tree_compare_with_kernel_shap!")
        return
    np.random.seed(10)

    n = 1000
    X = np.random.normal(size=(n,7))
    b = np.array([-2,1,3,5,2,20,-5])
    y = np.matmul(X,b)
    max_depth = 6

    # train a model with single tree
    Xd = xgboost.DMatrix(X, label=y)
    model = xgboost.train({'eta':1, 
                       'max_depth':max_depth, 
                       'base_score': 0, 
                       "lambda": 0}, 
                      Xd, 1)
    ypred = model.predict(Xd)

    # Compare for five random samples
    for i in range(5):
        x_ind = np.random.choice(X.shape[1]); x = X[x_ind:x_ind+1,:]

        expl = shap.TreeExplainer(model, X, feature_dependence="independent")
        f = lambda inp : model.predict(xgboost.DMatrix(inp))
        expl_kern = shap.KernelExplainer(f, X)

        itshap = expl.shap_values(x)
        kshap = expl_kern.shap_values(x, nsamples=150)
        assert np.allclose(itshap,kshap), \
        "Kernel SHAP doesn't match Independent Tree SHAP!"
        assert np.allclose(itshap.sum() + expl.expected_value, ypred[x_ind]), \
        "SHAP values don't sum to model output!"    

def test_several_trees():
    """ Make sure Independent Tree SHAP sums up to the correct value for
    larger models (20 trees).
    """    
    try:
        import xgboost
    except:
        print("Skipping test_several_trees!")
        return
    np.random.seed(10)

    n = 1000
    X = np.random.normal(size=(n,7))
    b = np.array([-2,1,3,5,2,20,-5])
    y = np.matmul(X,b)
    max_depth = 6

    # train a model with single tree
    Xd = xgboost.DMatrix(X, label=y)
    model = xgboost.train({'eta':1, 
                       'max_depth':max_depth, 
                       'base_score': 0, 
                       "lambda": 0}, 
                      Xd, 20)
    ypred = model.predict(Xd)

    # Compare for five random samples
    for i in range(5):
        x_ind = np.random.choice(X.shape[1]); x = X[x_ind:x_ind+1,:]
        expl = shap.TreeExplainer(model, X, feature_dependence="independent")
        itshap = expl.shap_values(x)
        assert np.allclose(itshap.sum() + expl.expected_value, ypred[x_ind]), \
        "SHAP values don't sum to model output!"
        
def test_single_tree_nonlinear_transformations():
    """ Make sure Independent Tree SHAP single trees with non-linear
    transformations.
    """
    # Supported non-linear transforms
    def sigmoid(x):
        return(1/(1+np.exp(-x)))

    def log_loss(yt,yp):
        return(-(yt*np.log(yp) + (1 - yt)*np.log(1 - yp)))

    def mse(yt,yp):
        return(np.square(yt-yp))

    try:
        import xgboost
    except:
        print("Skipping test_several_trees!")
        return

    np.random.seed(10)

    n = 1000
    X = np.random.normal(size=(n,7))
    b = np.array([-2,1,3,5,2,20,-5])
    y = np.matmul(X,b)
    y = y + abs(min(y))
    y = np.random.binomial(n=1,p=y/max(y))
    max_depth = 6

    # train a model with single tree
    Xd = xgboost.DMatrix(X, label=y)
    model = xgboost.train({'eta':1, 
                       'max_depth':max_depth, 
                       'base_score': y.mean(), 
                       "lambda": 0,
                       "objective": "binary:logistic"}, 
                      Xd, 1)
    pred = model.predict(Xd,output_margin=True) # In margin space (log odds)
    trans_pred = model.predict(Xd) # In probability space

    expl = shap.TreeExplainer(model, X, feature_dependence="independent")
    f = lambda inp : model.predict(xgboost.DMatrix(inp), output_margin=True)
    expl_kern = shap.KernelExplainer(f, X)

    x_ind = 0; x = X[x_ind:x_ind+1,:]
    itshap = expl.shap_values(x)
    kshap = expl_kern.shap_values(x, nsamples=300)
    assert np.allclose(itshap.sum() + expl.expected_value, pred[x_ind]), \
    "SHAP values don't sum to model output on explaining margin!"
    assert np.allclose(itshap, kshap), \
    "Independent Tree SHAP doesn't match Kernel SHAP on explaining margin!"

    model.set_attr(objective="binary:logistic")
    expl = shap.TreeExplainer(model, X, feature_dependence="independent", model_output="probability")
    itshap = expl.shap_values(x)
    assert np.allclose(itshap.sum() + expl.expected_value, trans_pred[x_ind]), \
    "SHAP values don't sum to model output on explaining logistic!"

    
    # expl = shap.TreeExplainer(model, X, feature_dependence="independent", model_output="logloss")
    # itshap = expl.shap_values(x,y=y[x_ind])
    # margin_pred = model.predict(xgb.DMatrix(x),output_margin=True)
    # currpred = log_loss(y[x_ind],sigmoid(margin_pred))
    # assert np.allclose(itshap.sum(), currpred - expl.expected_value), \
    # "SHAP values don't sum to model output on explaining logloss!"

def test_xgboost_classifier_independent_margin():
    try:
        import xgboost
    except:
        print("Skipping test_several_trees!")
        return
    
    # train XGBoost model
    np.random.seed(10)
    n = 1000
    X = np.random.normal(size=(n,7))
    b = np.array([-2,1,3,5,2,20,-5])
    y = np.matmul(X,b)
    y = y + abs(min(y))
    y = np.random.binomial(n=1,p=y/max(y))

    model = xgboost.XGBClassifier(n_estimators=10, max_depth=5)
    model.fit(X, y)

    # explain the model's predictions using SHAP values
    e = shap.TreeExplainer(model, X, feature_dependence="independent", model_output="margin")
    shap_values = e.shap_values(X)

    assert np.allclose(shap_values.sum(1) + e.expected_value, model.predict(X, output_margin=True))

def test_xgboost_classifier_independent_probability():
    try:
        import xgboost
    except:
        print("Skipping test_several_trees!")
        return
    
    # train XGBoost model
    np.random.seed(10)
    n = 1000
    X = np.random.normal(size=(n,7))
    b = np.array([-2,1,3,5,2,20,-5])
    y = np.matmul(X,b)
    y = y + abs(min(y))
    y = np.random.binomial(n=1,p=y/max(y))

    model = xgboost.XGBClassifier(n_estimators=10, max_depth=5)
    model.fit(X, y)

    # explain the model's predictions using SHAP values
    e = shap.TreeExplainer(model, X, feature_dependence="independent", model_output="probability")
    shap_values = e.shap_values(X)

    assert np.allclose(shap_values.sum(1) + e.expected_value, model.predict_proba(X)[:,1])

def test_front_page_xgboost_global_path_dependent():
    try:
        import xgboost
    except:
        print("Skipping test_front_page_xgboost!")
        return

    # train XGBoost model
    X, y = shap.datasets.boston()
    model = xgboost.XGBRegressor()
    model.fit(X, y)

    # explain the model's predictions using SHAP values
    explainer = shap.TreeExplainer(model, X, feature_dependence="global_path_dependent")
    shap_values = explainer.shap_values(X)

    assert np.allclose(shap_values.sum(1) + explainer.expected_value, model.predict(X))
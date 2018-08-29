import sklearn
import sklearn.ensemble


# This models are all tuned for the corrgroups60 dataset

def corrgroups60__lasso():
    """ Lasso Regression
    """
    return sklearn.linear_model.Lasso(alpha=0.1)

def corrgroups60__ridge():
    """ Ridge Regression
    """
    return sklearn.linear_model.Ridge(alpha=1.0)

def corrgroups60__decision_tree():
    """ Decision Tree
    """
    return sklearn.tree.DecisionTreeRegressor(random_state=0)

def corrgroups60__random_forest():
    """ Random Forest
    """
    return sklearn.ensemble.RandomForestRegressor(random_state=0)

def corrgroups60__gbm():
    """ Gradient Boosting Machines
    """
    import xgboost
    return xgboost.XGBRegressor(random_state=0)

# def corrgroups60__dnn_regression():
#     import keras
#     return xgboost.XGBRegressor(random_state=0)


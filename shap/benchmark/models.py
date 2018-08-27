import sklearn
import sklearn.ensemble

def lasso_regression():
    return sklearn.linear_model.Lasso(alpha=0.1)

def ridge_regression():
    return sklearn.linear_model.Ridge(alpha=1.0)

def decision_tree_regression():
    return sklearn.tree.DecisionTreeRegressor(random_state=0)
    
def random_forest_regression():
    return sklearn.ensemble.RandomForestRegressor(random_state=0)


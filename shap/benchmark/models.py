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

class KerasParamWrap(object):
    """ A simple wrapper that allows us to set keras `fit` parameters in the constructor.
    """
    def __init__(self, model, epochs, flatten_output=False):
        self.model = model
        self.epochs = epochs
        self.flatten_output = flatten_output
    
    def fit(self, X, y, verbose=0):
        return self.model.fit(X, y, epochs=self.epochs, verbose=verbose)
        
    def predict(self, X):
        if self.flatten_output:
            return self.model.predict(X).flatten()
        else:
            return self.model.predict(X)

def corrgroups60__ffnn():
    """ 4-Layer Neural Network
    """
    from keras.models import Sequential
    from keras.layers import Dense

    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=60))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam',
                loss='mean_squared_error',
                metrics=['mean_squared_error'])

    return KerasParamWrap(model, 100, flatten_output=True)


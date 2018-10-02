import sklearn
import sklearn.ensemble
import gc
from sklearn.preprocessing import StandardScaler

class KerasWrap(object):
    """ A wrapper that allows us to set parameters in the constructor and do a reset before fitting.
    """
    def __init__(self, model, epochs, flatten_output=False):
        self.model = model
        self.epochs = epochs
        self.flatten_output = flatten_output
        self.init_weights = None
        self.scaler = StandardScaler()
        
    def fit(self, X, y, verbose=0):
        if self.init_weights is None:
            self.init_weights = self.model.get_weights()
        else:
            self.model.set_weights(self.init_weights)
        self.scaler.fit(X)
        return self.model.fit(X, y, epochs=self.epochs, verbose=verbose)

    def predict(self, X):
        X = self.scaler.transform(X)
        if self.flatten_output:
            return self.model.predict(X).flatten()
        else:
            return self.model.predict(X)


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
    return xgboost.XGBRegressor(n_jobs=8, random_state=0)

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

    return KerasWrap(model, 30, flatten_output=True)


def cric__lasso():
    """ Lasso Regression
    """
    return sklearn.linear_model.LogisticRegression(penalty="l1", C=0.002)

def cric__ridge():
    """ Ridge Regression
    """
    return sklearn.linear_model.LogisticRegression(penalty="l2")

def cric__decision_tree():
    """ Decision Tree
    """
    return sklearn.tree.DecisionTreeClassifier(random_state=0)

def cric__random_forest():
    """ Random Forest
    """
    return sklearn.ensemble.RandomForestClassifier(100, random_state=0)

def cric__gbm():
    """ Gradient Boosting Machines
    """
    import xgboost
    return xgboost.XGBClassifier(n_estimators=500, learning_rate=0.01, subsample=0.2, n_jobs=8, random_state=0)

def cric__ffnn():
    """ 4-Layer Neural Network
    """
    from keras.models import Sequential
    from keras.layers import Dense, Dropout

    model = Sequential()
    model.add(Dense(10, activation='relu', input_dim=336))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    return KerasWrap(model, 30, flatten_output=True)
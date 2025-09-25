import numpy as np
import sklearn
import sklearn.ensemble
from sklearn.preprocessing import StandardScaler


class KerasWrap:
    """A wrapper that allows us to set parameters in the constructor and do a reset before fitting."""

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
    """Lasso Regression"""
    return sklearn.linear_model.Lasso(alpha=0.1)


def corrgroups60__ridge():
    """Ridge Regression"""
    return sklearn.linear_model.Ridge(alpha=1.0)


def corrgroups60__decision_tree():
    """Decision Tree"""
    # max_depth was chosen to minimise test error
    return sklearn.tree.DecisionTreeRegressor(random_state=0, max_depth=6)


def corrgroups60__random_forest():
    """Random Forest"""
    return sklearn.ensemble.RandomForestRegressor(100, random_state=0)


def corrgroups60__gbm():
    """Gradient Boosted Trees"""
    import xgboost

    # max_depth and learning_rate were fixed then n_estimators was chosen using a train/test split
    return xgboost.XGBRegressor(max_depth=6, n_estimators=50, learning_rate=0.1, n_jobs=8, random_state=0)


def corrgroups60__ffnn():
    """4-Layer Neural Network"""
    import tensorflow as tf

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(32, activation="relu", input_dim=60))
    model.add(tf.keras.layers.Dense(20, activation="relu"))
    model.add(tf.keras.layers.Dense(20, activation="relu"))
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mean_squared_error"])

    return KerasWrap(model, 30, flatten_output=True)


def independentlinear60__lasso():
    """Lasso Regression"""
    return sklearn.linear_model.Lasso(alpha=0.1)


def independentlinear60__ridge():
    """Ridge Regression"""
    return sklearn.linear_model.Ridge(alpha=1.0)


def independentlinear60__decision_tree():
    """Decision Tree"""
    # max_depth was chosen to minimise test error
    return sklearn.tree.DecisionTreeRegressor(random_state=0, max_depth=4)


def independentlinear60__random_forest():
    """Random Forest"""
    return sklearn.ensemble.RandomForestRegressor(100, random_state=0)


def independentlinear60__gbm():
    """Gradient Boosted Trees"""
    import xgboost

    # max_depth and learning_rate were fixed then n_estimators was chosen using a train/test split
    return xgboost.XGBRegressor(max_depth=6, n_estimators=100, learning_rate=0.1, n_jobs=8, random_state=0)


def independentlinear60__ffnn():
    """4-Layer Neural Network"""
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Sequential

    model = Sequential()
    model.add(Dense(32, activation="relu", input_dim=60))
    model.add(Dense(20, activation="relu"))
    model.add(Dense(20, activation="relu"))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mean_squared_error"])

    return KerasWrap(model, 30, flatten_output=True)


def cric__lasso():
    """Lasso Regression"""
    model = sklearn.linear_model.LogisticRegression(penalty="l1", C=0.002)

    # we want to explain the raw probability outputs of the trees
    model.predict = lambda X: model.predict_proba(X)[:, 1]

    return model


def cric__ridge():
    """Ridge Regression"""
    model = sklearn.linear_model.LogisticRegression(penalty="l2")

    # we want to explain the raw probability outputs of the trees
    model.predict = lambda X: model.predict_proba(X)[:, 1]

    return model


def cric__decision_tree():
    """Decision Tree"""
    model = sklearn.tree.DecisionTreeClassifier(random_state=0, max_depth=4)

    # we want to explain the raw probability outputs of the trees
    model.predict = lambda X: model.predict_proba(X)[:, 1]

    return model


def cric__random_forest():
    """Random Forest"""
    model = sklearn.ensemble.RandomForestClassifier(100, random_state=0)

    # we want to explain the raw probability outputs of the trees
    model.predict = lambda X: model.predict_proba(X)[:, 1]

    return model


def cric__gbm():
    """Gradient Boosted Trees"""
    import xgboost

    # max_depth and subsample match the params used for the full cric data in the paper
    # learning_rate was set a bit higher to allow for faster runtimes
    # n_estimators was chosen based on a train/test split of the data
    model = xgboost.XGBClassifier(
        max_depth=5, n_estimators=400, learning_rate=0.01, subsample=0.2, n_jobs=8, random_state=0
    )

    # we want to explain the margin, not the transformed probability outputs
    model.__orig_predict = model.predict
    model.predict = lambda X: model.__orig_predict(X, output_margin=True)

    return model


def cric__ffnn():
    """4-Layer Neural Network"""
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.models import Sequential

    model = Sequential()
    model.add(Dense(10, activation="relu", input_dim=336))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return KerasWrap(model, 30, flatten_output=True)


def human__decision_tree():
    """Decision Tree"""
    # build data
    N = 1000000
    M = 3
    X = np.zeros((N, M))
    X.shape
    y = np.zeros(N)
    X[0, 0] = 1
    y[0] = 8
    X[1, 1] = 1
    y[1] = 8
    X[2, 0:2] = 1
    y[2] = 4

    # fit model
    xor_model = sklearn.tree.DecisionTreeRegressor(max_depth=2)
    xor_model.fit(X, y)

    return xor_model

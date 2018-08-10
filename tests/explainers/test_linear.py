import matplotlib
import numpy as np
matplotlib.use('Agg')
import shap



def test_tied_pair():
    np.random.seed(0)
    beta = np.array([1, 0, 0])
    mu = np.zeros(3)
    Sigma = np.array([[1, 0.999999, 0], [0.999999, 1, 0], [0, 0, 1]])
    X = np.ones((1,3))
    explainer = shap.LinearExplainer((beta, 0), (mu, Sigma))
    assert np.abs(explainer.shap_values(X) - np.array([0.5, 0.5, 0])).max() < 0.05

def test_tied_triple():
    np.random.seed(0)
    beta = np.array([0, 1, 0, 0])
    mu = 1*np.ones(4)
    Sigma = np.array([[1, 0.999999, 0.999999, 0], [0.999999, 1, 0.999999, 0], [0.999999, 0.999999, 1, 0], [0, 0, 0, 1]])
    X = 2*np.ones((1,4))
    explainer = shap.LinearExplainer((beta, 0), (mu, Sigma))
    assert explainer.expected_value == 1
    assert np.abs(explainer.shap_values(X) - np.array([0.33333, 0.33333, 0.33333, 0])).max() < 0.05

def test_sklearn_linear():
    np.random.seed(0)
    from sklearn.linear_model import Ridge
    import shap

    # train linear model
    X,y = shap.datasets.boston()
    model = Ridge(0.1)
    model.fit(X, y)

    # explain the model's predictions using SHAP values
    explainer = shap.LinearExplainer(model, X)
    assert np.abs(explainer.expected_value - model.predict(X).mean()) < 1e-6
    shap_values = explainer.shap_values(X)

def test_perfect_colinear():
    import shap
    from sklearn.linear_model import LinearRegression
    import numpy as np

    X,y = shap.datasets.boston()
    X.iloc[:,0] = X.iloc[:,4] # test duplicated features
    X.iloc[:,5] = X.iloc[:,6] - X.iloc[:,6] # test multiple colinear features
    X.iloc[:,3] = 0 # test null features
    model = LinearRegression()
    model.fit(X, y)
    explainer = shap.LinearExplainer(model, X)
    shap_values = explainer.shap_values(X)
    assert np.abs(shap_values.sum(1) - model.predict(X) + model.predict(X).mean()).sum() < 1e-7

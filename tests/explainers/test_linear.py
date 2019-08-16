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
    explainer = shap.LinearExplainer((beta, 0), (mu, Sigma), feature_dependence="correlation")
    assert np.abs(explainer.shap_values(X) - np.array([0.5, 0.5, 0])).max() < 0.05

def test_tied_triple():
    np.random.seed(0)
    beta = np.array([0, 1, 0, 0])
    mu = 1*np.ones(4)
    Sigma = np.array([[1, 0.999999, 0.999999, 0], [0.999999, 1, 0.999999, 0], [0.999999, 0.999999, 1, 0], [0, 0, 0, 1]])
    X = 2*np.ones((1,4))
    explainer = shap.LinearExplainer((beta, 0), (mu, Sigma), feature_dependence="correlation")
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
    explainer.shap_values(X)

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
    explainer = shap.LinearExplainer(model, X, feature_dependence="correlation")
    shap_values = explainer.shap_values(X)
    assert np.abs(shap_values.sum(1) - model.predict(X) + model.predict(X).mean()).sum() < 1e-7

def test_shape_values_linear_many_features():
    from sklearn.linear_model import Ridge

    np.random.seed(0)

    coef = np.array([1, 2]).T

    # generate linear data
    X = np.random.normal(1, 10, size=(1000, len(coef)))
    y = np.dot(X, coef) + 1 + np.random.normal(scale=0.1, size=1000)

    # train linear model
    model = Ridge(0.1)
    model.fit(X, y)

    # explain the model's predictions using SHAP values
    explainer = shap.LinearExplainer(model, X)

    values = explainer.shap_values(X)

    assert values.shape == (1000, 2)

    expected = (X - X.mean(0)) * coef
    np.testing.assert_allclose(expected - values, 0, atol=0.01)

def test_single_feature():
    """ Make sure things work with a univariate linear regression.
    """
    import sklearn.linear_model

    np.random.seed(0)

    # generate linear data
    X = np.random.normal(1, 10, size=(1000, 1))
    y = 2 * X[:, 0] + 1 + np.random.normal(scale=0.1, size=1000)

    # train linear model
    model = sklearn.linear_model.Ridge(0.1)
    model.fit(X, y)

    # explain the model's predictions using SHAP values
    explainer = shap.LinearExplainer(model, X)
    shap_values = explainer.shap_values(X)
    assert np.abs(explainer.expected_value - model.predict(X).mean()) < 1e-6
    assert np.max(np.abs(explainer.expected_value + shap_values.sum(1) - model.predict(X))) < 1e-6

def test_sparse():
    """ Validate running LinearExplainer on scipy sparse data
    """
    import sklearn.linear_model
    from sklearn.datasets import make_multilabel_classification
    from scipy.special import expit

    np.random.seed(0)
    n_features = 20
    X, y = make_multilabel_classification(n_samples=100,
                                          sparse=True,
                                          n_features=n_features,
                                          n_classes=1,
                                          n_labels=2)

    # train linear model
    model = sklearn.linear_model.LogisticRegression()
    model.fit(X, y)

    # explain the model's predictions using SHAP values
    explainer = shap.LinearExplainer(model, X)
    shap_values = explainer.shap_values(X)
    assert np.max(np.abs(expit(explainer.expected_value + shap_values[0].sum(1)) - model.predict_proba(X)[:, 1])) < 1e-6

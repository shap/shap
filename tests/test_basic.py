import matplotlib
matplotlib.use('Agg')
import shap
import numpy as np

def test_null_model_small():
    explainer = shap.KernelExplainer(lambda x: np.zeros(x.shape[0]), np.ones((2,4)), nsamples=100)
    e = explainer.explain(np.ones((1,4)))
    assert np.sum(np.abs(e.effects)) < 1e-8

def test_null_model():
    explainer = shap.KernelExplainer(lambda x: np.zeros(x.shape[0]), np.ones((2,10)), nsamples=100)
    e = explainer.explain(np.ones((1,10)))

def test_front_page_model_agnostic():
    from shap import KernelExplainer, DenseData, visualize, initjs
    from sklearn import datasets,neighbors
    from numpy import random, arange

    # print the JS visualization code to the notebook
    initjs()

    # train a k-nearest neighbors classifier on a random subset
    iris = datasets.load_iris()
    random.seed(2)
    inds = arange(len(iris.target))
    random.shuffle(inds)
    knn = neighbors.KNeighborsClassifier()
    knn.fit(iris.data, iris.target == 0)

    # use Shap to explain a single prediction
    background = DenseData(iris.data[inds[:100],:], iris.feature_names) # name the features
    explainer = KernelExplainer(knn.predict, background, nsamples=100)
    x = iris.data[inds[102:103],:]
    visualize(explainer.explain(x))

def test_front_page_xgboost():
    import xgboost
    import shap

    # load JS visualization code to notebook
    shap.initjs()

    # train XGBoost model
    X,y,X_display = shap.datasets.boston()
    bst = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)

    # explain the model's predictions using SHAP values (use pred_contrib in LightGBM)
    shap_values = bst.predict(xgboost.DMatrix(X), pred_contribs=True)

    # visualize the first prediction's explaination
    shap.visualize(shap_values[0,:], X.iloc[0,:])

    # visualize the training set predictions
    shap.visualize(shap_values, X)

    # create a SHAP dependence plot to show the effect of a single feature across the whole dataset
    shap.dependence_plot(5, shap_values, X, show=False)
    shap.dependence_plot("RM", shap_values, X, show=False)

    # summarize the effects of all the features
    shap.summary_plot(shap_values, X, show=False)

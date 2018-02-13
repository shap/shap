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
    import sklearn
    import shap
    from sklearn.model_selection import train_test_split

    # print the JS visualization code to the notebook
    shap.initjs()

    # train a SVM classifier
    X_train,X_test,Y_train,Y_test = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)
    svm = sklearn.svm.SVC(kernel='rbf', probability=True)
    svm.fit(X, Y)

    # use Kernel SHAP to explain test set predictions
    explainer = shap.KernelExplainer(svm.predict_proba, X_train, nsamples=100, link="logit")
    shap_values = explainer.shap_values(X_test)

    # plot the SHAP values for the Setosa output of the first instance
    shap.force_plot(shap_values[0][0,:], X_test.iloc[0,:], link="logit")

def test_front_page_xgboost():
    import xgboost
    import shap

    # load JS visualization code to notebook
    shap.initjs()

    # train XGBoost model
    X,y = shap.datasets.boston()
    bst = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)

    # explain the model's predictions using SHAP values (use pred_contrib in LightGBM)
    shap_values = bst.predict(xgboost.DMatrix(X), pred_contribs=True)

    # visualize the first prediction's explaination
    shap.force_plot(shap_values[0,:], X.iloc[0,:])

    # visualize the training set predictions
    shap.force_plot(shap_values, X)

    # create a SHAP dependence plot to show the effect of a single feature across the whole dataset
    shap.dependence_plot(5, shap_values, X, show=False)
    shap.dependence_plot("RM", shap_values, X, show=False)

    # summarize the effects of all the features
    shap.summary_plot(shap_values, X, show=False)

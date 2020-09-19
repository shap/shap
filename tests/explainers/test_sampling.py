import matplotlib
matplotlib.use('Agg')


def test_null_model_small():
    import numpy as np
    import shap

    explainer = shap.SamplingExplainer(lambda x: np.zeros(x.shape[0]), np.ones((2, 4)), nsamples=100)
    shap_values = explainer.shap_values(np.ones((1, 4)))
    assert np.sum(np.abs(shap_values)) < 1e-8


def test_null_model():
    import numpy as np
    import shap

    explainer = shap.SamplingExplainer(lambda x: np.zeros(x.shape[0]), np.ones((2, 10)), nsamples=100)
    shap_values = explainer.shap_values(np.ones((1, 10)))
    assert np.sum(np.abs(shap_values)) < 1e-8


def test_front_page_model_agnostic():
    import sklearn
    from sklearn.model_selection import train_test_split
    import numpy as np
    import shap

    # print the JS visualization code to the notebook
    shap.initjs()

    # train a SVM classifier
    X_train, X_test, Y_train, Y_test = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)
    svm = sklearn.svm.SVC(kernel='rbf', probability=True)
    svm.fit(X_train, Y_train)

    # use Kernel SHAP to explain test set predictions
    explainer = shap.SamplingExplainer(svm.predict_proba, X_train, nsamples=100)
    shap_values = explainer.shap_values(X_test)

    # plot the SHAP values for the Setosa output of the first instance
    shap.force_plot(explainer.expected_value[0], shap_values[0][0, :], X_test.iloc[0, :])

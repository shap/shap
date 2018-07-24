import matplotlib
import numpy as np

matplotlib.use('Agg')
import shap


def test_null_model_small():
    explainer = shap.KernelExplainer(lambda x: np.zeros(x.shape[0]), np.ones((2, 4)), nsamples=100)
    e = explainer.explain(np.ones((1, 4)))
    assert np.sum(np.abs(e.effects)) < 1e-8


def test_null_model():
    explainer = shap.KernelExplainer(lambda x: np.zeros(x.shape[0]), np.ones((2, 10)), nsamples=100)
    e = explainer.explain(np.ones((1, 10)))
    assert np.sum(np.abs(e.effects)) < 1e-8


def test_front_page_model_agnostic():
    import sklearn
    import shap
    from sklearn.model_selection import train_test_split

    # print the JS visualization code to the notebook
    shap.initjs()

    # train a SVM classifier
    X_train, X_test, Y_train, Y_test = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)
    svm = sklearn.svm.SVC(kernel='rbf', probability=True)
    svm.fit(X_train, Y_train)

    # use Kernel SHAP to explain test set predictions
    explainer = shap.KernelExplainer(svm.predict_proba, X_train, nsamples=100, link="logit")
    shap_values = explainer.shap_values(X_test)

    # plot the SHAP values for the Setosa output of the first instance
    shap.force_plot(shap_values[0][0, :], X_test.iloc[0, :], link="logit")

def test_kernel_shap_with_dataframe():
    from sklearn.linear_model import LinearRegression
    import shap
    import pandas as pd
    import numpy as np
    np.random.seed(3)

    df_X = pd.DataFrame(np.random.random((10, 3)), columns=list('abc'))
    df_X.index = pd.date_range('2018-01-01', periods=10, freq='D', tz='UTC')

    df_y = df_X.eval('a - 2 * b + 3 * c')
    df_y = df_y + np.random.normal(0.0, 0.1, df_y.shape)

    linear_model = LinearRegression()
    linear_model.fit(df_X, df_y)

    explainer = shap.KernelExplainer(linear_model.predict, df_X, keep_index=True)
    shap_values = explainer.shap_values(df_X)


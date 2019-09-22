import numpy as np
import scipy as sp
import shap

import tests.explainers.common


def test_null_model_small():
    explainer = shap.KernelExplainer(lambda x: np.zeros(x.shape[0]), np.ones((2, 4)), nsamples=100)
    e = explainer.explain(np.ones((1, 4)))
    assert np.sum(np.abs(e)) < 1e-8

def test_null_model():
    explainer = shap.KernelExplainer(lambda x: np.zeros(x.shape[0]), np.ones((2, 10)), nsamples=100)
    e = explainer.explain(np.ones((1, 10)))
    assert np.sum(np.abs(e)) < 1e-8

def test_front_page_model_agnostic():
    import sklearn
    import shap
    from sklearn.model_selection import train_test_split

    # print the JS visualization code to the notebook
    shap.initjs()

    # train a SVM classifier
    X_train, X_test, Y_train, Y_test = train_test_split(*shap.datasets.iris(), test_size=0.1, random_state=0)
    svm = sklearn.svm.SVC(kernel='rbf', probability=True)
    svm.fit(X_train, Y_train)

    # use Kernel SHAP to explain test set predictions
    explainer = shap.KernelExplainer(svm.predict_proba, X_train, nsamples=100, link="logit")
    shap_values = explainer.shap_values(X_test)

    # plot the SHAP values for the Setosa output of the first instance
    shap.force_plot(explainer.expected_value[0], shap_values[0][0, :], X_test.iloc[0, :], link="logit")

def test_front_page_model_agnostic_rank():
    import sklearn
    import shap
    from sklearn.model_selection import train_test_split

    # print the JS visualization code to the notebook
    shap.initjs()

    # train a SVM classifier
    X_train, X_test, Y_train, Y_test = train_test_split(*shap.datasets.iris(), test_size=0.1, random_state=0)
    svm = sklearn.svm.SVC(kernel='rbf', probability=True)
    svm.fit(X_train, Y_train)

    # use Kernel SHAP to explain test set predictions
    explainer = shap.KernelExplainer(svm.predict_proba, X_train, nsamples=100, link="logit", l1_reg="rank(3)")
    shap_values = explainer.shap_values(X_test)

    # plot the SHAP values for the Setosa output of the first instance
    shap.force_plot(explainer.expected_value[0], shap_values[0][0, :], X_test.iloc[0, :], link="logit")

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

def test_kernel_shap_with_a1a_sparse_zero_background():
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    import shap

    X, y = shap.datasets.a1a() # pylint: disable=unbalanced-tuple-unpacking
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=0)
    linear_model = LinearRegression()
    linear_model.fit(x_train, y_train)

    _, cols = x_train.shape
    shape = 1, cols
    background = sp.sparse.csr_matrix(shape, dtype=x_train.dtype)
    explainer = shap.KernelExplainer(linear_model.predict, background)
    explainer.shap_values(x_test)

def test_kernel_shap_with_a1a_sparse_nonzero_background():
    np.set_printoptions(threshold=100000)
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.utils.sparsefuncs import csc_median_axis_0
    import shap
    np.random.seed(0)

    X, y = shap.datasets.a1a() # pylint: disable=unbalanced-tuple-unpacking
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=0)
    linear_model = LinearRegression()
    linear_model.fit(x_train, y_train)
    # Calculate median of background data
    median_dense = csc_median_axis_0(x_train.tocsc())
    median = sp.sparse.csr_matrix(median_dense)
    explainer = shap.KernelExplainer(linear_model.predict, median)
    shap_values = explainer.shap_values(x_test)

    def dense_to_sparse_predict(data):
        sparse_data = sp.sparse.csr_matrix(data)
        return linear_model.predict(sparse_data)

    explainer_dense = shap.KernelExplainer(dense_to_sparse_predict, median_dense.reshape((1, len(median_dense))))
    x_test_dense = x_test.toarray()
    shap_values_dense = explainer_dense.shap_values(x_test_dense)
    # Validate sparse and dense result is the same
    assert(np.allclose(shap_values, shap_values_dense, rtol=1e-02, atol=1e-01))

def test_kernel_shap_with_high_dim_sparse():
    # verifies we can run on very sparse data produced from feature hashing
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    import shap
    remove = ('headers', 'footers', 'quotes')
    categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
    ]
    from sklearn.datasets import fetch_20newsgroups
    ngroups = fetch_20newsgroups(subset='train', categories=categories,
                        shuffle=True, random_state=42,
                        remove=remove)
    x_train, x_test, y_train, y_validation = train_test_split(ngroups.data, ngroups.target,
                                                                    test_size=0.01, random_state=42)
    from sklearn.feature_extraction.text import HashingVectorizer
    vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False,
                                n_features=2**16)
    x_train = vectorizer.transform(x_train)
    x_test = vectorizer.transform(x_test)
    # Fit a linear regression model
    linear_model = LinearRegression()
    linear_model.fit(x_train, y_train)
    rows, cols = x_train.shape
    shape = 1, cols
    background = sp.sparse.csr_matrix(shape, dtype=x_train.dtype)
    explainer = shap.KernelExplainer(linear_model.predict, background)
    shap_values = explainer.shap_values(x_test)

def test_kernel_sparse_vs_dense_multirow_background():
    import sklearn
    import shap
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    # train a logistic regression classifier
    X_train, X_test, Y_train, _ = train_test_split(*shap.datasets.iris(), test_size=0.1, random_state=0)
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(X_train, Y_train)

    # use Kernel SHAP to explain test set predictions with dense data
    explainer = shap.KernelExplainer(lr.predict_proba, X_train, nsamples=100, link="logit", l1_reg="rank(3)")
    shap_values = explainer.shap_values(X_test)

    X_sparse_train = sp.sparse.csr_matrix(X_train)
    X_sparse_test = sp.sparse.csr_matrix(X_test)

    lr_sparse = LogisticRegression(solver='lbfgs')
    lr_sparse.fit(X_sparse_train, Y_train)

    # use Kernel SHAP again but with sparse data
    sparse_explainer = shap.KernelExplainer(lr.predict_proba, X_sparse_train, nsamples=100, link="logit", l1_reg="rank(3)")
    sparse_shap_values = sparse_explainer.shap_values(X_sparse_test)

    assert(np.allclose(shap_values, sparse_shap_values, rtol=1e-05, atol=1e-05))

    # Use sparse evaluation examples with dense background
    sparse_sv_dense_bg = explainer.shap_values(X_sparse_test)
    assert(np.allclose(shap_values, sparse_sv_dense_bg, rtol=1e-05, atol=1e-05))


def test_linear():
    """tests that KernelExplainer returns the correct result when the model is linear
    (as per corollary 1 of https://arxiv.org/abs/1705.07874)"""

    np.random.seed(2)
    x = np.random.normal(size=(200, 3), scale=1)

    # a linear model
    def f(x):
        return x[:, 0] + 2.0*x[:, 1]

    phi = shap.KernelExplainer(f, x).shap_values(x, l1_reg="num_features(2)", silent=True)
    assert phi.shape == x.shape

    # corollary 1
    expected = (x - x.mean(0)) * np.array([1.0, 2.0, 0.0])

    np.testing.assert_allclose(expected, phi, rtol=1e-3)


def test_kernel_shap_accuracy():
    np.random.seed(2)
    x = np.random.normal(size=(200, 3), scale=1)

    def f(x):
        return x[:, 0] * 2.0 * np.exp(-np.abs(x[:, 1]))

    phi = shap.KernelExplainer(f, x).shap_values(x, l1_reg='num_features(10)', silent=True)
    assert phi.shape == x.shape

    tests.explainers.common.assert_accuracy(phi, f, x, rtol=0.05, atol=0.05)


def test_kernel_shap_monotonicity():
    np.random.seed(3)
    x = np.random.normal(size=(200, 3), scale=1)

    def f1(x):
        return x[:, 0] * 2.0 * np.exp(-np.abs(x[:, 1]))

    def f2(x):
        return x[:, 0] * 2.0 + x[:, 1]

    phi1 = shap.KernelExplainer(f1, x).shap_values(x, l1_reg='num_features(10)', silent=True)
    phi2 = shap.KernelExplainer(f2, x).shap_values(x, l1_reg='num_features(10)', silent=True)

    tests.explainers.common.assert_monotonicity(x, f1, phi1, f2, phi2, 0)
    tests.explainers.common.assert_monotonicity(x, f1, phi1, f2, phi2, 1)

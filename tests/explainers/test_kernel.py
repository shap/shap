import numpy as np
import pandas as pd
import pytest
import scipy.sparse
import sklearn

import shap


def sigm(x):
    return np.exp(x) / (1 + np.exp(x))


def test_null_model_small():
    """Test a small null model."""
    explainer = shap.KernelExplainer(lambda x: np.zeros(x.shape[0]), np.ones((2, 4)), nsamples=100)
    e = explainer.explain(np.ones((1, 4)))
    assert np.sum(np.abs(e)) < 1e-8


def test_null_model():
    """Test a larger null model."""
    explainer = shap.KernelExplainer(lambda x: np.zeros(x.shape[0]), np.ones((2, 10)), nsamples=100)
    e = explainer.explain(np.ones((1, 10)))
    assert np.sum(np.abs(e)) < 1e-8


def test_front_page_model_agnostic():
    """Test the ReadMe kernel expainer example."""
    # print the JS visualization code to the notebook
    shap.initjs()

    # train a SVM classifier
    X_train, X_test, Y_train, _ = sklearn.model_selection.train_test_split(
        *shap.datasets.iris(), test_size=0.1, random_state=0
    )
    svm = sklearn.svm.SVC(kernel="rbf", probability=True)
    svm.fit(X_train, Y_train)

    # use Kernel SHAP to explain test set predictions
    explainer = shap.KernelExplainer(svm.predict_proba, X_train, nsamples=100, link="logit")
    shap_values = explainer.shap_values(X_test)

    # plot the SHAP values for the Setosa output of the first instance
    # this is a multi output model so we index to get the zero-th output (Setosa)
    shap.force_plot(explainer.expected_value[0], shap_values[0, :, 0], X_test.iloc[0, :], link="logit")


def test_front_page_model_agnostic_rank():
    """Test the rank regularized explanation of the ReadMe example."""
    # print the JS visualization code to the notebook
    shap.initjs()

    # train a SVM classifier
    X_train, X_test, Y_train, _ = sklearn.model_selection.train_test_split(
        *shap.datasets.iris(), test_size=0.1, random_state=0
    )
    svm = sklearn.svm.SVC(kernel="rbf", probability=True)
    svm.fit(X_train, Y_train)

    # use Kernel SHAP to explain test set predictions
    explainer = shap.KernelExplainer(svm.predict_proba, X_train, nsamples=100, link="logit", l1_reg="rank(3)")
    shap_values = explainer.shap_values(X_test)

    # plot the SHAP values for the Setosa output of the first instance
    shap.force_plot(explainer.expected_value[0], shap_values[0, :, 0], X_test.iloc[0, :], link="logit")


def test_kernel_shap_with_call_method():
    """Test the __call__ method of the Kernel class"""
    # print the JS visualization code to the notebook
    shap.initjs()

    # train a SVM classifier
    X_train, X_test, Y_train, _ = sklearn.model_selection.train_test_split(
        *shap.datasets.iris(), test_size=0.1, random_state=0
    )
    svm = sklearn.svm.SVC(kernel="rbf", probability=True)
    svm.fit(X_train, Y_train)

    # use Kernel SHAP to explain test set predictions
    explainer = shap.KernelExplainer(svm.predict_proba, X_train, nsamples=100, link="logit")
    shap_values = explainer(X_test)

    # plot the SHAP values for the Versicolour output of the first instance
    shap.force_plot(shap_values[0, :, 1])

    outputs = svm.predict_proba(X_test)
    # Call sigm since we use logit link
    np.testing.assert_allclose(sigm(shap_values.values.sum(1) + explainer.expected_value), outputs)

    shap_values = explainer.shap_values(X_test)
    np.testing.assert_allclose(sigm(shap_values.sum(1) + explainer.expected_value), outputs)


def test_kernel_shap_with_dataframe(random_seed):
    """Test with a Pandas DataFrame."""
    rs = np.random.RandomState(random_seed)

    df_X = pd.DataFrame(rs.random((10, 3)), columns=list("abc"))
    df_X.index = pd.date_range("2018-01-01", periods=10, freq="D", tz="UTC")

    df_y = df_X.eval("a - 2 * b + 3 * c")
    df_y = df_y + rs.normal(0.0, 0.1, df_y.shape)

    linear_model = sklearn.linear_model.LinearRegression()
    linear_model.fit(df_X, df_y)

    explainer = shap.KernelExplainer(linear_model.predict, df_X, keep_index=True)
    _ = explainer.shap_values(df_X)


def test_kernel_shap_with_dataframe_explanation(random_seed):
    """Test with a Pandas DataFrame with Explanation API.

    The Explanation.data is supposed to be a numpy array in many parts of the code,
    e.g., scatter plot will fail if it is not converted from pandas df to ndarray.

    cf. GH #1625
    """
    rs = np.random.RandomState(random_seed)

    df_X = pd.DataFrame(rs.random((10, 3)), columns=list("abc"))
    df_y = df_X.eval("a - 2 * b + 3 * c")
    df_y = df_y + rs.normal(0.0, 0.1, df_y.shape)

    linear_model = sklearn.linear_model.LinearRegression()
    linear_model.fit(df_X, df_y)

    explainer = shap.KernelExplainer(linear_model.predict, df_X, keep_index=True)
    explanation = explainer(df_X)

    # this shouldn't throw an error
    shap.plots.scatter(explanation[:, "a"], show=False)


def test_kernel_shap_with_a1a_sparse_zero_background():
    """Test with a sparse matrix for the background."""
    X, y = shap.datasets.a1a()
    x_train, x_test, y_train, _ = sklearn.model_selection.train_test_split(X, y, test_size=0.01, random_state=0)
    linear_model = sklearn.linear_model.LinearRegression()
    linear_model.fit(x_train, y_train)

    _, cols = x_train.shape
    shape = 1, cols
    background = scipy.sparse.csr_matrix(shape, dtype=x_train.dtype)
    explainer = shap.KernelExplainer(linear_model.predict, background)
    explainer.shap_values(x_test)


def test_kernel_shap_with_a1a_sparse_nonzero_background():
    """Check with a sparse non zero background matrix."""
    np.set_printoptions(threshold=100000)

    X, y = shap.datasets.a1a()
    x_train, x_test, y_train, _ = sklearn.model_selection.train_test_split(X, y, test_size=0.01, random_state=0)
    linear_model = sklearn.linear_model.LinearRegression()
    linear_model.fit(x_train, y_train)
    # Calculate median of background data
    median_dense = sklearn.utils.sparsefuncs.csc_median_axis_0(x_train.tocsc())
    median = scipy.sparse.csr_matrix(median_dense)
    explainer = shap.KernelExplainer(linear_model.predict, median)
    shap_values = explainer.shap_values(x_test)

    def dense_to_sparse_predict(data):
        sparse_data = scipy.sparse.csr_matrix(data)
        return linear_model.predict(sparse_data)

    explainer_dense = shap.KernelExplainer(dense_to_sparse_predict, median_dense.reshape((1, len(median_dense))))
    x_test_dense = x_test.toarray()
    shap_values_dense = explainer_dense.shap_values(x_test_dense)
    # Validate sparse and dense result is the same
    assert np.allclose(shap_values, shap_values_dense, rtol=1e-02, atol=1e-01)


def test_kernel_shap_with_high_dim_sparse():
    """Verifies we can run on very sparse data produced from feature hashing."""
    remove = ("headers", "footers", "quotes")
    categories = [
        "alt.atheism",
        "talk.religion.misc",
        "comp.graphics",
        "sci.space",
    ]
    ngroups = sklearn.datasets.fetch_20newsgroups(
        subset="train", categories=categories, shuffle=True, random_state=42, remove=remove
    )
    x_train, x_test, y_train, _ = sklearn.model_selection.train_test_split(
        ngroups.data, ngroups.target, test_size=0.01, random_state=42
    )
    vectorizer = sklearn.feature_extraction.text.HashingVectorizer(
        stop_words="english", alternate_sign=False, n_features=2**16
    )
    x_train = vectorizer.transform(x_train)
    x_test = vectorizer.transform(x_test)
    # Fit a linear regression model
    linear_model = sklearn.linear_model.LinearRegression()
    linear_model.fit(x_train, y_train)
    _, cols = x_train.shape
    shape = 1, cols
    background = scipy.sparse.csr_matrix(shape, dtype=x_train.dtype)
    explainer = shap.KernelExplainer(linear_model.predict, background)
    _ = explainer.shap_values(x_test)


def test_kernel_sparse_vs_dense_multirow_background():
    """Mix sparse and dense matrix values."""
    # train a logistic regression classifier
    X_train, X_test, Y_train, _ = sklearn.model_selection.train_test_split(
        *shap.datasets.iris(), test_size=0.1, random_state=0
    )
    lr = sklearn.linear_model.LogisticRegression(solver="lbfgs")
    lr.fit(X_train, Y_train)

    # use Kernel SHAP to explain test set predictions with dense data
    explainer = shap.KernelExplainer(lr.predict_proba, X_train, nsamples=100, link="logit", l1_reg="rank(3)")
    shap_values = explainer.shap_values(X_test)

    X_sparse_train = scipy.sparse.csr_matrix(X_train)
    X_sparse_test = scipy.sparse.csr_matrix(X_test)

    lr_sparse = sklearn.linear_model.LogisticRegression(solver="lbfgs")
    lr_sparse.fit(X_sparse_train, Y_train)

    # use Kernel SHAP again but with sparse data
    sparse_explainer = shap.KernelExplainer(
        lr.predict_proba, X_sparse_train, nsamples=100, link="logit", l1_reg="rank(3)"
    )
    sparse_shap_values = sparse_explainer.shap_values(X_sparse_test)

    assert np.allclose(shap_values, sparse_shap_values, rtol=1e-05, atol=1e-05)

    # Use sparse evaluation examples with dense background
    sparse_sv_dense_bg = explainer.shap_values(X_sparse_test)
    assert np.allclose(shap_values, sparse_sv_dense_bg, rtol=1e-05, atol=1e-05)


def test_linear(random_seed):
    """Tests that KernelExplainer returns the correct result when the model is linear.

    (as per corollary 1 of https://arxiv.org/abs/1705.07874)
    """
    rs = np.random.RandomState(random_seed)
    x = rs.normal(size=(200, 3), scale=1)

    # a linear model
    def f(x):
        return x[:, 0] + 2.0 * x[:, 1]

    explainer = shap.KernelExplainer(f, x)
    explanation = explainer(x, l1_reg="num_features(2)", silent=True)
    phi = explanation.values
    assert phi.shape == x.shape

    # corollary 1
    expected = (x - x.mean(0)) * np.array([1.0, 2.0, 0.0])

    np.testing.assert_allclose(expected, phi, rtol=1e-3)


def test_non_numeric():
    """Test using non-numeric data."""
    # create dummy data
    X = np.array([["A", "0", "0"], ["A", "1", "0"], ["B", "0", "0"], ["B", "1", "0"], ["A", "1", "0"]])
    y = np.array([0, 1, 2, 3, 4])

    # build and train the pipeline
    pipeline = sklearn.pipeline.Pipeline(
        [("oneHotEncoder", sklearn.preprocessing.OneHotEncoder()), ("linear", sklearn.linear_model.LinearRegression())]
    )
    pipeline.fit(X, y)

    # use KernelExplainer
    explainer = shap.KernelExplainer(pipeline.predict, X, nsamples=100)
    shap_values = explainer.explain(X[0, :].reshape(1, -1))

    assert np.abs(explainer.expected_value + shap_values.sum(0) - pipeline.predict(X[0, :].reshape(1, -1))[0]) < 1e-4
    assert shap_values[2] == 0

    # tests for shap.KernelExplainer.not_equal
    assert shap.KernelExplainer.not_equal(0, 0) == shap.KernelExplainer.not_equal("0", "0")
    assert shap.KernelExplainer.not_equal(0, 1) == shap.KernelExplainer.not_equal("0", "1")
    assert shap.KernelExplainer.not_equal(0, np.nan) == shap.KernelExplainer.not_equal("0", np.nan)
    assert shap.KernelExplainer.not_equal(0, np.nan) == shap.KernelExplainer.not_equal("0", None)
    assert shap.KernelExplainer.not_equal(np.nan, 0) == shap.KernelExplainer.not_equal(np.nan, "0")
    assert shap.KernelExplainer.not_equal(np.nan, 0) == shap.KernelExplainer.not_equal(None, "0")
    assert shap.KernelExplainer.not_equal("ab", "bc")
    assert not shap.KernelExplainer.not_equal("ab", "ab")
    assert shap.KernelExplainer.not_equal(pd.Timestamp("2017-01-01T12"), pd.Timestamp("2017-01-01T13"))
    assert not shap.KernelExplainer.not_equal(pd.Timestamp("2017-01-01T12"), pd.Timestamp("2017-01-01T12"))
    assert shap.KernelExplainer.not_equal(pd.Timestamp("2017-01-01T12"), pd.Timestamp("2017-01-01T13"))
    assert shap.KernelExplainer.not_equal(pd.Period("4Q2005"), pd.Period("3Q2005"))
    assert not shap.KernelExplainer.not_equal(pd.Period("4Q2005"), pd.Period("4Q2005"))


def test_kernel_explainer_with_tensors():
    # GH 3492
    tf = pytest.importorskip("tensorflow")
    tf.compat.v1.disable_eager_execution()

    X, _ = sklearn.datasets.make_classification(100, 6)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(10, input_shape=(6,), activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy")
    explainer = shap.KernelExplainer(model, X)
    explainer.shap_values(X[:1])


def test_kernel_multiclass_single_row():
    """Check a multi-input scenario."""
    X, y = shap.datasets.iris()

    lr = sklearn.linear_model.LogisticRegression(solver="lbfgs")
    lr.fit(X, y)
    pred = lr.predict_proba(X.iloc[[0], :])

    explainer = shap.KernelExplainer(lr.predict_proba, X)
    shap_values = explainer(X.iloc[0, :])
    np.testing.assert_allclose(shap_values.values.sum(0) + explainer.expected_value, pred.squeeze(), atol=1e-04)


def test_kernel_multiclass_multiple_rows():
    """Check a multi-input scenario."""
    X, y = shap.datasets.iris()

    lr = sklearn.linear_model.LogisticRegression(solver="lbfgs")
    lr.fit(X, y)
    pred = lr.predict_proba(X.iloc[[0, 1], :])

    explainer = shap.KernelExplainer(lr.predict_proba, X)
    shap_values = explainer(X.iloc[[0, 1], :])
    np.testing.assert_allclose(shap_values.values.sum(1) + explainer.expected_value, pred, atol=1e-04)


@pytest.mark.parametrize("nsamples", [3, 5, 10, 100])
def test_kernel_logits_zeros_ones_probs(nsamples):
    # GH 3912
    iris = sklearn.datasets.load_iris(as_frame=True)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        iris.data, iris.target, test_size=0.1, random_state=42
    )
    background_data = X_train.sample(10, random_state=42)

    rf = sklearn.ensemble.RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    X_test_sampled = X_test[:nsamples]

    explainer = shap.KernelExplainer(
        model=rf.predict_proba,
        data=background_data,
        keep_index=True,
        link="logit",
    )
    shap_values = explainer(X_test_sampled)
    pred = rf.predict_proba(X_test_sampled)

    np.testing.assert_allclose(sigm(shap_values.values.sum(1) + explainer.expected_value), pred, atol=1e-04)


@pytest.mark.parametrize("dt", [np.bool_, np.object_])
def test_explainer_non_number_dtype(dt):
    seed = 45479
    rng = np.random.default_rng(seed)
    X = rng.choice([True, False], size=(15, 8)).astype(dt)
    y = rng.choice([True, False], size=(15,)).astype(float)
    rf = sklearn.ensemble.RandomForestClassifier(random_state=seed)
    rf.fit(X, y)
    explainer = shap.KernelExplainer(model=rf.predict_proba, data=X, random_state=seed)
    shap_values = explainer(X)
    np.testing.assert_allclose(shap_values.values.max(), 0.26548, rtol=1e-2)

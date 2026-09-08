import logging
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse
import sklearn
import sklearn.datasets
import sklearn.ensemble
import sklearn.linear_model
from scipy.special import expit

import shap
from shap.utils._exceptions import DimensionError

from . import common


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
    shap.force_plot(explainer.expected_value[0], shap_values[0, :, 0], X_test.iloc[0, :], link="logit")  # type: ignore[index]


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
    shap.force_plot(explainer.expected_value[0], shap_values[0, :, 0], X_test.iloc[0, :], link="logit")  # type: ignore[index]


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

    shap_values = explainer.shap_values(X_test)  # type: ignore[assignment]
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
    # Skip test for Python versions below 3.9.17 and 3.10.12
    python_version = sys.version_info
    if python_version.major == 3 and python_version.minor == 9 and (python_version.micro < 17):
        pytest.skip(
            "Skipping test for Python 3.9 versions below 3.9.17. Loading the dataset will run into a tarfile error otherwise due to the missing filter keyword. See https://docs.python.org/3.9/library/tarfile.html#tarfile.TarFile.extractall"
        )
    elif python_version.major == 3 and python_version.minor == 10 and (python_version.micro < 12):
        pytest.skip(
            "Skipping test for Python 3.10 versions below 3.10.12. Loading the dataset will run into a tarfile error otherwise due to missing filter keyword. See https://docs.python.org/3.10/library/tarfile.html#tarfile.TarFile.extractall"
        )

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


@pytest.mark.parametrize("dt", [bool, object])
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


def test_serialization():
    model, data = common.basic_sklearn_scenario()
    common.test_serialization(shap.explainers.KernelExplainer, model.predict, data, data)


def _make_linear_data(n=30, p=4, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = X[:, 0] + 2.0 * X[:, 1]
    model = sklearn.linear_model.LinearRegression().fit(X, y)
    return X, y, model


def _make_classification_data(n=100, p=5, seed=0):
    X, y = sklearn.datasets.make_classification(n, p, random_state=seed)
    model = sklearn.linear_model.LogisticRegression(max_iter=1000).fit(X, y)
    return X, y, model


class TestL1RegVariants:
    def setup_method(self):
        self.X, _, self.model = _make_linear_data()
        self.explainer = shap.KernelExplainer(self.model.predict, self.X)
        self.X_test = self.X[:2]

    def test_l1_reg_aic(self):
        sv = self.explainer.shap_values(self.X_test, l1_reg="aic", silent=True)
        assert sv.shape == self.X_test.shape

    def test_l1_reg_bic(self):
        sv = self.explainer.shap_values(self.X_test, l1_reg="bic", silent=True)
        assert sv.shape == self.X_test.shape

    def test_l1_reg_float(self):
        sv = self.explainer.shap_values(self.X_test, l1_reg=0.05, silent=True)
        assert sv.shape == self.X_test.shape

    def test_l1_reg_false(self):
        sv = self.explainer.shap_values(self.X_test, l1_reg=False, silent=True)
        assert sv.shape == self.X_test.shape

    def test_l1_reg_zero(self):
        sv = self.explainer.shap_values(self.X_test, l1_reg=0, silent=True)
        assert sv.shape == self.X_test.shape

    def test_l1_reg_num_features_string(self):
        sv = self.explainer.shap_values(self.X_test, l1_reg="num_features(2)", silent=True)
        assert sv.shape == self.X_test.shape

    def test_l1_reg_auto_raises_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.explainer.shap_values(self.X[:1], l1_reg="auto", silent=True)
        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(dep_warnings) >= 1, "Expected DeprecationWarning for l1_reg='auto'"


class TestNsamples:
    def setup_method(self):
        self.X, _, self.model = _make_linear_data(p=4)

    def test_nsamples_auto_capped_at_max_samples(self):
        e = shap.KernelExplainer(self.model.predict, self.X)
        e.explain(self.X[:1], nsamples="auto", silent=True)
        assert e.nsamples == e.max_samples

    def test_explicit_nsamples_also_capped(self):
        e = shap.KernelExplainer(self.model.predict, self.X)
        e.explain(self.X[:1], nsamples=99999, silent=True)
        assert e.nsamples == e.max_samples

    @pytest.mark.parametrize("ns", [10, 50, 200])
    def test_explicit_nsamples_values(self, ns):
        X_large, _, model_large = _make_linear_data(p=12, n=30)
        e = shap.KernelExplainer(model_large.predict, X_large)
        sv = e.shap_values(X_large[:1], nsamples=ns, silent=True)
        assert sv.shape == (1, 12)

    def test_auto_nsamples_large_m_not_capped(self):
        rng = np.random.RandomState(0)
        p = 35
        X = rng.randn(10, p)
        model = sklearn.linear_model.LinearRegression().fit(X, X[:, 0])
        e = shap.KernelExplainer(model.predict, X)
        e.explain(X[:1], nsamples="auto", silent=True)
        assert e.nsamples == 2 * e.M + 2**11


def test_gc_collect_runs_without_error():
    X, _, model = _make_linear_data()
    e = shap.KernelExplainer(model.predict, X)
    sv = e.shap_values(X[:3], gc_collect=True, silent=True)
    assert sv.shape == (3, X.shape[1])


def test_silent_suppresses_progress_output(capsys):
    X, _, model = _make_linear_data()
    e = shap.KernelExplainer(model.predict, X)
    e.shap_values(X[:3], silent=True)
    captured = capsys.readouterr()
    assert "it/s" not in captured.err and "%" not in captured.err


def test_feature_names_from_constructor():
    X, _, model = _make_linear_data(p=3)
    names = ["alpha", "beta", "gamma"]
    e = shap.KernelExplainer(model.predict, X, feature_names=names)
    assert e.data_feature_names == names


def test_feature_names_from_dataframe_background():
    df = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "x2": [4.0, 5.0, 6.0]})
    model = sklearn.linear_model.LinearRegression().fit(df, df["x1"] + df["x2"])
    e = shap.KernelExplainer(model.predict, df)
    assert e.data_feature_names == ["x1", "x2"]


def test_explanation_call_stores_feature_names():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    model = sklearn.linear_model.LinearRegression().fit(df, df["a"] + df["b"])
    e = shap.KernelExplainer(model.predict, df)
    expl = e(df)
    assert expl.feature_names == ["a", "b"]


def test_explanation_data_is_numpy_array():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    model = sklearn.linear_model.LinearRegression().fit(df, df["a"])
    e = shap.KernelExplainer(model.predict, df)
    expl = e(df)
    assert isinstance(expl.data, np.ndarray)


def test_dimension_error_on_3d_input():
    X, _, model = _make_linear_data()
    e = shap.KernelExplainer(model.predict, X)
    with pytest.raises(DimensionError, match="1 or 2 dimensions"):
        e.shap_values(np.ones((2, 3, 4)))


def test_large_background_logs_warning(caplog):
    X = np.random.randn(150, 3)
    with caplog.at_level(logging.WARNING, logger="shap"):
        shap.KernelExplainer(lambda x: x[:, 0], X)
    assert any("150" in msg for msg in caplog.messages)


def test_no_features_vary_phi_is_zero():
    X = np.ones((5, 4))
    e = shap.KernelExplainer(lambda x: x[:, 0], X)
    phi = e.explain(np.ones((1, 4)))
    assert np.all(phi == 0), f"Expected all-zero phi, got {phi}"


def test_single_feature_varies_carries_full_effect():
    X = np.ones((5, 4))
    e = shap.KernelExplainer(lambda x: x[:, 0].astype(float), X)
    x_test = np.ones((1, 4))
    x_test[0, 0] = 10.0
    phi = e.explain(x_test)
    assert np.abs(phi[0] - 9.0) < 1e-8, f"Feature-0 phi should be 9.0, got {phi[0]}"
    assert np.all(phi[1:] == 0), f"All other phis should be 0, got {phi[1:]}"


def test_single_row_background_additivity():
    X_bg = np.array([[1.0, 2.0, 3.0]])
    X_test = np.array([[4.0, 5.0, 6.0]])

    def f(x):
        return x[:, 0] + x[:, 1]

    e = shap.KernelExplainer(f, X_bg)
    sv = e.shap_values(X_test, silent=True)
    residual = np.abs(sv.sum() + e.expected_value - f(X_test)[0])
    assert residual < 1e-6, f"Additivity violated: residual={residual}"


def test_kmeans_background():
    X, _, model = _make_linear_data(n=60)
    e = shap.KernelExplainer(model.predict, shap.kmeans(X, 5))
    sv = e.shap_values(X[:3], silent=True)
    assert sv.shape == (3, X.shape[1])


def test_sample_background():
    X, _, model = _make_linear_data(n=60)
    e = shap.KernelExplainer(model.predict, shap.sample(X, 8))
    sv = e.shap_values(X[:3], silent=True)
    assert sv.shape == (3, X.shape[1])


def test_model_returning_pandas_series():
    X, _, _ = _make_linear_data()
    e = shap.KernelExplainer(lambda x: pd.Series(x[:, 0]), X)
    sv = e.shap_values(X[:2], silent=True)
    assert sv.shape == (2, X.shape[1])


def test_model_returning_dataframe_initializes_correctly():
    X, _, _ = _make_linear_data()

    def model_df(x):
        return pd.DataFrame({"out": x[:, 0]})

    e = shap.KernelExplainer(model_df, X)
    assert e.fnull.shape == (1,)
    assert np.isfinite(float(e.expected_value))


def test_expected_value_equals_mean_model_output():
    X, _, model = _make_linear_data(n=50)
    e = shap.KernelExplainer(model.predict, X)
    assert np.abs(e.expected_value - float(np.mean(model.predict(X)))) < 1e-8


def test_expected_value_is_vector_for_multi_output():
    X, _, _ = _make_linear_data()
    e = shap.KernelExplainer(lambda x: np.column_stack([x[:, 0], x[:, 1]]), X)
    assert hasattr(e.expected_value, "__len__")
    assert len(e.expected_value) == 2


def test_expected_value_is_scalar_for_single_output():
    X, _, model = _make_linear_data()
    e = shap.KernelExplainer(model.predict, X)
    assert isinstance(e.expected_value, float)


def test_additivity_linear_model(random_seed):
    X, _, model = _make_linear_data(seed=random_seed)
    e = shap.KernelExplainer(model.predict, X)
    sv = e.shap_values(X[:5], nsamples=500, silent=True)
    residuals = np.abs(sv.sum(1) + e.expected_value - model.predict(X[:5]))
    assert residuals.max() < 1e-6


def test_additivity_random_forest():
    X, y, _ = _make_linear_data(n=80)
    rf = sklearn.ensemble.RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y)
    e = shap.KernelExplainer(rf.predict, X[:20])
    sv = e.shap_values(X[:3], nsamples=300, silent=True)
    residuals = np.abs(sv.sum(1) + e.expected_value - rf.predict(X[:3]))
    assert residuals.max() < 5e-2


def test_additivity_multioutput():
    rng = np.random.RandomState(0)
    X = rng.randn(20, 4)

    def multi_out(x):
        return np.column_stack([x[:, 0] + x[:, 1], x[:, 2] - x[:, 3]])

    e = shap.KernelExplainer(multi_out, X)
    sv = e.shap_values(X[:3], nsamples=300, silent=True)
    pred = multi_out(X[:3])
    for d in range(2):
        ev = np.asarray(e.expected_value)
        residuals = np.abs(sv[:, :, d].sum(1) + ev[d] - pred[:, d])
        assert residuals.max() < 1e-4


def test_identity_link_sum_property():
    X, _, model = _make_linear_data()
    e = shap.KernelExplainer(model.predict, X)
    sv = e.shap_values(X[:2], silent=True)
    np.testing.assert_allclose(sv.sum(1) + e.expected_value, model.predict(X[:2]), atol=1e-6)


def test_logit_link_expected_value_is_finite():
    X, y, model = _make_classification_data()
    e = shap.KernelExplainer(model.predict_proba, X, link="logit")
    assert np.all(np.isfinite(np.asarray(e.expected_value)))


def test_logit_link_recovers_probabilities():
    X, y, model = _make_classification_data(n=60)
    e = shap.KernelExplainer(model.predict_proba, X[:20], link="logit")
    sv = e.shap_values(X[:3], nsamples=300, silent=True)
    recovered = expit(sv.sum(1) + e.expected_value)
    np.testing.assert_allclose(recovered, model.predict_proba(X[:3]), atol=1e-3)


def test_vector_out_false_for_single_output():
    X, _, model = _make_linear_data()
    e = shap.KernelExplainer(model.predict, X)
    assert e.vector_out is False
    assert e.D == 1


def test_vector_out_true_for_multi_output():
    X, _, _ = _make_linear_data()
    e = shap.KernelExplainer(lambda x: np.column_stack([x[:, 0], x[:, 1]]), X)
    assert e.vector_out is True
    assert e.D == 2


def test_multioutput_shap_values_shape():
    rng = np.random.RandomState(7)
    X = rng.randn(15, 4)
    e = shap.KernelExplainer(lambda x: np.column_stack([x[:, 0], x[:, 1], x[:, 2]]), X)
    sv = e.shap_values(X[:3], silent=True)
    assert sv.shape == (3, 4, 3)


def test_not_equal_numpy_float32_equal():
    assert shap.KernelExplainer.not_equal(np.float32(1.0), np.float32(1.0)) == 0


def test_not_equal_numpy_float32_unequal():
    assert shap.KernelExplainer.not_equal(np.float32(1.0), np.float32(2.0)) == 1


def test_not_equal_numpy_int_equal():
    assert shap.KernelExplainer.not_equal(np.int32(5), np.int32(5)) == 0


def test_not_equal_numpy_int_unequal():
    assert shap.KernelExplainer.not_equal(np.int32(5), np.int32(6)) == 1


def test_not_equal_numpy_nan_equals_nan():
    assert shap.KernelExplainer.not_equal(np.float64(np.nan), np.float64(np.nan)) == 0


def test_not_equal_int_float_cross_type_equal():
    assert shap.KernelExplainer.not_equal(1, 1.0) == 0


def test_not_equal_int_float_cross_type_unequal():
    assert shap.KernelExplainer.not_equal(1, 2.0) == 1


def test_not_equal_none_vs_none():
    assert shap.KernelExplainer.not_equal(None, None) == 0


def test_not_equal_none_vs_value():
    assert shap.KernelExplainer.not_equal(None, 1) == 1


def test_not_equal_bool_numpy_equal():
    a, b = np.array([True], dtype=bool), np.array([True], dtype=bool)
    assert shap.KernelExplainer.not_equal(a[0], b[0]) == 0


def test_not_equal_bool_numpy_unequal():
    a, b = np.array([True], dtype=bool), np.array([False], dtype=bool)
    assert shap.KernelExplainer.not_equal(a[0], b[0]) == 1


def test_keep_index_ordered_runs_without_error():
    df = pd.DataFrame({"a": [3.0, 1.0, 2.0], "b": [6.0, 4.0, 5.0]}, index=[30, 10, 20])
    e = shap.KernelExplainer(lambda x: x["a"].values + x["b"].values, df, keep_index=True, keep_index_ordered=True)
    sv = e.shap_values(df, silent=True)
    assert sv.shape == (3, 2)


def test_sparse_zero_background_with_sparse_test():
    rng = np.random.RandomState(0)
    X_dense = rng.randn(20, 8)
    model = sklearn.linear_model.LinearRegression().fit(X_dense, X_dense[:, 0])
    _, cols = X_dense.shape
    bg = scipy.sparse.csr_matrix((1, cols))
    e = shap.KernelExplainer(model.predict, bg)
    sv = e.shap_values(scipy.sparse.csr_matrix(X_dense[:2]), silent=True)
    assert sv.shape == (2, cols)


def test_sparse_csr_background_and_input():
    rng = np.random.RandomState(1)
    X_dense = rng.randn(30, 6)
    model = sklearn.linear_model.LinearRegression().fit(X_dense, X_dense[:, 0])
    e = shap.KernelExplainer(model.predict, scipy.sparse.csr_matrix(X_dense[:5]))
    sv = e.shap_values(scipy.sparse.csr_matrix(X_dense[:3]), silent=True)
    assert sv.shape == (3, 6)


def test_shap_values_with_pandas_series_input():
    X, _, model = _make_linear_data(p=4)
    e = shap.KernelExplainer(model.predict, X)
    row = pd.Series(X[0], index=[f"f{i}" for i in range(4)])
    sv = e.shap_values(row, silent=True)
    assert sv.shape == (4,)


def test_multiclass_additivity_all_outputs():
    X, y = shap.datasets.iris()
    lr = sklearn.linear_model.LogisticRegression(solver="lbfgs", max_iter=200).fit(X, y)
    X_test = X.iloc[:3, :]
    e = shap.KernelExplainer(lr.predict_proba, X)
    sv = e.shap_values(X_test, nsamples=200, silent=True)
    pred = lr.predict_proba(X_test)
    for cls in range(3):
        ev = np.asarray(e.expected_value)
        residuals = np.abs(sv[:, :, cls].sum(1) + ev[cls] - pred[:, cls])
        assert residuals.max() < 1e-3


def test_call_method_returns_explanation_object():
    X, _, model = _make_linear_data()
    e = shap.KernelExplainer(model.predict, X)
    expl = e(X[:3])
    assert isinstance(expl, shap.Explanation)
    assert expl.values.shape == (3, X.shape[1])


def test_call_method_base_values_tiled_correctly():
    X, _, model = _make_linear_data()
    e = shap.KernelExplainer(model.predict, X)
    expl = e(X[:4])
    assert expl.base_values.shape == (4,)
    np.testing.assert_array_equal(expl.base_values, e.expected_value)


def test_call_method_multioutput_base_values_shape():
    rng = np.random.RandomState(0)
    X = rng.randn(20, 4)
    e = shap.KernelExplainer(lambda x: np.column_stack([x[:, 0], x[:, 1]]), X)
    expl = e(X[:3])
    assert expl.base_values.shape == (3, 2)


def test_unsupported_background_type_raises_type_error():
    with pytest.raises(TypeError):
        shap.KernelExplainer(lambda x: x[:, 0], "not_a_valid_data_source")


def test_n_and_p_attributes_set_correctly():
    X = np.random.randn(12, 7)
    e = shap.KernelExplainer(lambda x: x[:, 0], X)
    assert e.N == 12
    assert e.P == 7


def test_constant_feature_gets_zero_shap_value():
    rng = np.random.RandomState(0)
    X = rng.randn(20, 4)
    X[:, 2] = 5.0
    e = shap.KernelExplainer(lambda x: x[:, 0] + x[:, 1], X)
    x_test = np.array([[1.0, 2.0, 5.0, 3.0]])
    sv = e.shap_values(x_test, nsamples=100, silent=True)
    assert sv[0, 2] == 0.0


def test_varying_groups_sparse_nonzero_detection():
    rng = np.random.RandomState(0)
    X_dense = rng.randn(10, 5)
    model = sklearn.linear_model.LinearRegression().fit(X_dense, X_dense[:, 0])
    e = shap.KernelExplainer(model.predict, scipy.sparse.csr_matrix(np.zeros((1, 5))))
    sv = e.shap_values(scipy.sparse.csr_matrix(X_dense[:2]), silent=True)
    assert sv.shape == (2, 5)

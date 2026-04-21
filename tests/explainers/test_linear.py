"""Unit tests for the Linear explainer."""

import numpy as np
import pytest
import scipy.special
from sklearn.datasets import make_multilabel_classification
from sklearn.linear_model import LogisticRegression, Ridge

import shap
from shap import maskers
from shap.utils._exceptions import InvalidFeaturePerturbationError


def test_tied_pair():
    beta = np.array([1, 0, 0])
    mu = np.zeros(3)
    Sigma = np.array([[1, 0.999999, 0], [0.999999, 1, 0], [0, 0, 1]])
    X = np.ones((1, 3))
    masker = maskers.Impute({"mean": mu, "cov": Sigma})
    explainer = shap.LinearExplainer((beta, 0), masker)
    assert np.abs(explainer.shap_values(X) - np.array([0.5, 0.5, 0])).max() < 0.05


def test_tied_pair_independent():
    beta = np.array([1, 0, 0])
    mu = np.zeros(3)
    Sigma = np.array([[1, 0.999999, 0], [0.999999, 1, 0], [0, 0, 1]])
    X = np.ones((1, 3))
    masker = maskers.Independent({"mean": mu, "cov": Sigma})
    explainer = shap.LinearExplainer((beta, 0), masker)
    assert np.abs(explainer.shap_values(X) - np.array([1, 0, 0])).max() < 0.05


def test_tied_pair_new():
    beta = np.array([1, 0, 0])
    mu = np.zeros(3)
    Sigma = np.array([[1, 0.999999, 0], [0.999999, 1, 0], [0, 0, 1]])
    X = np.ones((1, 3))
    explainer = shap.explainers.LinearExplainer((beta, 0), shap.maskers.Impute({"mean": mu, "cov": Sigma}))
    assert np.abs(explainer.shap_values(X) - np.array([0.5, 0.5, 0])).max() < 0.05


def test_wrong_masker():
    with pytest.raises(NotImplementedError):
        shap.explainers.LinearExplainer((0, 0), shap.maskers.Fixed())


def test_tied_triple():
    beta = np.array([0, 1, 0, 0])
    mu = 1 * np.ones(4)
    Sigma = np.array([[1, 0.999999, 0.999999, 0], [0.999999, 1, 0.999999, 0], [0.999999, 0.999999, 1, 0], [0, 0, 0, 1]])
    X = 2 * np.ones((1, 4))
    masker = maskers.Impute({"mean": mu, "cov": Sigma})
    explainer = shap.LinearExplainer((beta, 0), masker)
    assert explainer.expected_value == 1
    assert np.abs(explainer.shap_values(X) - np.array([0.33333, 0.33333, 0.33333, 0])).max() < 0.05


def test_sklearn_linear():
    Ridge = pytest.importorskip("sklearn.linear_model").Ridge

    # train linear model
    X, y = shap.datasets.california(n_points=100)
    model = Ridge(0.1)
    model.fit(X, y)

    # explain the model's predictions using SHAP values
    explainer = shap.LinearExplainer(model, X)
    assert np.abs(explainer.expected_value - model.predict(X).mean()) < 1e-6
    explainer.shap_values(X)


def test_sklearn_linear_old_style():
    Ridge = pytest.importorskip("sklearn.linear_model").Ridge

    # train linear model
    X, y = shap.datasets.california(n_points=100)
    model = Ridge(0.1)
    model.fit(X, y)

    # explain the model's predictions using SHAP values
    explainer = shap.LinearExplainer(model, maskers.Independent(X))
    assert np.abs(explainer.expected_value - model.predict(X).mean()) < 1e-6
    explainer.shap_values(X)


def test_sklearn_linear_new():
    Ridge = pytest.importorskip("sklearn.linear_model").Ridge

    # train linear model
    X, y = shap.datasets.california(n_points=100)
    model = Ridge(0.1)
    model.fit(X, y)

    # explain the model's predictions using SHAP values
    explainer = shap.explainers.LinearExplainer(model, X)
    shap_values = explainer(X)
    assert np.abs(shap_values.values.sum(1) + shap_values.base_values - model.predict(X)).max() < 1e-6  # type: ignore[union-attr, union-attr]
    assert np.abs(shap_values.base_values[0] - model.predict(X).mean()) < 1e-6  # type: ignore[union-attr]


def test_sklearn_multiclass_no_intercept():
    Ridge = pytest.importorskip("sklearn.linear_model").Ridge

    # train linear model
    X, y = shap.datasets.california(n_points=100)

    # make y multiclass
    multiclass_y = np.expand_dims(y, axis=-1)
    model = Ridge(fit_intercept=False)
    model.fit(X, multiclass_y)

    # explain the model's predictions using SHAP values
    explainer = shap.LinearExplainer(model, X)
    assert np.abs(explainer.expected_value - model.predict(X).mean()) < 1e-6
    explainer.shap_values(X)


def test_perfect_colinear():
    LinearRegression = pytest.importorskip("sklearn.linear_model").LinearRegression

    X, y = shap.datasets.california(n_points=100)
    X.iloc[:, 0] = X.iloc[:, 4]  # test duplicated features
    X.iloc[:, 5] = X.iloc[:, 6] - X.iloc[:, 6]  # test multiple colinear features
    X.iloc[:, 3] = 0  # test null features
    model = LinearRegression()
    model.fit(X, y)
    explainer = shap.LinearExplainer(model, maskers.Impute(X))
    shap_values = explainer.shap_values(X)
    assert np.abs(shap_values.sum(1) - model.predict(X) + model.predict(X).mean()).sum() < 1e-7


def test_shape_values_linear_many_features():
    Ridge = pytest.importorskip("sklearn.linear_model").Ridge

    coef = np.array([1, 2]).T

    # FIXME: this test should ideally pass with any random seed. See #2960
    random_seed = 0
    rs = np.random.RandomState(random_seed)
    # generate linear data
    X = rs.normal(1, 10, size=(1000, len(coef)))
    y = np.dot(X, coef) + 1 + rs.normal(scale=0.1, size=1000)

    # train linear model
    model = Ridge(0.1)
    model.fit(X, y)

    # explain the model's predictions using SHAP values
    explainer = shap.LinearExplainer(model, X.mean(0).reshape(1, -1))

    values = explainer.shap_values(X)

    assert values.shape == (1000, 2)

    expected = (X - X.mean(0)) * coef
    np.testing.assert_allclose(expected - values, 0, atol=0.01)


def test_single_feature(random_seed):
    """Make sure things work with a univariate linear regression."""
    Ridge = pytest.importorskip("sklearn.linear_model").Ridge

    # generate linear data
    rs = np.random.RandomState(random_seed)
    X = rs.normal(1, 10, size=(100, 1))
    y = 2 * X[:, 0] + 1 + rs.normal(scale=0.1, size=100)

    # train linear model
    model = Ridge(0.1)
    model.fit(X, y)

    # explain the model's predictions using SHAP values
    explainer = shap.LinearExplainer(model, X)
    shap_values = explainer.shap_values(X)
    assert np.abs(explainer.expected_value - model.predict(X).mean()) < 1e-6
    assert np.max(np.abs(explainer.expected_value + shap_values.sum(1) - model.predict(X))) < 1e-6


def test_sparse():
    """Validate running LinearExplainer on scipy sparse data"""
    n_features = 20
    X, y = make_multilabel_classification(n_samples=100, sparse=True, n_features=n_features, n_classes=1, n_labels=2)

    # train linear model
    model = LogisticRegression()
    model.fit(X, y.squeeze())

    # explain the model's predictions using SHAP values
    explainer = shap.LinearExplainer(model, X)
    shap_values = explainer.shap_values(X)
    assert (
        np.max(
            np.abs(scipy.special.expit(explainer.expected_value + shap_values.sum(1)) - model.predict_proba(X)[:, 1])
        )
        < 1e-6
    )


@pytest.mark.xfail(reason="This should pass but it doesn't.")
def test_sparse_multi_class():
    """Validate running LinearExplainer on scipy sparse data"""
    n_features = 4
    X, y = make_multilabel_classification(n_samples=100, sparse=False, n_features=n_features, n_classes=3, n_labels=2)
    y = np.argmax(y, axis=1)

    # train linear model
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    pred = model.predict_proba(X)

    # explain the model's predictions using SHAP values
    explainer = shap.LinearExplainer(model, X)
    shap_values = explainer(X)
    np.testing.assert_allclose(
        scipy.special.expit(shap_values.values.sum(1) + shap_values.base_values),  # type: ignore[union-attr]
        pred,
        atol=1e-6,
    )


@pytest.mark.filterwarnings("ignore:The feature_perturbation option is now deprecated")
def test_invalid_feature_perturbation_raises():
    # train linear model
    X, y = shap.datasets.california(n_points=100)
    model = Ridge(0.1).fit(X, y)

    with pytest.raises(InvalidFeaturePerturbationError, match="feature_perturbation must be one of "):
        shap.LinearExplainer(model, X, feature_perturbation="nonsense")  # type: ignore[arg-type]


@pytest.mark.filterwarnings("ignore:The feature_perturbation option is now deprecated")
@pytest.mark.parametrize(
    "feature_pertubation,masker",
    [
        (None, shap.maskers.Independent),
        ("interventional", shap.maskers.Independent),
        ("correlation_dependent", shap.maskers.Impute),
    ],
)
def test_feature_perturbation_sets_correct_masker(feature_pertubation, masker):
    Ridge = pytest.importorskip("sklearn.linear_model").Ridge

    # train linear model
    X, y = shap.datasets.california(n_points=100)
    model = Ridge(0.1)
    model.fit(X, y)

    explainer = shap.explainers.LinearExplainer(model, X, feature_perturbation=feature_pertubation)
    assert isinstance(explainer.masker, masker)


def test_interventional_multi_regression():
    ridge = pytest.importorskip("sklearn.linear_model").Ridge

    # train linear model
    X, y = shap.datasets.linnerud(n_points=100)
    model = ridge(0.1)
    model.fit(X, y)
    outputs = model.predict(X)

    explainer = shap.explainers.LinearExplainer(model, maskers.Independent(X))
    shap_values = explainer.shap_values(X)
    assert np.allclose(shap_values.sum(1) + explainer.expected_value, outputs, atol=1e-6)


def test_feature_dependence_kwarg_raises():
    """feature_dependence was renamed to feature_perturbation; passing it must error."""
    X, _ = shap.datasets.california(n_points=20)
    with pytest.raises(ValueError, match="feature_dependence has been renamed"):
        shap.LinearExplainer((np.zeros(X.shape[1]), 0.0), X, feature_dependence="interventional")  # type: ignore[arg-type]


def test_background_tuple_mean_cov_interventional():
    """Passing a (mean, cov) tuple as the background should wrap it in Independent."""
    rs = np.random.RandomState(0)
    n_features = 4
    mean = rs.randn(n_features)
    cov = np.eye(n_features)
    beta = np.ones(n_features)

    explainer = shap.LinearExplainer((beta, 0.5), (mean, cov))
    assert isinstance(explainer.masker, maskers.Independent)
    # expected_value = beta @ mean + intercept
    np.testing.assert_allclose(explainer.expected_value, beta @ mean + 0.5, atol=1e-6)


@pytest.mark.filterwarnings("ignore:The feature_perturbation option is now deprecated")
def test_background_tuple_mean_cov_correlation_dependent():
    """Passing a (mean, cov) tuple with correlation_dependent should wrap it in Impute."""
    rs = np.random.RandomState(0)
    n_features = 3
    mean = rs.randn(n_features)
    cov = np.eye(n_features)
    beta = np.ones(n_features)

    explainer = shap.LinearExplainer(
        (beta, 0.0),
        (mean, cov),
        feature_perturbation="correlation_dependent",
        nsamples=50,
    )
    assert isinstance(explainer.masker, maskers.Impute)


def test_no_background_raises():
    """An Independent masker built without data must still cause a ValueError in the explainer."""
    beta = np.ones(3)
    # pass a masker whose .data is None by constructing Independent with a data=None shortcut
    masker = maskers.Independent({"mean": np.zeros(3), "cov": np.eye(3)})
    # manually clear data attributes to hit the "data is None" branch
    masker.mean = None  # type: ignore[attr-defined]
    masker.data = None  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="background data distribution must be provided"):
        shap.LinearExplainer((beta, 0.0), masker)


def test_sparse_background_with_correlation_dependent_raises():
    """Sparse background data is not supported for correlation_dependent."""
    from scipy.sparse import csr_matrix

    rs = np.random.RandomState(0)
    X_sparse = csr_matrix(rs.randn(50, 4))
    # bypass the masker-wrapping shortcut by passing an Impute masker directly on sparse data
    with pytest.raises((NotImplementedError, TypeError)):
        masker = maskers.Impute(X_sparse)  # may itself reject sparse
        shap.LinearExplainer((np.ones(4), 0.0), masker)


def test_unknown_model_raises():
    """A model without coef_/intercept_ and not a tuple must raise InvalidModelError."""
    from shap.utils._exceptions import InvalidModelError

    class NotALinearModel:
        pass

    X, _ = shap.datasets.california(n_points=20)
    with pytest.raises(InvalidModelError, match="unknown model type"):
        shap.LinearExplainer(NotALinearModel(), X)


def test_nsamples_warning_for_interventional():
    """nsamples is only meaningful for correlation_dependent; interventional emits a warning."""
    X, _ = shap.datasets.california(n_points=20)
    with pytest.warns(UserWarning, match="nsamples has no effect"):
        shap.LinearExplainer((np.zeros(X.shape[1]), 0.0), X, nsamples=500)


def test_explain_row_via_call_with_dataframe():
    """Calling the explainer (rather than shap_values) on a DataFrame hits explain_row."""
    Ridge = pytest.importorskip("sklearn.linear_model").Ridge
    X, y = shap.datasets.california(n_points=30)
    model = Ridge(0.1).fit(X, y)

    explainer = shap.explainers.LinearExplainer(model, X)
    result = explainer(X.iloc[:5])  # DataFrame routed through explain_row
    assert result.values.shape == (5, X.shape[1])
    np.testing.assert_allclose(
        result.values.sum(axis=1) + result.base_values, model.predict(X.iloc[:5]), atol=1e-6
    )


def test_shap_values_series_input():
    """A pandas Series (single row) should be handled by shap_values."""
    Ridge = pytest.importorskip("sklearn.linear_model").Ridge
    X, y = shap.datasets.california(n_points=20)
    model = Ridge(0.1).fit(X, y)

    explainer = shap.LinearExplainer(model, X)
    row = X.iloc[0]  # pandas Series — 1D
    values = explainer.shap_values(row)
    assert values.shape == (X.shape[1],)


@pytest.mark.parametrize("bad_input", [np.zeros((2, 2, 2)), np.zeros((3, 3, 3, 3))])
def test_shap_values_dimension_error(bad_input):
    """shap_values must reject inputs with more than 2 dimensions."""
    from shap.utils._exceptions import DimensionError

    X, _ = shap.datasets.california(n_points=20)
    explainer = shap.LinearExplainer((np.ones(X.shape[1]), 0.0), X)
    with pytest.raises(DimensionError, match="1 or 2 dimensions"):
        explainer.shap_values(bad_input)


def test_supports_model_with_masker():
    """supports_model_with_masker must return False for unsupported maskers or models."""
    X, _ = shap.datasets.california(n_points=20)
    independent = maskers.Independent(X)

    # tuple model + Independent masker → supported
    assert shap.LinearExplainer.supports_model_with_masker((np.ones(X.shape[1]), 0.0), independent) is True

    # unsupported masker type
    assert shap.LinearExplainer.supports_model_with_masker((np.ones(X.shape[1]), 0.0), maskers.Fixed()) is False

    # unsupported model type
    class NotALinearModel:
        pass

    assert shap.LinearExplainer.supports_model_with_masker(NotALinearModel(), independent) is False


def test_sparse_multi_output_shap_values():
    """Sparse input with a multi-output linear model returns a stacked ndarray."""
    from scipy.sparse import csr_matrix
    from sklearn.linear_model import Ridge

    rs = np.random.RandomState(0)
    X_dense = rs.randn(50, 4)
    Y = np.column_stack([X_dense[:, 0] + rs.randn(50) * 0.1, X_dense[:, 1] * 2 + rs.randn(50) * 0.1])
    model = Ridge(0.1).fit(X_dense, Y)

    X_sparse = csr_matrix(X_dense[:5])
    explainer = shap.LinearExplainer(model, X_dense)
    values = explainer.shap_values(X_sparse)
    assert values.shape == (5, 4, 2)


def test_dense_multi_output_shap_values():
    """Dense input with a multi-output linear model returns a stacked ndarray."""
    from sklearn.linear_model import Ridge

    rs = np.random.RandomState(0)
    X = rs.randn(50, 4)
    Y = np.column_stack([X[:, 0] + rs.randn(50) * 0.1, X[:, 1] * 2 + rs.randn(50) * 0.1])
    model = Ridge(0.1).fit(X, Y)

    explainer = shap.LinearExplainer(model, X)
    values = explainer.shap_values(X[:3])
    assert values.shape == (3, 4, 2)
    # additivity per output
    preds = model.predict(X[:3])
    np.testing.assert_allclose(values.sum(axis=1) + explainer.expected_value, preds, atol=1e-6)


def test_explain_row_dimension_error():
    """explain_row (via __call__) must reject >2D input."""
    from shap.utils._exceptions import DimensionError

    X, _ = shap.datasets.california(n_points=20)
    explainer = shap.explainers.LinearExplainer((np.ones(X.shape[1]), 0.0), X)
    with pytest.raises(DimensionError, match="1 or 2 dimensions"):
        explainer.explain_row(
            np.zeros((2, 2, 2)),
            max_evals="auto",
            main_effects=False,
            error_bounds=False,
            outputs=None,
            silent=True,
        )

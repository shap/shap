# pylint: disable=missing-function-docstring
"""Test gpu accelerated tree functions."""
import sklearn
import pytest
import numpy as np
import shap


def test_front_page_xgboost():
    xgboost = pytest.importorskip("xgboost")

    # load JS visualization code to notebook
    shap.initjs()

    # train XGBoost model
    X, y = shap.datasets.boston()
    model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)

    # explain the model's predictions using SHAP values
    explainer = shap.GPUTreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # visualize the first prediction's explaination
    shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :])

    # visualize the training set predictions
    shap.force_plot(explainer.expected_value, shap_values, X)

    # create a SHAP dependence plot to show the effect of a single feature across the whole dataset
    shap.dependence_plot(5, shap_values, X, show=False)
    shap.dependence_plot("RM", shap_values, X, show=False)

    # summarize the effects of all the features
    shap.summary_plot(shap_values, X, show=False)


rs = np.random.RandomState(15921)  # pylint: disable=no-member
n = 100
m = 4
datasets = {'regression': (rs.randn(n, m), rs.randn(n)),
            'binary': (rs.randn(n, m), rs.binomial(1, 0.5, n)),
            'multiclass': (rs.randn(n, m), rs.randint(0, 5, n))}


def task_xfail(func):
    def inner():
        return pytest.param(func(), marks=pytest.mark.xfail)

    return inner


def xgboost_base():
    # pylint: disable=import-outside-toplevel
    try:
        import xgboost
    except ImportError:
        return pytest.param(marks=pytest.mark.skip)
    X, y = datasets['regression']

    model = xgboost.XGBRegressor()
    model.fit(X, y)
    return model.get_booster(), X, model.predict(X)


def xgboost_regressor():
    # pylint: disable=import-outside-toplevel
    try:
        import xgboost
    except ImportError:
        return pytest.param(marks=pytest.mark.skip)

    X, y = datasets['regression']

    model = xgboost.XGBRegressor()
    model.fit(X, y)
    return model, X, model.predict(X)


def xgboost_binary_classifier():
    # pylint: disable=import-outside-toplevel
    try:
        import xgboost
    except ImportError:
        return pytest.param(marks=pytest.mark.skip)

    X, y = datasets['binary']

    model = xgboost.XGBClassifier()
    model.fit(X, y)
    return model, X, model.predict(X, output_margin=True)


@task_xfail
def xgboost_multiclass_classifier():
    # pylint: disable=import-outside-toplevel
    try:
        import xgboost
    except ImportError:
        return pytest.param(marks=pytest.mark.skip)

    X, y = datasets['multiclass']

    model = xgboost.XGBClassifier()
    model.fit(X, y)
    return model, X, model.predict(X, output_margin=True)


def lightgbm_base():
    # pylint: disable=import-outside-toplevel
    try:
        import lightgbm
    except ImportError:
        return pytest.param(marks=pytest.mark.skip)
    X, y = datasets['regression']

    model = lightgbm.LGBMRegressor()
    model.fit(X, y)
    return model.booster_, X, model.predict(X)


def lightgbm_regression():
    # pylint: disable=import-outside-toplevel
    try:
        import lightgbm
    except ImportError:
        return pytest.param(marks=pytest.mark.skip)
    X, y = datasets['regression']

    model = lightgbm.LGBMRegressor()
    model.fit(X, y)
    return model, X, model.predict(X)


def lightgbm_binary_classifier():
    # pylint: disable=import-outside-toplevel
    try:
        import lightgbm
    except ImportError:
        return pytest.param(marks=pytest.mark.skip)
    X, y = datasets['binary']

    model = lightgbm.LGBMClassifier()
    model.fit(X, y)
    return model, X, model.predict(X, raw_score=True)


@task_xfail
def lightgbm_multiclass_classifier():
    # pylint: disable=import-outside-toplevel
    try:
        import lightgbm
    except ImportError:
        return pytest.param(marks=pytest.mark.skip)
    X, y = datasets['multiclass']

    model = lightgbm.LGBMClassifier()
    model.fit(X, y)
    return model, X, model.predict(X, raw_score=True)


def rf_regressor():
    X, y = datasets['regression']
    model = sklearn.ensemble.RandomForestRegressor()
    model.fit(X, y)
    return model, X, model.predict(X)


@task_xfail
def rf_binary_classifier():
    X, y = datasets['binary']
    model = sklearn.ensemble.RandomForestClassifier()
    model.fit(X, y)
    return model, X, model.predict(X)


@task_xfail
def rf_multiclass_classifier():
    X, y = datasets['multiclass']
    model = sklearn.ensemble.RandomForestClassifier()
    model.fit(X, y)
    return model, X, model.predict(X)


tasks = [xgboost_base(), xgboost_regressor(), xgboost_binary_classifier(),
         xgboost_multiclass_classifier(), lightgbm_base(), lightgbm_regression(),
         lightgbm_binary_classifier(), lightgbm_multiclass_classifier(), rf_binary_classifier(),
         rf_regressor(), rf_multiclass_classifier()]


# pretty print tasks
def idfn(task):
    model, _, _ = task
    return type(model).__module__ + '.' + type(model).__qualname__


@pytest.mark.parametrize("task", tasks, ids=idfn)
@pytest.mark.parametrize("feature_perturbation",
                         [pytest.param("interventional", marks=pytest.mark.xfail),
                          "tree_path_dependent"])
def test_gpu_tree_explainer_shap(task, feature_perturbation):
    model, X, margin = task
    ex = shap.GPUTreeExplainer(model, X, feature_perturbation=feature_perturbation)
    shap_values = ex.shap_values(X, check_additivity=False)

    assert np.abs(np.sum(shap_values, 1) + ex.expected_value - margin).max() < 1e-4, \
        "SHAP values don't sum to model output!"


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("task", tasks, ids=idfn)
@pytest.mark.parametrize("feature_perturbation",
                         ["tree_path_dependent"])
def test_gpu_tree_explainer_shap_interactions(task, feature_perturbation):
    model, X, margin = task
    ex = shap.GPUTreeExplainer(model, X, feature_perturbation=feature_perturbation)
    shap_values = np.array(ex.shap_interaction_values(X), copy=False)

    assert np.abs(np.sum(shap_values, axis=(len(shap_values.shape) - 1, len(
        shap_values.shape) - 2)) + ex.expected_value - margin).max() < 1e-4, \
        "SHAP values don't sum to model output!"

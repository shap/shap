"""Tests for TreeEnsemble.from_trees."""

import sys

import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import shap
from shap.explainers._tree import TreeEnsemble


def _assert_explanation_additivity(explanation, predictions, err_msg, atol=1e-5):
    summed = explanation.values.sum(1) + explanation.base_values
    pred = np.asarray(predictions)
    if summed.ndim == 1 and pred.ndim > 1 and pred.shape[1] == 1:
        pred = pred[:, 0]
    np.testing.assert_allclose(summed, pred, err_msg=err_msg, atol=atol)


def _roundtrip_tree_ensemble(model, X, data_missing=None, model_output="raw"):
    existing = TreeEnsemble(model, data=X, data_missing=data_missing, model_output=model_output)
    rebuilt = TreeEnsemble.from_trees(
        existing.trees,
        base_offset=existing.base_offset,
        model_output=existing.model_output,
        objective=existing.objective,
        tree_output=existing.tree_output,
        internal_dtype=existing.internal_dtype,
        input_dtype=existing.input_dtype,
        data=None,
        data_missing=data_missing,
        fully_defined_weighting=existing.fully_defined_weighting,
        tree_limit=existing.tree_limit,
        num_stacked_models=existing.num_stacked_models,
        cat_feature_indices=existing.cat_feature_indices,
        model_type="internal",
    )
    return existing, rebuilt


@pytest.fixture
def configure_pyspark_python(monkeypatch):
    monkeypatch.setenv("PYSPARK_PYTHON", sys.executable)
    monkeypatch.setenv("PYSPARK_DRIVER_PYTHON", sys.executable)


def test_from_trees_accepts_sklearn_tree_structure():
    X, y = shap.datasets.california(n_points=200)
    model = DecisionTreeRegressor(max_depth=3, random_state=0)
    model.fit(X, y)

    existing = TreeEnsemble(model, data=X, data_missing=None, model_output="raw")
    rebuilt = TreeEnsemble.from_trees(
        [model.tree_],
        base_offset=existing.base_offset,
        model_output=existing.model_output,
        objective=existing.objective,
        tree_output=existing.tree_output,
        internal_dtype=existing.internal_dtype,
        input_dtype=existing.input_dtype,
        data=None,
        data_missing=None,
        fully_defined_weighting=existing.fully_defined_weighting,
        model_type="internal",
    )

    np.testing.assert_allclose(existing.predict(X), rebuilt.predict(X))

    existing_explanation = shap.TreeExplainer(existing)(X[:10])
    rebuilt_explanation = shap.TreeExplainer(rebuilt)(X[:10])
    np.testing.assert_allclose(existing_explanation.values, rebuilt_explanation.values)
    np.testing.assert_allclose(existing_explanation.base_values, rebuilt_explanation.base_values)

    auto_explainer = shap.Explainer(rebuilt, masker=X)
    assert isinstance(auto_explainer, shap.TreeExplainer)


def test_from_trees_roundtrip_supported_sklearn_models():
    cases = []

    X_reg, y_reg = shap.datasets.california(n_points=200)
    regressor_models = [
        DecisionTreeRegressor(max_depth=3, random_state=0),
        sklearn.ensemble.RandomForestRegressor(n_estimators=5, random_state=0),
        sklearn.ensemble.ExtraTreesRegressor(n_estimators=5, random_state=0),
        GradientBoostingRegressor(n_estimators=5, random_state=0),
        sklearn.ensemble.HistGradientBoostingRegressor(max_depth=3, random_state=0),
    ]
    for model in regressor_models:
        model.fit(X_reg, y_reg)
        cases.append((type(model).__name__, model, X_reg, None, "raw"))

    X_clf, y_clf = shap.datasets.iris()
    y_clf = y_clf.copy()
    y_clf[y_clf == 2] = 1
    classifier_models = [
        DecisionTreeClassifier(max_depth=3, random_state=0),
        RandomForestClassifier(n_estimators=5, random_state=0),
        sklearn.ensemble.ExtraTreesClassifier(n_estimators=5, random_state=0),
        GradientBoostingClassifier(n_estimators=5, random_state=0),
        sklearn.ensemble.HistGradientBoostingClassifier(max_depth=3, random_state=0),
    ]
    for model in classifier_models:
        model.fit(X_clf, y_clf)
        cases.append((type(model).__name__, model, X_clf, None, "raw"))

    for case_name, model, X, data_missing, model_output in cases:
        existing, rebuilt = _roundtrip_tree_ensemble(model, X, data_missing=data_missing, model_output=model_output)
        np.testing.assert_allclose(existing.predict(X), rebuilt.predict(X), err_msg=case_name)

        sample = X.iloc[:10] if isinstance(X, pd.DataFrame) else X[:10]
        existing_explanation = shap.TreeExplainer(existing, data=X, feature_perturbation="interventional")(sample)
        rebuilt_explanation = shap.TreeExplainer(rebuilt, data=X, feature_perturbation="interventional")(sample)
        np.testing.assert_allclose(
            existing_explanation.values, rebuilt_explanation.values, err_msg=case_name, atol=1e-6
        )
        _assert_explanation_additivity(existing_explanation, existing.predict(sample), err_msg=case_name)
        _assert_explanation_additivity(rebuilt_explanation, rebuilt.predict(sample), err_msg=case_name)


def test_from_trees_roundtrip_isolation_forest_models():
    X, _ = shap.datasets.california(n_points=200)
    cases = [
        (
            "IsolationForest",
            sklearn.ensemble.IsolationForest(max_features=1.0, random_state=0).fit(X),
        ),
        (
            "IsolationForest_subsampled_features",
            sklearn.ensemble.IsolationForest(max_features=0.75, random_state=0).fit(X),
        ),
    ]

    for case_name, model in cases:
        existing, rebuilt = _roundtrip_tree_ensemble(model, X, data_missing=None, model_output="raw")
        np.testing.assert_allclose(existing.predict(X), rebuilt.predict(X), err_msg=case_name)

        sample = X.iloc[:10] if isinstance(X, pd.DataFrame) else X[:10]
        existing_explanation = shap.TreeExplainer(existing, feature_perturbation="tree_path_dependent")(sample)
        rebuilt_explanation = shap.TreeExplainer(rebuilt, feature_perturbation="tree_path_dependent")(sample)
        np.testing.assert_allclose(
            existing_explanation.values, rebuilt_explanation.values, err_msg=case_name, atol=1e-6
        )
        np.testing.assert_allclose(
            existing_explanation.base_values,
            rebuilt_explanation.base_values,
            err_msg=case_name,
            atol=1e-6,
        )


def test_from_trees_roundtrip_optional_models():
    xgboost = pytest.importorskip("xgboost")
    ngboost = pytest.importorskip("ngboost")

    X_reg, y_reg = shap.datasets.california(n_points=200)
    X_clf, y_clf = shap.datasets.iris()
    y_clf = y_clf.copy()
    y_clf[y_clf == 2] = 1

    dtrain = xgboost.DMatrix(X_reg, label=y_reg)
    booster = xgboost.train(
        params={"objective": "reg:squarederror", "max_depth": 3, "eta": 0.1},
        dtrain=dtrain,
        num_boost_round=5,
    )

    cases = [
        (
            "XGBRegressor",
            xgboost.XGBRegressor(n_estimators=5, max_depth=3, learning_rate=0.1, random_state=0).fit(X_reg, y_reg),
            X_reg,
            None,
            "raw",
        ),
        (
            "XGBClassifier",
            xgboost.XGBClassifier(n_estimators=5, max_depth=3, learning_rate=0.1, random_state=0).fit(X_clf, y_clf),
            X_clf,
            None,
            "raw",
        ),
        (
            "XGBooster",
            booster,
            X_reg,
            None,
            "raw",
        ),
        (
            "NGBRegressor",
            ngboost.NGBRegressor(n_estimators=5, col_sample=1.0).fit(X_reg, y_reg),
            X_reg,
            None,
            0,
        ),
    ]

    for case_name, model, X, data_missing, model_output in cases:
        existing, rebuilt = _roundtrip_tree_ensemble(model, X, data_missing=data_missing, model_output=model_output)
        np.testing.assert_allclose(existing.predict(X), rebuilt.predict(X), err_msg=case_name)

        sample = X.iloc[:10] if isinstance(X, pd.DataFrame) else X[:10]
        existing_explanation = shap.TreeExplainer(existing, data=X, feature_perturbation="interventional")(sample)
        rebuilt_explanation = shap.TreeExplainer(rebuilt, data=X, feature_perturbation="interventional")(sample)
        np.testing.assert_allclose(
            existing_explanation.values, rebuilt_explanation.values, err_msg=case_name, atol=1e-6
        )
        _assert_explanation_additivity(existing_explanation, existing.predict(sample), err_msg=case_name)
        _assert_explanation_additivity(rebuilt_explanation, rebuilt.predict(sample), err_msg=case_name)


def test_from_trees_roundtrip_smaller_tree_libraries():
    X, y = shap.datasets.california(n_points=200)
    X_clf, y_clf = shap.datasets.iris()
    y_clf = y_clf.copy()
    y_clf[y_clf == 2] = 1

    cases = []

    lightgbm = pytest.importorskip("lightgbm")
    cases.append(
        (
            "lightgbm_regressor",
            lightgbm.LGBMRegressor(n_estimators=5, random_state=0).fit(X, y),
            X,
            None,
            "raw",
        )
    )
    cases.append(
        (
            "lightgbm_classifier",
            lightgbm.LGBMClassifier(n_estimators=5, random_state=0).fit(X_clf, y_clf),
            X_clf,
            None,
            "raw",
        )
    )
    lgb_train = lightgbm.Dataset(X, y)
    cases.append(
        (
            "lightgbm_booster",
            lightgbm.train(
                params={"objective": "regression", "learning_rate": 0.1, "verbose": -1},
                train_set=lgb_train,
                num_boost_round=5,
            ),
            X,
            None,
            "raw",
        )
    )

    catboost = pytest.importorskip("catboost")
    cases.append(
        (
            "catboost_regressor",
            catboost.CatBoostRegressor(iterations=5, depth=3, verbose=False, random_seed=0).fit(X, y),
            X,
            None,
            "raw",
        )
    )
    cases.append(
        (
            "catboost_classifier",
            catboost.CatBoostClassifier(iterations=5, depth=3, verbose=False, random_seed=0).fit(X_clf, y_clf),
            X_clf,
            None,
            "raw",
        )
    )

    skopt = pytest.importorskip("skopt")
    cases.append(
        (
            "skopt_random_forest_regressor",
            skopt.learning.forest.RandomForestRegressor(n_estimators=5, random_state=0).fit(X, y),
            X,
            None,
            "raw",
        )
    )

    imblearn = pytest.importorskip("imblearn")
    cases.append(
        (
            "imbalanced_learn_balanced_random_forest",
            imblearn.ensemble.BalancedRandomForestClassifier(n_estimators=5, random_state=0).fit(X_clf, y_clf),
            X_clf,
            None,
            "raw",
        )
    )

    for case_name, model, X_case, data_missing, model_output in cases:
        existing, rebuilt = _roundtrip_tree_ensemble(
            model, X_case, data_missing=data_missing, model_output=model_output
        )
        np.testing.assert_allclose(existing.predict(X_case), rebuilt.predict(X_case), err_msg=case_name)
        sample = X_case.iloc[:10] if isinstance(X_case, pd.DataFrame) else X_case[:10]
        existing_explanation = shap.TreeExplainer(existing, data=X_case, feature_perturbation="interventional")(sample)
        rebuilt_explanation = shap.TreeExplainer(rebuilt, data=X_case, feature_perturbation="interventional")(sample)
        np.testing.assert_allclose(
            existing_explanation.values, rebuilt_explanation.values, err_msg=case_name, atol=1e-6
        )
        _assert_explanation_additivity(existing_explanation, existing.predict(sample), err_msg=case_name)
        _assert_explanation_additivity(rebuilt_explanation, rebuilt.predict(sample), err_msg=case_name)


def test_from_trees_roundtrip_gpboost():
    gpboost = pytest.importorskip("gpboost")
    X, y = shap.datasets.california(n_points=200)
    data_train = gpboost.Dataset(X, y)
    model = gpboost.train(
        params={"objective": "regression_l2", "learning_rate": 0.1, "verbose": 0},
        train_set=data_train,
        num_boost_round=5,
    )

    existing, rebuilt = _roundtrip_tree_ensemble(model, X, data_missing=None, model_output="raw")
    np.testing.assert_allclose(existing.predict(X), rebuilt.predict(X), err_msg="gpboost")

    sample = X.iloc[:10] if isinstance(X, pd.DataFrame) else X[:10]
    existing_explanation = shap.TreeExplainer(existing, feature_perturbation="tree_path_dependent")(sample)
    rebuilt_explanation = shap.TreeExplainer(rebuilt, feature_perturbation="tree_path_dependent")(sample)
    np.testing.assert_allclose(existing_explanation.values, rebuilt_explanation.values, err_msg="gpboost", atol=1e-6)
    np.testing.assert_allclose(
        existing_explanation.base_values, rebuilt_explanation.base_values, err_msg="gpboost", atol=1e-6
    )


def test_from_trees_roundtrip_pyod_iforest():
    pytest.importorskip("pyod.models.iforest")
    from pyod.models.iforest import IForest

    X, _ = shap.datasets.california(n_points=200)
    X = sklearn.utils.check_array(X)
    model = IForest(max_features=0.75)
    model.fit(X)

    existing, rebuilt = _roundtrip_tree_ensemble(model, X, data_missing=None, model_output="raw")
    np.testing.assert_allclose(existing.predict(X), rebuilt.predict(X), err_msg="pyod_iforest")

    sample = X[:10]
    existing_explanation = shap.TreeExplainer(existing, feature_perturbation="tree_path_dependent")(sample)
    rebuilt_explanation = shap.TreeExplainer(rebuilt, feature_perturbation="tree_path_dependent")(sample)
    np.testing.assert_allclose(
        existing_explanation.values, rebuilt_explanation.values, err_msg="pyod_iforest", atol=1e-6
    )
    np.testing.assert_allclose(
        existing_explanation.base_values, rebuilt_explanation.base_values, err_msg="pyod_iforest", atol=1e-6
    )


@pytest.mark.skipif(sys.platform == "win32", reason="fails due to OOM errors, see #4021")
def test_from_trees_roundtrip_pyspark(configure_pyspark_python):
    pyspark = pytest.importorskip("pyspark")
    pytest.importorskip("pyspark.ml")
    try:
        spark = pyspark.sql.SparkSession.builder.config(
            conf=pyspark.SparkConf().set("spark.master", "local[*]")
        ).getOrCreate()
    except Exception:
        pytest.skip("Could not create pyspark context")

    iris_sk = sklearn.datasets.load_iris()
    iris = pd.DataFrame(data=np.c_[iris_sk["data"], iris_sk["target"]], columns=iris_sk["feature_names"] + ["target"])[
        :100
    ]
    col = ["sepal_length", "sepal_width", "petal_length", "petal_width", "type"]
    iris = spark.createDataFrame(iris, col)
    iris = pyspark.ml.feature.VectorAssembler(inputCols=col[:-1], outputCol="features").transform(iris)
    iris = pyspark.ml.feature.StringIndexer(inputCol="type", outputCol="label").fit(iris).transform(iris)

    classifiers = [
        pyspark.ml.classification.GBTClassifier(labelCol="label", featuresCol="features"),
        pyspark.ml.classification.RandomForestClassifier(labelCol="label", featuresCol="features"),
        pyspark.ml.classification.DecisionTreeClassifier(labelCol="label", featuresCol="features"),
    ]
    X = pd.DataFrame(data=iris_sk.data, columns=iris_sk.feature_names)[:100]
    for classifier in classifiers:
        model = classifier.fit(iris)
        existing, rebuilt = _roundtrip_tree_ensemble(model, X, data_missing=None, model_output="raw")
        np.testing.assert_allclose(existing.predict(X), rebuilt.predict(X), err_msg=str(type(model)))

        sample = X.iloc[:10]
        existing_explanation = shap.TreeExplainer(existing, feature_perturbation="tree_path_dependent")(sample)
        rebuilt_explanation = shap.TreeExplainer(rebuilt, feature_perturbation="tree_path_dependent")(sample)
        np.testing.assert_allclose(existing_explanation.values, rebuilt_explanation.values, err_msg=str(type(model)))
        np.testing.assert_allclose(
            existing_explanation.base_values, rebuilt_explanation.base_values, err_msg=str(type(model))
        )

    regressors = [
        pyspark.ml.regression.GBTRegressor(labelCol="sepal_length", featuresCol="features"),
        pyspark.ml.regression.RandomForestRegressor(labelCol="sepal_length", featuresCol="features"),
        pyspark.ml.regression.DecisionTreeRegressor(labelCol="sepal_length", featuresCol="features"),
    ]
    iris_reg = spark.createDataFrame(
        pd.DataFrame(data=np.c_[iris_sk["data"], iris_sk["target"]], columns=iris_sk["feature_names"] + ["target"])[
            :100
        ],
        col,
    ).drop("type")
    iris_reg = pyspark.ml.feature.VectorAssembler(inputCols=col[1:-1], outputCol="features").transform(iris_reg)
    X_reg = pd.DataFrame(data=iris_sk.data, columns=iris_sk.feature_names).drop("sepal length (cm)", axis=1)[:100]
    for regressor in regressors:
        model = regressor.fit(iris_reg)
        existing, rebuilt = _roundtrip_tree_ensemble(model, X_reg, data_missing=None, model_output="raw")
        np.testing.assert_allclose(existing.predict(X_reg), rebuilt.predict(X_reg), err_msg=str(type(model)))

        sample = X_reg.iloc[:10]
        existing_explanation = shap.TreeExplainer(existing, feature_perturbation="tree_path_dependent")(sample)
        rebuilt_explanation = shap.TreeExplainer(rebuilt, feature_perturbation="tree_path_dependent")(sample)
        np.testing.assert_allclose(existing_explanation.values, rebuilt_explanation.values, err_msg=str(type(model)))
        np.testing.assert_allclose(
            existing_explanation.base_values, rebuilt_explanation.base_values, err_msg=str(type(model))
        )

    spark.stop()

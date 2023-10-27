"""Test tree functions."""
import itertools
import math
import pickle
import sys

import numpy as np
import pandas as pd
import pytest
import sklearn
import sklearn.pipeline
from sklearn.utils import check_array

import shap
from shap.explainers._tree import SingleTree
from shap.utils._exceptions import InvalidModelError


def test_unsupported_model_raises_error():
    """Unsupported model inputs to TreeExplainer should raise an Exception."""

    class CustomEstimator:
        ...

    emsg = "Model type not yet supported by TreeExplainer:"
    with pytest.raises(InvalidModelError, match=emsg):
        _ = shap.TreeExplainer(CustomEstimator())


def test_front_page_xgboost():
    xgboost = pytest.importorskip("xgboost")

    # load JS visualization code to notebook
    shap.initjs()

    # train XGBoost model
    X, y = shap.datasets.california(n_points=500)
    model = xgboost.train({"learning_rate": 0.01, "silent": 1}, xgboost.DMatrix(X, label=y), 100)

    # explain the model's predictions using SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # visualize the first prediction's explanation
    shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :])

    # visualize the training set predictions
    shap.force_plot(explainer.expected_value, shap_values, X)

    # create a SHAP dependence plot to show the effect of a single feature across the whole dataset
    shap.dependence_plot(5, shap_values, X, show=False)
    shap.dependence_plot("Longitude", shap_values, X, show=False)

    # summarize the effects of all the features
    shap.summary_plot(shap_values, X, show=False)


def test_front_page_sklearn():
    # load JS visualization code to notebook
    shap.initjs()

    # train model
    X, y = shap.datasets.california(n_points=500)
    models = [
        sklearn.ensemble.RandomForestRegressor(n_estimators=10),
        sklearn.ensemble.ExtraTreesRegressor(n_estimators=10),
    ]
    for model in models:
        model.fit(X, y)

        # explain the model's predictions using SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # visualize the first prediction's explanation
        shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :])

        # visualize the training set predictions
        shap.force_plot(explainer.expected_value, shap_values, X)

        # create a SHAP dependence plot to show the effect of a single feature across the whole
        # dataset
        shap.dependence_plot(5, shap_values, X, show=False)
        shap.dependence_plot("Longitude", shap_values, X, show=False)

        # summarize the effects of all the features
        shap.summary_plot(shap_values, X, show=False)


def _conditional_expectation(tree, S, x):
    tree_ind = 0

    def R(node_ind):

        f = tree.features[tree_ind, node_ind]
        lc = tree.children_left[tree_ind, node_ind]
        rc = tree.children_right[tree_ind, node_ind]
        if lc < 0:
            return tree.values[tree_ind, node_ind]
        if f in S:
            if x[f] <= tree.thresholds[tree_ind, node_ind]:
                return R(lc)
            return R(rc)
        lw = tree.node_sample_weight[tree_ind, lc]
        rw = tree.node_sample_weight[tree_ind, rc]
        return (R(lc) * lw + R(rc) * rw) / (lw + rw)

    out = 0.0
    l = tree.values.shape[0] if tree.tree_limit is None else tree.tree_limit
    for i in range(l):
        tree_ind = i
        out += R(0)
    return out


def _brute_force_tree_shap(tree, x):
    m = len(x)
    phi = np.zeros(m)
    for p in itertools.permutations(list(range(m))):
        for i in range(m):
            phi[p[i]] += _conditional_expectation(tree, p[:i + 1], x) - _conditional_expectation(
                tree, p[:i], x)
    return phi / math.factorial(m)


def _validate_shap_values(model, x_test):
    # explain the model's predictions using SHAP values
    tree_explainer = shap.TreeExplainer(model)

    explanation = tree_explainer(x_test)
    # check the properties of Explanation object
    assert explanation.values.shape == (*x_test.shape,)
    assert explanation.base_values.shape == (x_test.shape[0],)

    # validate values sum to the margin prediction of the model plus expected_value
    assert np.allclose(
        explanation.values.sum(1) + explanation.base_values,
        model.predict(x_test),
    )


@pytest.mark.parametrize("col_sample", [1.0, 0.9])
def test_ngboost_models_prediction_equal(col_sample):
    from shap.explainers._tree import TreeEnsemble

    ngboost = pytest.importorskip("ngboost")
    X, y = shap.datasets.california(n_points=500)

    model = ngboost.NGBRegressor(n_estimators=2, col_sample=col_sample).fit(X, y)

    tree_ensemble = TreeEnsemble(model=model,
                                data=X,
                                data_missing=None,
                                model_output=0,
                                )
    y_pred = model.predict(X)
    y_pred_tree_ensemble = tree_ensemble.predict(X)
    assert (y_pred == y_pred_tree_ensemble).all()


@pytest.mark.parametrize("col_sample", [1.0, 0.9])
def test_ngboost_sum_of_shap_values(col_sample):
    ngboost = pytest.importorskip("ngboost")

    X, y = shap.datasets.california(n_points=500)
    model = ngboost.NGBRegressor(n_estimators=20, col_sample=col_sample).fit(X, y)
    predicted = model.predict(X)

    # explain the model's predictions using SHAP values
    explainer = shap.TreeExplainer(model, model_output=0)

    explanation = explainer(X)
    # check the properties of Explanation object
    assert explanation.values.shape == (*X.shape,)
    assert explanation.base_values.shape == (len(X),)

    # check that SHAP values sum to model output
    assert (
        np.abs(explanation.values.sum(1) + explanation.base_values - predicted).max()
        < 1e-5
    )


@pytest.fixture
def configure_pyspark_python(monkeypatch):
    monkeypatch.setenv("PYSPARK_PYTHON", sys.executable)
    monkeypatch.setenv("PYSPARK_DRIVER_PYTHON", sys.executable)


def test_pyspark_classifier_decision_tree(configure_pyspark_python):
    # pylint: disable=bare-except
    pyspark = pytest.importorskip("pyspark")
    pytest.importorskip("pyspark.ml")
    try:
        spark = pyspark.sql.SparkSession.builder.config(
            conf=pyspark.SparkConf().set("spark.master", "local[*]")).getOrCreate()
    except Exception:
        pytest.skip("Could not create pyspark context")

    iris_sk = sklearn.datasets.load_iris()
    iris = pd.DataFrame(data=np.c_[iris_sk['data'], iris_sk['target']],
                        columns=iris_sk['feature_names'] + ['target'])[:100]
    col = ["sepal_length", "sepal_width", "petal_length", "petal_width", "type"]
    iris = spark.createDataFrame(iris, col)
    iris = pyspark.ml.feature.VectorAssembler(inputCols=col[:-1], outputCol="features").transform(
        iris)
    iris = pyspark.ml.feature.StringIndexer(inputCol="type", outputCol="label").fit(iris).transform(
        iris)

    classifiers = [
        pyspark.ml.classification.GBTClassifier(labelCol="label", featuresCol="features"),
        pyspark.ml.classification.RandomForestClassifier(labelCol="label", featuresCol="features"),
        pyspark.ml.classification.DecisionTreeClassifier(labelCol="label", featuresCol="features")]
    for classifier in classifiers:
        model = classifier.fit(iris)
        explainer = shap.TreeExplainer(model)
        # Make sure the model can be serializable to run shap values with spark
        pickle.dumps(explainer)
        X = pd.DataFrame(data=iris_sk.data, columns=iris_sk.feature_names)[  # pylint: disable=E1101
            :100]

        shap_values = explainer.shap_values(X, check_additivity=False)
        expected_values = explainer.expected_value

        predictions = model.transform(iris).select("rawPrediction").rdd.map(
            lambda x: [float(y) for y in x['rawPrediction']]).toDF(['class0', 'class1']).toPandas()

        if str(type(model)).endswith("GBTClassificationModel'>"):
            diffs = expected_values + shap_values.sum(1) - predictions.class1
            assert np.max(np.abs(diffs)) < 1e-4, "SHAP values don't sum to model output for class0!"
        else:
            normalizedPredictions = (predictions.T / predictions.sum(1)).T
            diffs = expected_values[0] + shap_values[0].sum(1) - normalizedPredictions.class0
            assert np.max(
                np.abs(diffs)) < 1e-4, "SHAP values don't sum to model output for class0!" + model
            diffs = expected_values[1] + shap_values[1].sum(1) - normalizedPredictions.class1
            assert np.max(
                np.abs(diffs)) < 1e-4, "SHAP values don't sum to model output for class1!" + model
            assert (np.abs(
                expected_values - normalizedPredictions.mean()) < 1e-1).all(), \
                "Bad expected_value!" + model
    spark.stop()


def test_pyspark_regression_decision_tree(configure_pyspark_python):
    # pylint: disable=bare-except
    pyspark = pytest.importorskip("pyspark")
    pytest.importorskip("pyspark.ml")
    try:
        spark = pyspark.sql.SparkSession.builder.config(
            conf=pyspark.SparkConf().set("spark.master", "local[*]")).getOrCreate()
    except Exception:
        pytest.skip("Could not create pyspark context")

    iris_sk = sklearn.datasets.load_iris()
    iris = pd.DataFrame(data=np.c_[iris_sk['data'], iris_sk['target']],
                        columns=iris_sk['feature_names'] + ['target'])[:100]

    # Simple regressor: try to predict sepal length based on the other features
    col = ["sepal_length", "sepal_width", "petal_length", "petal_width", "type"]
    iris = spark.createDataFrame(iris, col).drop("type")
    iris = pyspark.ml.feature.VectorAssembler(inputCols=col[1:-1], outputCol="features").transform(
        iris)

    regressors = [
        pyspark.ml.regression.GBTRegressor(labelCol="sepal_length", featuresCol="features"),
        pyspark.ml.regression.RandomForestRegressor(labelCol="sepal_length", featuresCol="features"),
        pyspark.ml.regression.DecisionTreeRegressor(labelCol="sepal_length", featuresCol="features")
    ]
    for regressor in regressors:
        model = regressor.fit(iris)
        explainer = shap.TreeExplainer(model)
        X = pd.DataFrame(data=iris_sk.data, columns=iris_sk.feature_names).drop('sepal length (cm)', axis=1)[:100] # pylint: disable=E1101

        shap_values = explainer.shap_values(X, check_additivity=False)
        expected_values = explainer.expected_value

        # validate values sum to the margin prediction of the model plus expected_value
        predictions = model.transform(iris).select("prediction").toPandas()
        diffs = expected_values + shap_values.sum(1) - predictions["prediction"]
        assert np.max(np.abs(diffs)) < 1e-4, "SHAP values don't sum to model output for class0!"
        assert (np.abs(expected_values - predictions.mean()) < 1e-1).all(), "Bad expected_value!"
    spark.stop()


def create_binary_newsgroups_data():
    categories = ['alt.atheism', 'soc.religion.christian']
    newsgroups_train = sklearn.datasets.fetch_20newsgroups(subset='train', categories=categories)
    newsgroups_test = sklearn.datasets.fetch_20newsgroups(subset='test', categories=categories)
    class_names = ['atheism', 'christian']
    return newsgroups_train, newsgroups_test, class_names


def test_gpboost():
    gpboost = pytest.importorskip("gpboost")
    # train gpboost model
    X, y = shap.datasets.california(n_points=500)
    data_train = gpboost.Dataset(X, y)
    model = gpboost.train(
        params={"objective": "regression_l2", "learning_rate": 0.1, "verbose": 0},
        train_set=data_train,
        num_boost_round=10,
    )
    predicted = model.predict(X, pred_latent=True)

    # explain the model's predictions using SHAP values
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")

    explanation = explainer(X)
    # check the properties of Explanation object
    assert explanation.values.shape == (*X.shape,)
    assert explanation.base_values.shape == (len(X),)

    # check that SHAP values sum to model output
    assert (
        np.abs(explanation.values.sum(1) + explanation.base_values - predicted).max()
        < 1e-4
    )


def test_catboost():
    catboost = pytest.importorskip("catboost")
    # train catboost model
    X, y = shap.datasets.california(n_points=500)
    X["IsOld"] = (X["HouseAge"] > 30).astype(str)
    model = catboost.CatBoostRegressor(iterations=30, learning_rate=0.1, random_seed=123)
    p = catboost.Pool(X, y, cat_features=["IsOld"])
    model.fit(p, verbose=False, plot=False)
    predicted = model.predict(X)

    # explain the model's predictions using SHAP values
    explainer = shap.TreeExplainer(model)

    explanation = explainer(X)
    # check the properties of Explanation object
    assert explanation.values.shape == (*X.shape,)
    assert explanation.base_values.shape == (len(X),)

    # check that SHAP values sum to model output
    assert (
        np.abs(explanation.values.sum(1) + explanation.base_values - predicted).max()
        < 1e-4
    )

    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    model = catboost.CatBoostClassifier(iterations=10, learning_rate=0.5, random_seed=12)
    model.fit(X, y, verbose=False, plot=False)
    predicted = model.predict(X, prediction_type="RawFormulaVal")

    # explain the model's predictions using SHAP values
    explainer = shap.TreeExplainer(model)

    explanation = explainer(X)
    # check the properties of Explanation object
    assert explanation.values.shape == (*X.shape,)
    assert explanation.base_values.shape == (len(X),)

    # check that SHAP values sum to model output
    assert (
        np.abs(explanation.values.sum(1) + explanation.base_values - predicted).max()
        < 1e-4
    )


def test_catboost_categorical():
    catboost = pytest.importorskip("catboost")
    X, y = shap.datasets.california(n_points=500)
    X["IsOld"] = (X["HouseAge"] > 30).astype(str)

    model = catboost.CatBoostRegressor(100, cat_features=['IsOld'], verbose=False)
    model.fit(X, y)
    predicted = model.predict(X)

    # explain the model's predictions using SHAP values
    explainer = shap.TreeExplainer(model)

    explanation = explainer(X)
    # check the properties of Explanation object
    assert explanation.values.shape == (*X.shape,)
    assert explanation.base_values.shape == (len(X),)

    # check that SHAP values sum to model output
    assert (
        np.abs(explanation.values.sum(1) + explanation.base_values - predicted).max()
        < 1e-4
    )


def test_catboost_interactions():
    # GH 3324
    catboost = pytest.importorskip("catboost")

    X, y = shap.datasets.adult(n_points=50)

    model = catboost.CatBoostClassifier(depth=1, iterations=10).fit(X, y)
    predicted = model.predict(X, prediction_type="RawFormulaVal")

    ex_cat = shap.TreeExplainer(model)

    # catboost explanations
    explanation = ex_cat(X, interactions=True)
    assert (
        np.abs(explanation.values.sum(axis=(1, 2)) + explanation.base_values - predicted).max()
        < 1e-4
    )

# TODO: Test tree_limit argument


def _average_path_length(n_samples_leaf):
    """
    Vendored from: https://github.com/scikit-learn/scikit-learn/blob/399131c8545cd525724e4bacf553416c512ac82c/sklearn/ensemble/_iforest.py#L531

    For use in isolation forest tests.
    """
    n_samples_leaf = check_array(n_samples_leaf, ensure_2d=False)

    n_samples_leaf_shape = n_samples_leaf.shape
    n_samples_leaf = n_samples_leaf.reshape((1, -1))
    average_path_length = np.zeros(n_samples_leaf.shape)

    mask_1 = n_samples_leaf <= 1
    mask_2 = n_samples_leaf == 2
    not_mask = ~np.logical_or(mask_1, mask_2)

    average_path_length[mask_1] = 0.0
    average_path_length[mask_2] = 1.0
    average_path_length[not_mask] = (
        2.0 * (np.log(n_samples_leaf[not_mask] - 1.0) + np.euler_gamma)
        - 2.0 * (n_samples_leaf[not_mask] - 1.0) / n_samples_leaf[not_mask]
    )

    return average_path_length.reshape(n_samples_leaf_shape)


def test_isolation_forest():
    from sklearn.ensemble import IsolationForest

    X, _ = shap.datasets.california(n_points=500)
    for max_features in [1.0, 0.75]:
        iso = IsolationForest(max_features=max_features)
        iso.fit(X)

        explainer = shap.TreeExplainer(iso)

        explanation = explainer(X)
        # check the properties of Explanation object
        assert explanation.values.shape == (*X.shape,)
        assert explanation.base_values.shape == (len(X),)

        path_length = _average_path_length(np.array([iso.max_samples_]))[0]
        score_from_shap = -2 ** (-(explanation.values.sum(1) + explanation.base_values) / path_length)
        assert np.allclose(iso.score_samples(X), score_from_shap, atol=1e-7)


def test_pyod_isolation_forest():
    pytest.importorskip("pyod.models.iforest")
    from pyod.models.iforest import IForest

    X, _ = shap.datasets.california(n_points=500)
    for max_features in [1.0, 0.75]:
        iso = IForest(max_features=max_features)
        iso.fit(X)

        explainer = shap.TreeExplainer(iso)

        explanation = explainer(X)
        # check the properties of Explanation object
        assert explanation.values.shape == (*X.shape,)
        assert explanation.base_values.shape == (len(X),)

        path_length = _average_path_length(np.array([iso.max_samples_]))[0]
        score_from_shap = -2 ** (-(explanation.values.sum(1) + explanation.base_values) / path_length)
        assert np.allclose(iso.detector_.score_samples(X), score_from_shap, atol=1e-7)


def test_provided_background_tree_path_dependent():
    """Tests xgboost explainer when feature_perturbation is tree_path_dependent and when background
    data is provided.
    """
    xgboost = pytest.importorskip("xgboost")

    X, y = shap.datasets.adult(n_points=100)
    dtrain = xgboost.DMatrix(X, label=y, feature_names=list(X.columns))

    params = {
        "booster": "gbtree",
        "objective": "binary:logistic",
        "max_depth": 2,
        "eta": 0.05,
        "nthread": -1,
        "random_state": 42,
    }
    bst = xgboost.train(params=params, dtrain=dtrain, num_boost_round=10)
    pred_scores = bst.predict(dtrain, output_margin=True)

    explainer = shap.TreeExplainer(bst, data=X, feature_perturbation="tree_path_dependent")
    diffs = explainer.expected_value + explainer.shap_values(X).sum(axis=1) - pred_scores
    assert np.max(np.abs(diffs)) < 1e-4, "SHAP values don't sum to model output!"
    assert np.abs(explainer.expected_value - pred_scores.mean()) < 1e-6, "Bad expected_value!"


def test_provided_background_independent():
    xgboost = pytest.importorskip("xgboost")

    X, y = shap.datasets.iris()
    # Select the first 100 rows, so that the y values contain only 0s and 1s
    X = X[:100]
    y = y[:100]
    train_x, test_x, train_y, _ = sklearn.model_selection.train_test_split(X, y, random_state=1)
    feature_names = ["a", "b", "c", "d"]
    dtrain = xgboost.DMatrix(train_x, label=train_y, feature_names=feature_names)
    dtest = xgboost.DMatrix(test_x, feature_names=feature_names)

    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'max_depth': 4,
        'eta': 0.1,
        'nthread': -1,
    }

    bst = xgboost.train(params=params, dtrain=dtrain, num_boost_round=100)

    explainer = shap.TreeExplainer(bst, test_x, feature_perturbation="interventional")
    diffs = explainer.expected_value + \
            explainer.shap_values(test_x).sum(1) - bst.predict(dtest, output_margin=True)
    assert np.max(np.abs(diffs)) < 1e-4, "SHAP values don't sum to model output!"
    assert np.abs(explainer.expected_value - bst.predict(dtest,
                                                         output_margin=True).mean()) < 1e-4, \
        "Bad expected_value!"


def test_provided_background_independent_prob_output():
    xgboost = pytest.importorskip("xgboost")

    X, y = shap.datasets.iris()
    # Select the first 100 rows, so that the y values contain only 0s and 1s
    X = X[:100]
    y = y[:100]
    train_x, test_x, train_y, _ = sklearn.model_selection.train_test_split(X, y, random_state=1)
    feature_names = ["a", "b", "c", "d"]
    dtrain = xgboost.DMatrix(train_x, label=train_y, feature_names=feature_names)
    dtest = xgboost.DMatrix(test_x, feature_names=feature_names)

    for objective in ["reg:logistic", "binary:logistic"]:
        params = {
            'booster': 'gbtree',
            'objective': objective,
            'max_depth': 4,
            'eta': 0.1,
            'nthread': -1,
        }

        bst = xgboost.train(params=params, dtrain=dtrain, num_boost_round=100)

        explainer = shap.TreeExplainer(bst, test_x, feature_perturbation="interventional",
                                       model_output="probability")
        diffs = explainer.expected_value + explainer.shap_values(test_x).sum(1) - bst.predict(dtest)
        assert np.max(np.abs(diffs)) < 1e-4, "SHAP values don't sum to model output!"
        assert np.abs(
            explainer.expected_value - bst.predict(dtest).mean()) < 1e-4, "Bad expected_value!"


def test_single_tree_compare_with_kernel_shap():
    """ Compare with Kernel SHAP, which makes the same independence assumptions
    as Independent Tree SHAP.  Namely, they both assume independence between the
    set being conditioned on, and the remainder set.
    """
    xgboost = pytest.importorskip("xgboost")

    # FIXME: this test should ideally pass with any random seed. See #2960
    random_seed = 0
    rs = np.random.RandomState(random_seed)

    n = 100
    X = rs.normal(size=(n, 7))
    y = np.matmul(X, [-2, 1, 3, 5, 2, 20, -5])

    # train a model with single tree
    Xd = xgboost.DMatrix(X, label=y)
    model = xgboost.train({'eta': 1,
                           'max_depth': 6,
                           'base_score': 0,
                           "lambda": 0},
                          Xd, 1)
    ypred = model.predict(Xd)

    # Compare for five random samples
    for _ in range(5):
        x_ind = rs.choice(X.shape[1])
        x = X[x_ind:x_ind + 1, :]

        expl = shap.TreeExplainer(model, X, feature_perturbation="interventional")
        def f(inp):
            return model.predict(xgboost.DMatrix(inp))
        expl_kern = shap.KernelExplainer(f, X)

        itshap = expl.shap_values(x)
        kshap = expl_kern.shap_values(x, nsamples=150)
        assert np.allclose(itshap, kshap), \
            "Kernel SHAP doesn't match Independent Tree SHAP!"
        assert np.allclose(itshap.sum() + expl.expected_value, ypred[x_ind]), \
            "SHAP values don't sum to model output!"


def test_several_trees():
    """ Make sure Independent Tree SHAP sums up to the correct value for
    larger models (20 trees).
    """
    # FIXME: this test should ideally pass with any random seed. See #2960
    random_seed = 0
    xgboost = pytest.importorskip("xgboost")
    rs = np.random.RandomState(random_seed)

    n = 1000
    X = rs.normal(size=(n, 7))
    b = np.array([-2, 1, 3, 5, 2, 20, -5])
    y = np.matmul(X, b)
    max_depth = 6

    # train a model with single tree
    Xd = xgboost.DMatrix(X, label=y)
    model = xgboost.train({'eta': 1,
                           'max_depth': max_depth,
                           'base_score': 0,
                           "lambda": 0},
                          Xd, 20)
    ypred = model.predict(Xd)

    # Compare for five random samples
    for _ in range(5):
        x_ind = rs.choice(X.shape[1])
        x = X[x_ind:x_ind + 1, :]
        expl = shap.TreeExplainer(model, X, feature_perturbation="interventional")
        itshap = expl.shap_values(x)
        assert np.allclose(itshap.sum() + expl.expected_value, ypred[x_ind]), \
            "SHAP values don't sum to model output!"


def test_single_tree_nonlinear_transformations():
    """ Make sure Independent Tree SHAP single trees with non-linear
    transformations.
    """
    # Supported non-linear transforms
    # def sigmoid(x):
    #     return(1/(1+np.exp(-x)))

    # def log_loss(yt,yp):
    #     return(-(yt*np.log(yp) + (1 - yt)*np.log(1 - yp)))

    # def mse(yt,yp):
    #     return(np.square(yt-yp))

    # FIXME: this test should ideally pass with any random seed. See #2960
    random_seed = 0

    xgboost = pytest.importorskip("xgboost")
    rs = np.random.RandomState(random_seed)

    n = 100
    X = rs.normal(size=(n, 7))
    y = np.matmul(X, [-2, 1, 3, 5, 2, 20, -5])
    y = y + abs(min(y))
    y = rs.binomial(n=1, p=y / max(y))

    # train a model with single tree
    Xd = xgboost.DMatrix(X, label=y)
    model = xgboost.train({'eta': 1,
                           'max_depth': 6,
                           'base_score': y.mean(),
                           "lambda": 0,
                           "objective": "binary:logistic"},
                          Xd, 1)
    pred = model.predict(Xd, output_margin=True)  # In margin space (log odds)
    trans_pred = model.predict(Xd)  # In probability space

    expl = shap.TreeExplainer(model, X, feature_perturbation="interventional")
    def f(inp):
        return model.predict(xgboost.DMatrix(inp), output_margin=True)
    expl_kern = shap.KernelExplainer(f, X)

    x_ind = 0
    x = X[x_ind:x_ind + 1, :]
    itshap = expl.shap_values(x)
    kshap = expl_kern.shap_values(x, nsamples=300)
    assert np.allclose(itshap.sum() + expl.expected_value, pred[x_ind]), \
        "SHAP values don't sum to model output on explaining margin!"
    assert np.allclose(itshap, kshap), \
        "Independent Tree SHAP doesn't match Kernel SHAP on explaining margin!"

    model.set_attr(objective="binary:logistic")
    expl = shap.TreeExplainer(model, X, feature_perturbation="interventional",
                              model_output="probability")
    itshap = expl.shap_values(x)
    assert np.allclose(itshap.sum() + expl.expected_value, trans_pred[x_ind]), \
        "SHAP values don't sum to model output on explaining logistic!"

    # expl = shap.TreeExplainer(model, X, feature_perturbation="interventional",
    # model_output="logloss")
    # itshap = expl.shap_values(x,y=y[x_ind])
    # margin_pred = model.predict(xgb.DMatrix(x),output_margin=True)
    # currpred = log_loss(y[x_ind],sigmoid(margin_pred))
    # assert np.allclose(itshap.sum(), currpred - expl.expected_value), \
    # "SHAP values don't sum to model output on explaining logloss!"


def test_skopt_rf_et():
    skopt = pytest.importorskip("skopt")

    # Define an objective function for skopt to optimise.
    def objective_function(x):
        return x[0] ** 2 - x[1] ** 2 + x[1] * x[0]

    # Uneven bounds to prevent "objective has been evaluated" warnings.
    problem_bounds = [(-1e6, 3e6), (-1e6, 3e6)]

    # Don't worry about "objective has been evaluated" warnings.
    result_et = skopt.forest_minimize(objective_function, problem_bounds, n_calls=100,
                                      base_estimator="ET")
    result_rf = skopt.forest_minimize(objective_function, problem_bounds, n_calls=100,
                                      base_estimator="RF")

    et_df = pd.DataFrame(result_et.x_iters, columns=["X0", "X1"])

    # Explain the model's predictions.
    explainer_et = shap.TreeExplainer(result_et.models[-1], et_df)
    shap_values_et = explainer_et.shap_values(et_df)

    rf_df = pd.DataFrame(result_rf.x_iters, columns=["X0", "X1"])

    # Explain the model's predictions (Random forest).
    explainer_rf = shap.TreeExplainer(result_rf.models[-1], rf_df)
    shap_values_rf = explainer_rf.shap_values(rf_df)

    assert np.allclose(shap_values_et.sum(1) + explainer_et.expected_value,
                       result_et.models[-1].predict(et_df))
    assert np.allclose(shap_values_rf.sum(1) + explainer_rf.expected_value,
                       result_rf.models[-1].predict(rf_df))


class TestSingleTree:
    """Tests for the SingleTree class."""

    def test_singletree_lightgbm_basic(self):
        """A basic test for checking that a LightGBM `dump_model()["tree_info"]`
        dictionary is parsed properly into a `SingleTree` object.
        """

        # Stump (only root node) tree
        sample_tree = {
            "tree_index": 256,
            "num_leaves": 1,
            "num_cat": 0,
            "shrinkage": 1,
            "tree_structure": {
                "leaf_value": 0,
                # "leaf_count": 123,  # FIXME(upstream): microsoft/LightGBM#5962
            },
        }
        stree = SingleTree(sample_tree)
        # just ensure that this does not error out
        assert stree.children_left[0] == -1
        # assert stree.node_sample_weight[0] == 123
        assert hasattr(stree, "values")

        # Depth=1 tree
        sample_tree = {
            "tree_index": 0,
            "num_leaves": 2,
            "num_cat": 0,
            "shrinkage": 0.1,
            "tree_structure": {
                "split_index": 0,
                "split_feature": 1,
                "split_gain": 0.001471,
                "threshold": 0,
                "decision_type": "<=",
                "default_left": True,
                "missing_type": "None",
                "internal_value": 0,
                "internal_weight": 0,
                "internal_count": 100,
                "left_child": {"leaf_index": 0, "leaf_value": 0.0667, "leaf_weight": 0.00157, "leaf_count": 33},
                "right_child": {"leaf_index": 1, "leaf_value": -0.0667, "leaf_weight": 0.00175, "leaf_count": 67},
            },
        }

        stree = SingleTree(sample_tree)
        # just ensure that the tree is parsed correctly
        assert stree.node_sample_weight[0] == 100
        assert hasattr(stree, "values")


class TestExplainerSklearn:
    """Tests for the TreeExplainer when the model passed in from scikit-learn (core).

    Included models:
        * tree.DecisionTreeClassifier
        * ensemble.RandomForestClassifier
        * ensemble.RandomForestRegressor
        * ensemble.ExtraTreesRegressor
        * ensemble.GradientBoostingClassifier
        * ensemble.GradientBoostingRegressor
        * ensemble.HistGradientBoostingClassifier
        * ensemble.HistGradientBoostingRegressor
    """

    def test_sklearn_decision_tree_multiclass(self):
        X, y = shap.datasets.iris()
        y[y == 2] = 1
        model = sklearn.tree.DecisionTreeClassifier(
            max_depth=None, min_samples_split=2, random_state=0
        )
        model.fit(X, y)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        assert np.abs(shap_values[0][0, 0] - 0.05) < 1e-1
        assert np.abs(shap_values[1][0, 0] + 0.05) < 1e-1

    def test_sum_match_random_forest_classifier(self):
        X_train, X_test, Y_train, _ = sklearn.model_selection.train_test_split(
            *shap.datasets.adult(), test_size=0.2, random_state=0
        )
        clf = sklearn.ensemble.RandomForestClassifier(
            random_state=202, n_estimators=10, max_depth=10
        )
        clf.fit(X_train, Y_train)
        predicted = clf.predict_proba(X_test)
        explainer = shap.TreeExplainer(clf)

        explanation = explainer(X_test)
        # check the properties of Explanation object
        num_classes = 2
        assert explanation.values.shape == (*X_test.shape, num_classes)
        assert explanation.base_values.shape == (len(X_test), num_classes)

        # check that SHAP values sum to model output
        class0_exp = explanation[..., 0]
        assert (
            np.abs(class0_exp.values.sum(1) + class0_exp.base_values - predicted[:, 0]).max()
            < 1e-4
        )

    def test_sklearn_random_forest_multiclass(self):
        X, y = shap.datasets.iris()
        y[y == 2] = 1
        model = sklearn.ensemble.RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            random_state=0,
        )
        model.fit(X, y)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        assert np.abs(shap_values[0][0, 0] - 0.05) < 1e-3
        assert np.abs(shap_values[1][0, 0] + 0.05) < 1e-3


    def test_sklearn_interaction_values(self):
        X, _ = shap.datasets.iris()
        X_train, _, Y_train, _ = sklearn.model_selection.train_test_split(
            *shap.datasets.iris(), test_size=0.2, random_state=0
        )
        rforest = sklearn.ensemble.RandomForestClassifier(
            n_estimators=10,
            max_depth=None,
            min_samples_split=2,
            random_state=0,
        )
        model = rforest.fit(X_train, Y_train)

        # verify symmetry of the interaction values (this typically breaks if anything is wrong)
        interaction_vals = shap.TreeExplainer(model).shap_interaction_values(X)
        for int_vals in interaction_vals:
            assert np.allclose(int_vals, np.swapaxes(int_vals, 1, 2))

        # ensure the interaction plot works
        shap.summary_plot(interaction_vals[0], X, show=False)

        # text interaction call from TreeExplainer
        X, y = shap.datasets.adult(n_points=50)

        rfc = sklearn.ensemble.RandomForestClassifier(max_depth=1).fit(X, y)
        predicted = rfc.predict_proba(X)
        ex_rfc = shap.TreeExplainer(rfc)
        explanation = ex_rfc(X, interactions=True)
        assert np.allclose(explanation.values[1].sum(axis=(1, 2)) + explanation.base_values[:, 1],
                           predicted[:, 1]
                           )
        assert np.allclose(explanation.values[0].sum(axis=(1, 2)) + explanation.base_values[:, 0],
                           predicted[:, 0]
                           )

    def _create_vectorizer_for_randomforestclassifier(self):
        """Helper setup function"""
        vectorizer = sklearn.feature_extraction.text.CountVectorizer(
            lowercase=False, min_df=0.0, binary=True
        )

        class DenseTransformer(sklearn.base.TransformerMixin):
            def fit(self, X, y=None, **fit_params):
                return self

            def transform(self, X, y=None, **fit_params):
                return X.toarray()

        rf = sklearn.ensemble.RandomForestClassifier(n_estimators=10, random_state=777)
        return sklearn.pipeline.Pipeline(
            [('vectorizer', vectorizer), ('to_dense', DenseTransformer()), ('rf', rf)]
        )

    def test_sklearn_random_forest_newsgroups(self):
        """
        note: this test used to fail in native TreeExplainer code due to memory corruption
        """
        newsgroups_train, newsgroups_test, _ = create_binary_newsgroups_data()
        pipeline = self._create_vectorizer_for_randomforestclassifier()
        pipeline.fit(newsgroups_train.data, newsgroups_train.target)
        rf = pipeline.named_steps['rf']
        vectorizer = pipeline.named_steps["vectorizer"]
        densifier = pipeline.named_steps["to_dense"]

        dense_bg = densifier.transform(vectorizer.transform(newsgroups_test.data[0:20]))

        test_row = newsgroups_test.data[83:84]
        explainer = shap.TreeExplainer(rf, dense_bg, feature_perturbation="interventional")
        vec_row = vectorizer.transform(test_row)
        dense_row = densifier.transform(vec_row)
        explainer.shap_values(dense_row)

    def test_multi_target_random_forest_regressor(self):
        X_train, X_test, Y_train, _ = sklearn.model_selection.train_test_split(
            *shap.datasets.linnerud(),
            test_size=0.2,
            random_state=0,
        )
        est = sklearn.ensemble.RandomForestRegressor(
            random_state=202, n_estimators=10, max_depth=10
        )
        est.fit(X_train, Y_train)
        predicted = est.predict(X_test)

        explainer = shap.TreeExplainer(est)
        expected_values = np.asarray(explainer.expected_value)
        assert (
            len(expected_values) == est.n_outputs_
        ), "Length of expected_values doesn't match n_outputs_"

        explanation = explainer(X_test)
        # check the properties of Explanation object
        assert explanation.values.shape == (*X_test.shape, est.n_outputs_)
        assert explanation.base_values.shape == (len(X_test), est.n_outputs_)

        # check that SHAP values sum to model output for all multioutputs
        assert (
            np.abs(explanation.values.sum(1) + explanation.base_values - predicted).max()
            < 1e-4
        )

    def test_sum_match_extra_trees(self):
        X_train, X_test, Y_train, _ = sklearn.model_selection.train_test_split(
            *shap.datasets.adult(), test_size=0.2, random_state=0
        )
        clf = sklearn.ensemble.ExtraTreesRegressor(
            random_state=202, n_estimators=10, max_depth=10
        )
        clf.fit(X_train, Y_train)
        predicted = clf.predict(X_test)
        ex = shap.TreeExplainer(clf)
        shap_values = ex.shap_values(X_test)

        # check that SHAP values sum to model output
        assert np.abs(shap_values.sum(1) + ex.expected_value - predicted).max() < 1e-4

    # TODO: this has sometimes failed with strange answers, should run memcheck on this for any
    #  memory issues at some point...
    def test_multi_target_extra_trees(self):
        X_train, X_test, Y_train, _ = sklearn.model_selection.train_test_split(
            *shap.datasets.linnerud(),
            test_size=0.2,
            random_state=0,
        )
        est = sklearn.ensemble.ExtraTreesRegressor(
            random_state=202, n_estimators=10, max_depth=10
        )
        est.fit(X_train, Y_train)
        predicted = est.predict(X_test)

        explainer = shap.TreeExplainer(est)
        expected_values = np.asarray(explainer.expected_value)
        assert (
            len(expected_values) == est.n_outputs_
        ), "Length of expected_values doesn't match n_outputs_"

        explanation = explainer(X_test)
        # check the properties of Explanation object
        assert explanation.values.shape == (*X_test.shape, est.n_outputs_)
        assert explanation.base_values.shape == (len(X_test), est.n_outputs_)

        # check that SHAP values sum to model output for all multioutputs
        assert (
            np.abs(explanation.values.sum(1) + explanation.base_values - predicted).max()
            < 1e-4
        )

    def test_gradient_boosting_classifier_invalid_init_estimator(self):
        """Currently only the logodds estimators are supported, so this test checks that
        an appropriate error is thrown when other estimator types are passed in.

        Remove/modify this test if we support other init estimator types in the future.
        """
        clf = sklearn.ensemble.GradientBoostingClassifier(
            n_estimators=10, init="zero",
        )
        clf.fit(*shap.datasets.adult())
        with pytest.raises(InvalidModelError):
            shap.TreeExplainer(clf)

    def test_single_row_gradient_boosting_classifier(self):
        X_train, X_test, Y_train, _ = sklearn.model_selection.train_test_split(
            *shap.datasets.adult(),
            test_size=0.2,
            random_state=0,
        )
        clf = sklearn.ensemble.GradientBoostingClassifier(
            random_state=202,
            n_estimators=10,
            max_depth=10,
        )
        clf.fit(X_train, Y_train)
        predicted = clf.decision_function(X_test)
        ex = shap.TreeExplainer(clf)
        shap_values = ex.shap_values(X_test.iloc[0, :])

        # check that SHAP values sum to model output
        assert np.abs(shap_values.sum() + ex.expected_value - predicted[0]) < 1e-4

    def test_sum_match_gradient_boosting_classifier(self):
        X_train, X_test, Y_train, _ = sklearn.model_selection.train_test_split(
            *shap.datasets.adult(),
            test_size=0.2,
            random_state=0,
        )
        clf = sklearn.ensemble.GradientBoostingClassifier(
            random_state=202,
            n_estimators=10,
            max_depth=10,
        )
        clf.fit(X_train, Y_train)

        # Use decision function to get prediction before it is mapped to a probability
        predicted = clf.decision_function(X_test)

        explainer = shap.TreeExplainer(clf)
        initial_ex_value = explainer.expected_value

        explanation = explainer(X_test)
        # check the properties of Explanation object
        assert explanation.values.shape == (*X_test.shape,)
        assert explanation.base_values.shape == (len(X_test),)

        # check that SHAP values sum to model output
        assert (
            np.abs(explanation.values.sum(1) + explanation.base_values - predicted).max()
            < 1e-4
        )

        # check initial expected value
        assert (
            np.abs(initial_ex_value - explainer.expected_value) < 1e-4
        ), "Initial expected value is wrong!"

        # check SHAP interaction values sum to model output
        shap_interaction_values = explainer.shap_interaction_values(X_test.iloc[:10, :])
        assert (
            np.abs(
                shap_interaction_values.sum(1).sum(1)
                + explainer.expected_value
                - predicted[:10]
            ).max()
            < 1e-4
        )

    def test_single_row_gradient_boosting_regressor(self):
        X_train, X_test, Y_train, _ = sklearn.model_selection.train_test_split(
            *shap.datasets.adult(),
            test_size=0.2,
            random_state=0,
        )
        clf = sklearn.ensemble.GradientBoostingRegressor(
            random_state=202, n_estimators=10, max_depth=10
        )
        clf.fit(X_train, Y_train)

        predicted = clf.predict(X_test)
        ex = shap.TreeExplainer(clf)
        shap_values = ex.shap_values(X_test.iloc[0, :])

        # check that SHAP values sum to model output
        assert np.abs(shap_values.sum() + ex.expected_value - predicted[0]) < 1e-4

    def test_sum_match_gradient_boosting_regressor(self):
        X_train, X_test, Y_train, _ = sklearn.model_selection.train_test_split(
            *shap.datasets.adult(),
            test_size=0.2,
            random_state=0,
        )
        clf = sklearn.ensemble.GradientBoostingRegressor(
            random_state=202, n_estimators=10, max_depth=10
        )
        clf.fit(X_train, Y_train)

        predicted = clf.predict(X_test)
        explainer = shap.TreeExplainer(clf)

        explanation = explainer(X_test)
        # check the properties of Explanation object
        assert explanation.values.shape == (*X_test.shape,)
        assert explanation.base_values.shape == (len(X_test),)

        # check that SHAP values sum to model output
        assert (
            np.abs(explanation.values.sum(1) + explanation.base_values - predicted).max()
            < 1e-4
        )

    def test_HistGradientBoostingClassifier_proba(self):
        X, y = shap.datasets.adult()
        model = sklearn.ensemble.HistGradientBoostingClassifier(
            max_iter=10, max_depth=6
        ).fit(X, y)
        predicted = model.predict_proba(X)
        explainer = shap.TreeExplainer(
            model, shap.sample(X, 10), model_output="predict_proba"
        )

        explanation = explainer(X)
        # check the properties of Explanation object
        num_classes = 2
        assert explanation.values.shape == (*X.shape, num_classes)
        assert explanation.base_values.shape == (len(X), num_classes)

        # check that SHAP values sum to model output
        assert (
            np.abs(explanation.values.sum(1) + explanation.base_values - predicted).max()
            < 1e-4
        )

    def test_HistGradientBoostingClassifier_multidim(self, random_seed):
        X, y = shap.datasets.adult(n_points=400)
        rs = np.random.RandomState(random_seed)
        y = rs.randint(0, 3, len(y))
        model = sklearn.ensemble.HistGradientBoostingClassifier(
            max_iter=10, max_depth=6
        ).fit(X, y)
        predicted = model.decision_function(X)
        explainer = shap.TreeExplainer(model, shap.sample(X, 10), model_output="raw")

        explanation = explainer(X)
        # check the properties of Explanation object
        num_classes = 3
        assert explanation.values.shape == (*X.shape, num_classes)
        assert explanation.base_values.shape == (len(X), num_classes)

        # check that SHAP values sum to model output
        assert (
            np.abs(explanation.values.sum(1) + explanation.base_values - predicted).max()
            < 1e-4
        )

    def test_HistGradientBoostingRegressor(self):
        X, y = shap.datasets.diabetes()
        model = sklearn.ensemble.HistGradientBoostingRegressor(
            max_iter=500, max_depth=6
        ).fit(X, y)
        predicted = model.predict(X)
        explainer = shap.TreeExplainer(model)

        explanation = explainer(X)
        # check the properties of Explanation object
        assert explanation.values.shape == (*X.shape,)
        assert explanation.base_values.shape == (len(X),)

        # check that SHAP values sum to model output
        assert (
            np.abs(explanation.values.sum(1) + explanation.base_values - predicted).max()
            < 1e-4
        )


class TestExplainerXGBoost:
    """Tests for the TreeExplainer with XGBoost models.

    Included models:
        * XGBRegressor
        * XGBClassifier
        * XGBRanker
    """

    def test_xgboost_regression(self):
        xgboost = pytest.importorskip("xgboost")

        # train xgboost model
        X, y = shap.datasets.california(n_points=500)
        model = xgboost.XGBRegressor().fit(X, y)
        predicted = model.predict(X)

        # explain the model's predictions using SHAP values
        explainer = shap.TreeExplainer(model)
        explanation = explainer(X)
        # check the properties of Explanation object
        assert explanation.values.shape == (*X.shape,)
        assert explanation.base_values.shape == (len(X),)

        # check that SHAP values sum to model output
        expected_diff = np.abs(explanation.values.sum(1) + explanation.base_values - predicted).max()
        assert expected_diff < 1e-4, "SHAP values don't sum to model output!"

    def test_xgboost_dmatrix_propagation(self):
        """
        Test that xgboost sklearn attributues are properly passed to the DMatrix
        initiated during shap value calculation. see GH #3313
        """
        xgboost = pytest.importorskip("xgboost")

        X, y = shap.datasets.adult(n_points=100)

        # Randomly add missing data to the input where missing data is encoded as 1e-8
        X_nan = X.copy()
        X_nan.loc[
            X_nan.sample(frac=0.3, random_state=42).index,
            X_nan.columns.to_series().sample(frac=0.5, random_state=42),
        ] = 1e-8

        clf = xgboost.XGBClassifier(missing=1e-8, random_state=42)
        clf.fit(X_nan, y)
        margin = clf.predict(X_nan, output_margin=True)
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_nan)
        # check that SHAP values sum to model output
        assert np.allclose(margin, explainer.expected_value + shap_values.sum(axis=1))

    def test_xgboost_direct(self):
        xgboost = pytest.importorskip("xgboost")

        # FIXME: this test should ideally pass with any random seed. See #2960
        random_seed = 0
        rs = np.random.RandomState(random_seed)
        N = 100
        M = 4
        X = rs.standard_normal(size=(N, M))
        y = rs.standard_normal(size=N)

        model = xgboost.XGBRegressor(random_state=rs)
        model.fit(X, y)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        assert np.allclose(shap_values[0, :], _brute_force_tree_shap(explainer.model, X[0, :]))

    def test_xgboost_multiclass(self):
        xgboost = pytest.importorskip("xgboost")

        # train XGBoost model
        X, y = shap.datasets.iris()
        model = xgboost.XGBClassifier(n_estimators=10, max_depth=4)
        model.fit(X, y)
        predicted = model.predict(X, output_margin=True)

        # explain the model's predictions using SHAP values
        explainer = shap.TreeExplainer(model)

        assert np.allclose(explainer.model.predict(X), predicted)

        explanation = explainer(X)
        # check the properties of Explanation object
        num_classes = 3
        assert explanation.values.shape == (*X.shape, num_classes)
        assert explanation.base_values.shape == (len(X), num_classes)

        # check that SHAP values sum to model output
        assert np.allclose(
            explanation.values.sum(1) + explanation.base_values,
            predicted,
        )

        # ensure plot works for first class
        shap.dependence_plot(0, explanation[..., 0].values, X, show=False)

    def test_xgboost_ranking(self):
        xgboost = pytest.importorskip("xgboost")

        # train xgboost ranker model
        x_train, y_train, x_test, _, q_train, _ = shap.datasets.rank()
        params = {
            "objective": "rank:pairwise",
            "learning_rate": 0.1,
            "gamma": 1.0,
            "min_child_weight": 0.1,
            "max_depth": 5,
            "n_estimators": 4,
        }
        model = xgboost.sklearn.XGBRanker(**params)
        model.fit(x_train, y_train, group=q_train.astype(int))
        _validate_shap_values(model, x_test)

    def test_xgboost_mixed_types(self):
        xgboost = pytest.importorskip("xgboost")

        X, y = shap.datasets.california(n_points=500)
        X["HouseAge"] = X["HouseAge"].astype(np.int64)
        X['IsOld'] = (X['HouseAge'] > 30)
        bst = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 1000)
        shap_values = shap.TreeExplainer(bst).shap_values(X)
        shap.dependence_plot(0, shap_values, X, show=False)

    def test_xgboost_classifier_independent_margin(self):
        # FIXME: this test should ideally pass with any random seed. See #2960
        random_seed = 0

        xgboost = pytest.importorskip("xgboost")
        # train XGBoost model
        rs = np.random.RandomState(random_seed)
        n = 1000
        X = rs.normal(size=(n, 7))
        y = np.matmul(X, [-2, 1, 3, 5, 2, 20, -5])
        y = y + abs(min(y))
        y = rs.binomial(n=1, p=y / max(y))

        model = xgboost.XGBClassifier(
            n_estimators=10, max_depth=5, random_state=random_seed, tree_method="exact"
        )
        model.fit(X, y)
        predicted = model.predict(X, output_margin=True)

        # explain the model's predictions using SHAP values
        explainer = shap.TreeExplainer(
            model,
            X,
            feature_perturbation="interventional",
            model_output="raw",
        )
        explanation = explainer(X)
        # check the properties of Explanation object
        assert explanation.values.shape == (*X.shape,)
        assert explanation.base_values.shape == (len(X),)

        # check that SHAP values sum to model output
        assert np.allclose(
            explanation.values.sum(1) + explanation.base_values,
            predicted,
            atol=1e-7,
        )

    def test_xgboost_classifier_independent_probability(self, random_seed):
        xgboost = pytest.importorskip("xgboost")

        # train XGBoost model
        rs = np.random.RandomState(random_seed)
        n = 1000
        X = rs.normal(size=(n, 7))
        b = np.array([-2, 1, 3, 5, 2, 20, -5])
        y = np.matmul(X, b)
        y = y + abs(min(y))
        y = rs.binomial(n=1, p=y / max(y))

        model = xgboost.XGBClassifier(n_estimators=10, max_depth=5, random_state=random_seed)
        model.fit(X, y)
        predicted = model.predict_proba(X)

        # explain the model's predictions using SHAP values
        explainer = shap.TreeExplainer(
            model,
            X,
            feature_perturbation="interventional",
            model_output="probability",
        )
        explanation = explainer(X)
        # check the properties of Explanation object
        assert explanation.values.shape == (*X.shape,)
        assert explanation.base_values.shape == (len(X),)

        # check that SHAP values sum to model output
        assert np.allclose(
            explanation.values.sum(1) + explanation.base_values,
            predicted[:, 1],
        )

    # def test_front_page_xgboost_global_path_dependent():
    #     try:
    #         xgboost = pytest.importorskip("xgboost")
    #     except Exception:
    #         print("Skipping test_front_page_xgboost!")
    #         return
    #
    #     # train XGBoost model
    #     X, y = shap.datasets.california(n_points=500)
    #     model = xgboost.XGBRegressor()
    #     model.fit(X, y)

    #     # explain the model's predictions using SHAP values
    #     explainer = shap.TreeExplainer(model, X, feature_perturbation="global_path_dependent")
    #     shap_values = explainer.shap_values(X)

    #     assert np.allclose(shap_values.sum(1) + explainer.expected_value, model.predict(X))

    def test_xgboost_buffer_strip(self, random_seed):
        # test to make sure bug #1864 doesn't get reintroduced
        xgboost = pytest.importorskip("xgboost")
        X = np.array([[1, 2, 3, 4, 5], [3, 3, 3, 2, 4]])
        y = np.array([1, 0])
        # specific values (e.g. 1.3) caused the bug previously
        model = xgboost.XGBRegressor(base_score=1.3, eval_metric="rmse", random_state=random_seed)
        model.fit(X, y)
        # previous bug did .lstrip('binf'), so would have incorrectly handled
        # buffer starting with binff
        assert model.get_booster().save_raw().startswith(b"binff")

        # after this fix, this line should not error
        explainer = shap.TreeExplainer(model)
        assert isinstance(explainer, shap.explainers.TreeExplainer)

    def test_explanation_data_not_dmatrix(self, random_seed):
        """Checks that DMatrix is not stored in Explanation.data after TreeExplainer.__call__,
        since it is not supported by our plotting functions.

        See GH #3357 for more information."""
        xgboost = pytest.importorskip("xgboost")

        rs = np.random.RandomState(random_seed)
        X = rs.normal(size=(100, 7))
        y = np.matmul(X, [-2, 1, 3, 5, 2, 20, -5])

        # train a model with single tree
        Xd = xgboost.DMatrix(X, label=y)
        model = xgboost.train({"eta": 1, "max_depth": 6, "base_score": 0, "lambda": 0}, Xd, 1)

        explainer = shap.TreeExplainer(model)
        explanation = explainer(Xd)

        assert not isinstance(explanation.data, xgboost.core.DMatrix)
        assert hasattr(explanation.data, "shape")


class TestExplainerLightGBM:
    """Tests for the TreeExplainer when the model passed in is a LightGBM instance.

    Included models:
        * LGBMRegressor
        * LGBMClassifier
    """

    def test_lightgbm(self):
        """Test the basic `shap_values` calculation."""
        lightgbm = pytest.importorskip("lightgbm")

        # train lightgbm model
        X, y = shap.datasets.california(n_points=500)
        dataset = lightgbm.Dataset(data=X, label=y, categorical_feature=[8])
        model = lightgbm.train(
            {
                "objective": "regression",
                "verbosity": -1,
                "num_threads": 1,
            },
            train_set=dataset,
            num_boost_round=1_000,
        )
        predicted = model.predict(X, raw_score=True)

        # explain the model's predictions using SHAP values
        explainer = shap.TreeExplainer(model)

        explanation = explainer(X)
        # check the properties of Explanation object
        assert explanation.values.shape == (*X.shape,)
        assert explanation.base_values.shape == (len(X),)

        # check that SHAP values sum to model output
        assert (
            np.abs(explanation.values.sum(1) + explanation.base_values - predicted).max()
            < 1e-4
        )

    def test_lightgbm_constant_prediction(self):
        # note: this test used to fail with lightgbm 2.2.1 with error:
        # ValueError: zero-size array to reduction operation maximum which has no identity
        # on TreeExplainer when trying to compute max nodes:
        # max_nodes = np.max([len(t.values) for t in self.trees])
        # The test does not fail with latest lightgbm 2.2.3 however
        lightgbm = pytest.importorskip("lightgbm")

        # train lightgbm model with a constant value for y
        X, y = shap.datasets.california(n_points=500)
        # use the mean for all values
        y.fill(np.mean(y))
        dataset = lightgbm.Dataset(data=X, label=y, categorical_feature=[8])
        model = lightgbm.train({"objective": "regression", "verbosity": -1,
                                "num_threads": 1},
                               train_set=dataset, num_boost_round=1000)

        # explain the model's predictions using SHAP values
        shap.TreeExplainer(model).shap_values(X)

    def test_lightgbm_binary(self):
        lightgbm = pytest.importorskip("lightgbm")

        # train lightgbm model
        X_train, X_test, Y_train, _ = sklearn.model_selection.train_test_split(
            *shap.datasets.adult(n_points=500),
            test_size=0.2,
            random_state=0,
        )
        dataset = lightgbm.Dataset(data=X_train, label=Y_train)
        model = lightgbm.train(
            {
                "objective": "binary",
                "verbosity": -1,
                "num_threads": 1,
            },
            train_set=dataset,
            num_boost_round=1_000,
        )
        predicted = model.predict(X_test, raw_score=True)

        # explain the model's predictions using SHAP values
        explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(X_test)
        # validate structure of shap values, must be a list of ndarray for both classes
        assert isinstance(shap_values, list)
        assert len(shap_values) == 2

        explanation = explainer(X_test)
        # check the properties of Explanation object
        assert explanation.values.shape == (*X_test.shape, 2)
        assert explanation.base_values.shape == (len(X_test), 2)

        # check that SHAP values sum to model output
        class1_exp = explanation[..., 1]
        assert (
            np.abs(class1_exp.values.sum(1) + class1_exp.base_values - predicted).max()
            < 1e-4
        )

        # ensure plot works for first class
        shap.dependence_plot(0, shap_values[0], X_test, show=False)

    def test_lightgbm_constant_multiclass(self):
        # note: this test used to fail with lightgbm 2.2.1 with error:
        # ValueError: zero-size array to reduction operation maximum which has no identity
        # on TreeExplainer when trying to compute max nodes:
        # max_nodes = np.max([len(t.values) for t in self.trees])
        # The test does not fail with latest lightgbm 2.2.3 however
        lightgbm = pytest.importorskip("lightgbm")

        # train lightgbm model
        X, Y = shap.datasets.iris()
        Y.fill(1)
        model = lightgbm.LGBMClassifier(
            n_estimators=50,
            num_classes=3,
            objective="multiclass",
            n_jobs=1,
        )
        model.fit(X, Y)

        # explain the model's predictions using SHAP values
        shap.TreeExplainer(model).shap_values(X)

    def test_lightgbm_multiclass(self):
        lightgbm = pytest.importorskip("lightgbm")

        # train lightgbm model
        X, Y = shap.datasets.iris()
        model = lightgbm.LGBMClassifier(n_jobs=1)
        model.fit(X, Y)
        predicted = model.predict(X, raw_score=True)

        # explain the model's predictions using SHAP values
        explainer = shap.TreeExplainer(model)

        explanation = explainer(X)
        # check the properties of Explanation object
        num_classes = 3
        assert explanation.values.shape == (*X.shape, num_classes)
        assert explanation.base_values.shape == (len(X), num_classes)

        # check that SHAP values sum to model output
        assert (
            np.abs(explanation.values.sum(1) + explanation.base_values - predicted).max()
            < 1e-4
        )

    # def test_lightgbm_ranking(self):
    #     try:
    #         import lightgbm
    #     except Exception:
    #         print("Skipping test_lightgbm_ranking!")
    #         return
    #
    #     # train lightgbm ranker model
    #     x_train, y_train, x_test, y_test, q_train, q_test = shap.datasets.rank()
    #     model = lightgbm.LGBMRanker()
    #     model.fit(
    #         x_train, y_train, group=q_train,
    #         eval_set=[(x_test, y_test)],
    #         eval_group=[q_test],
    #         eval_at=[1, 3],
    #         early_stopping_rounds=5,
    #         verbose=False,
    #         callbacks=[lightgbm.reset_parameter(learning_rate=lambda x: 0.95 ** x * 0.1)],
    #     )
    #     _validate_shap_values(model, x_test)

    def test_lightgbm_interaction(self):
        lightgbm = pytest.importorskip("lightgbm")

        # train LightGBM model
        X, y = shap.datasets.california(n_points=500)
        model = lightgbm.LGBMRegressor(n_estimators=20, n_jobs=1)
        model.fit(X, y)

        # verify symmetry of the interaction values (this typically breaks if anything is wrong)
        interaction_vals = shap.TreeExplainer(model).shap_interaction_values(X)
        for j, jval in enumerate(interaction_vals):
            for k, kval in enumerate(jval):
                for m, _ in enumerate(kval):
                    assert (
                        abs(interaction_vals[j][k][m] - interaction_vals[j][m][k])
                        < 1e-4
                    )

    def test_lightgbm_call_explanation(self):
        """Checks that __call__ runs without error and returns a valid Explanation object.

        Related to GH issue dsgibbons#66.
        """
        lightgbm = pytest.importorskip("lightgbm")

        # NOTE: the categorical column is necessary for testing GH issue dsgibbons#66.
        X, y = shap.datasets.adult(n_points=300)
        X["categ"] = pd.Categorical(
            [p for p in ("M", "F") for _ in range(150)],
            ordered=False,
        )
        model = lightgbm.LGBMClassifier(n_estimators=7, n_jobs=1)
        model.fit(X, y)

        explainer = shap.TreeExplainer(model)
        explanation = explainer(X)

        shap_values_raw: list[np.ndarray] = explainer.shap_values(X)
        shap_values: np.ndarray = np.stack(shap_values_raw, axis=-1)

        # checks that the call returns a valid Explanation object
        assert len(explanation.base_values) == len(y)
        assert isinstance(explanation.values, np.ndarray)
        assert isinstance(shap_values, np.ndarray)
        assert (explanation.values == shap_values).all()

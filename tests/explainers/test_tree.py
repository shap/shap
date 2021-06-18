# pylint: disable=missing-function-docstring,too-many-lines,fixme
"""Test tree functions."""
import itertools
import math
import pickle
import numpy as np
import pandas as pd
import pytest
import sklearn
import sklearn.pipeline
from sklearn.experimental import enable_hist_gradient_boosting  # pylint: disable=unused-import
import shap


def test_front_page_xgboost():
    xgboost = pytest.importorskip('xgboost')

    # load JS visualization code to notebook
    shap.initjs()

    # train XGBoost model
    X, y = shap.datasets.boston()
    model = xgboost.train({"learning_rate": 0.01, "silent": 1}, xgboost.DMatrix(X, label=y), 100)

    # explain the model's predictions using SHAP values
    explainer = shap.TreeExplainer(model)
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


def test_front_page_sklearn():
    # load JS visualization code to notebook
    shap.initjs()

    # train model
    X, y = shap.datasets.boston()
    models = [
        sklearn.ensemble.RandomForestRegressor(n_estimators=10),
        sklearn.ensemble.ExtraTreesRegressor(n_estimators=10),
    ]
    for model in models:
        model.fit(X, y)

        # explain the model's predictions using SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # visualize the first prediction's explaination
        shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :])

        # visualize the training set predictions
        shap.force_plot(explainer.expected_value, shap_values, X)

        # create a SHAP dependence plot to show the effect of a single feature across the whole
        # dataset
        shap.dependence_plot(5, shap_values, X, show=False)
        shap.dependence_plot("RM", shap_values, X, show=False)

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


def test_xgboost_direct():
    xgboost = pytest.importorskip('xgboost')

    N = 100
    M = 4
    X = np.random.randn(N, M)
    y = np.random.randn(N)

    model = xgboost.XGBRegressor()
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    assert np.allclose(shap_values[0, :], _brute_force_tree_shap(explainer.model, X[0, :]))


def test_xgboost_multiclass():
    xgboost = pytest.importorskip('xgboost')

    # train XGBoost model
    X, Y = shap.datasets.iris()
    model = xgboost.XGBClassifier(objective="binary:logistic", max_depth=4)
    model.fit(X, Y)

    # explain the model's predictions using SHAP values (use pred_contrib in LightGBM)
    shap_values = shap.TreeExplainer(model).shap_values(X)

    # ensure plot works for first class
    shap.dependence_plot(0, shap_values[0], X, show=False)


def _validate_shap_values(model, x_test):
    # explain the model's predictions using SHAP values
    tree_explainer = shap.TreeExplainer(model)
    shap_values = tree_explainer.shap_values(x_test)
    expected_values = tree_explainer.expected_value
    # validate values sum to the margin prediction of the model plus expected_value
    assert np.allclose(np.sum(shap_values, axis=1) + expected_values, model.predict(x_test))


def test_xgboost_ranking():
    xgboost = pytest.importorskip('xgboost')

    # train lightgbm ranker model
    x_train, y_train, x_test, _, q_train, _ = shap.datasets.rank()
    params = {'objective': 'rank:pairwise', 'learning_rate': 0.1,
              'gamma': 1.0, 'min_child_weight': 0.1,
              'max_depth': 5, 'n_estimators': 4}
    model = xgboost.sklearn.XGBRanker(**params)
    model.fit(x_train, y_train, q_train.astype(int))
    _validate_shap_values(model, x_test)


def test_xgboost_mixed_types():
    xgboost = pytest.importorskip('xgboost')

    X, y = shap.datasets.boston()
    X["LSTAT"] = X["LSTAT"].astype(np.int64)
    X["B"] = X["B"].astype(bool)
    bst = xgboost.train({"learning_rate": 0.01, "silent": 1}, xgboost.DMatrix(X, label=y), 1000)
    shap_values = shap.TreeExplainer(bst).shap_values(X)
    shap.dependence_plot(0, shap_values, X, show=False)


def test_ngboost():
    ngboost = pytest.importorskip('ngboost')

    X, y = shap.datasets.boston()
    model = ngboost.NGBRegressor(n_estimators=20).fit(X, y)
    explainer = shap.TreeExplainer(model, model_output=0)
    assert np.max(np.abs(
        explainer.shap_values(X).sum(1) + explainer.expected_value - model.predict(X))) < 1e-5


def test_pyspark_classifier_decision_tree():
    # pylint: disable=bare-except
    pyspark = pytest.importorskip("pyspark")
    pytest.importorskip("pyspark.ml")
    try:
        spark = pyspark.sql.SparkSession.builder.config(
            conf=pyspark.SparkConf().set("spark.master", "local[*]")).getOrCreate()
    except:
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

        shap_values = explainer.shap_values(X)
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


def test_pyspark_regression_decision_tree():
    # pylint: disable=bare-except
    pyspark = pytest.importorskip("pyspark")
    pytest.importorskip("pyspark.ml")
    try:
        spark = pyspark.sql.SparkSession.builder.config(
            conf=pyspark.SparkConf().set("spark.master", "local[*]")).getOrCreate()
    except:
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
        X = pd.DataFrame(data=iris_sk.data, columns=iris_sk.feature_names).drop('sepal length (cm)', 1)[:100] # pylint: disable=E1101

        shap_values = explainer.shap_values(X)
        expected_values = explainer.expected_value

        # validate values sum to the margin prediction of the model plus expected_value
        predictions = model.transform(iris).select("prediction").toPandas()
        diffs = expected_values + shap_values.sum(1) - predictions["prediction"]
        assert np.max(np.abs(diffs)) < 1e-4, "SHAP values don't sum to model output for class0!"
        assert (np.abs(expected_values - predictions.mean()) < 1e-1).all(), "Bad expected_value!"
    spark.stop()


def test_sklearn_random_forest_multiclass():
    X, y = shap.datasets.iris()
    y[y == 2] = 1
    model = sklearn.ensemble.RandomForestClassifier(n_estimators=100, max_depth=None,
                                                    min_samples_split=2,
                                                    random_state=0)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    assert np.abs(shap_values[0][0, 0] - 0.05) < 1e-3
    assert np.abs(shap_values[1][0, 0] + 0.05) < 1e-3


def create_binary_newsgroups_data():
    categories = ['alt.atheism', 'soc.religion.christian']
    newsgroups_train = sklearn.datasets.fetch_20newsgroups(subset='train', categories=categories)
    newsgroups_test = sklearn.datasets.fetch_20newsgroups(subset='test', categories=categories)
    class_names = ['atheism', 'christian']
    return newsgroups_train, newsgroups_test, class_names


def create_random_forest_vectorizer():
    # pylint: disable=unused-argument,no-self-use,missing-class-docstring
    vectorizer = sklearn.feature_extraction.text.CountVectorizer(lowercase=False, min_df=0.0,
                                                                 binary=True)

    class DenseTransformer(sklearn.base.TransformerMixin):
        def fit(self, X, y=None, **fit_params):
            return self

        def transform(self, X, y=None, **fit_params):
            return X.toarray()

    rf = sklearn.ensemble.RandomForestClassifier(n_estimators=10, random_state=777)
    return sklearn.pipeline.Pipeline(
        [('vectorizer', vectorizer), ('to_dense', DenseTransformer()), ('rf', rf)])


def test_sklearn_random_forest_newsgroups():
    # note: this test used to fail in native TreeExplainer code due to memory corruption
    newsgroups_train, newsgroups_test, _ = create_binary_newsgroups_data()
    pipeline = create_random_forest_vectorizer()
    pipeline.fit(newsgroups_train.data, newsgroups_train.target)
    rf = pipeline.named_steps['rf']
    vectorizer = pipeline.named_steps['vectorizer']
    densifier = pipeline.named_steps['to_dense']

    dense_bg = densifier.transform(vectorizer.transform(newsgroups_test.data[0:20]))

    test_row = newsgroups_test.data[83:84]
    explainer = shap.TreeExplainer(rf, dense_bg, feature_perturbation="interventional")
    vec_row = vectorizer.transform(test_row)
    dense_row = densifier.transform(vec_row)
    explainer.shap_values(dense_row)


def test_sklearn_decision_tree_multiclass():
    X, y = shap.datasets.iris()
    y[y == 2] = 1
    model = sklearn.tree.DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    assert np.abs(shap_values[0][0, 0] - 0.05) < 1e-1
    assert np.abs(shap_values[1][0, 0] + 0.05) < 1e-1


def test_lightgbm():
    lightgbm = pytest.importorskip("lightgbm")

    # train lightgbm model
    X, y = shap.datasets.boston()
    model = lightgbm.sklearn.LGBMRegressor(categorical_feature=[8])
    model.fit(X, y)

    # explain the model's predictions using SHAP values
    ex = shap.TreeExplainer(model)
    shap_values = ex.shap_values(X)

    predicted = model.predict(X, raw_score=True)

    assert np.abs(shap_values.sum(1) + ex.expected_value - predicted).max() < 1e-4, \
        "SHAP values don't sum to model output!"


def test_gpboost():
    gpboost = pytest.importorskip("gpboost")
    # train gpboost model
    X, y = shap.datasets.boston()
    data_train = gpboost.Dataset(X, y, categorical_feature=[8])
    model = gpboost.train(params={'objective': 'regression_l2', 'learning_rate': 0.1, 'verbose': 0},
                          train_set=data_train, num_boost_round=10)

    # explain the model's predictions using SHAP values
    ex = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    shap_values = ex.shap_values(X)

    predicted = model.predict(X, raw_score=True)

    assert np.abs(shap_values.sum(1) + ex.expected_value - predicted).max() < 1e-4, \
        "SHAP values don't sum to model output!"


def test_catboost():
    catboost = pytest.importorskip("catboost")
    # train catboost model
    X, y = shap.datasets.boston()
    X["RAD"] = X["RAD"].astype(np.int)
    model = catboost.CatBoostRegressor(iterations=30, learning_rate=0.1, random_seed=123)
    p = catboost.Pool(X, y, cat_features=["RAD"])
    model.fit(p, verbose=False, plot=False)

    # explain the model's predictions using SHAP values
    ex = shap.TreeExplainer(model)
    shap_values = ex.shap_values(p)

    predicted = model.predict(X)

    assert np.abs(shap_values.sum(1) + ex.expected_value - predicted).max() < 1e-4, \
        "SHAP values don't sum to modThisel output!"

    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    model = catboost.CatBoostClassifier(iterations=10, learning_rate=0.5, random_seed=12)
    model.fit(
        X,
        y,
        verbose=False,
        plot=False
    )
    ex = shap.TreeExplainer(model)
    shap_values = ex.shap_values(X)

    predicted = model.predict(X, prediction_type="RawFormulaVal")
    assert np.abs(shap_values.sum(1) + ex.expected_value - predicted).max() < 1e-4, \
        "SHAP values don't sum to model output!"


def test_catboost_categorical():
    catboost = pytest.importorskip("catboost")
    bunch = sklearn.datasets.load_boston()
    X, y = sklearn.datasets.load_boston(return_X_y=True)
    X = pd.DataFrame(X, columns=bunch.feature_names)  # pylint: disable=no-member
    X['CHAS'] = X['CHAS'].astype(str)

    model = catboost.CatBoostRegressor(100, cat_features=['CHAS'], verbose=False)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    predicted = model.predict(X)

    assert np.abs(shap_values.sum(1) + explainer.expected_value - predicted).max() < 1e-4, \
        "SHAP values don't sum to model output!"


def test_lightgbm_constant_prediction():
    # note: this test used to fail with lightgbm 2.2.1 with error:
    # ValueError: zero-size array to reduction operation maximum which has no identity
    # on TreeExplainer when trying to compute max nodes:
    # max_nodes = np.max([len(t.values) for t in self.trees])
    # The test does not fail with latest lightgbm 2.2.3 however
    lightgbm = pytest.importorskip("lightgbm")
    # train lightgbm model with a constant value for y
    X, y = shap.datasets.boston()
    # use the mean for all values
    mean = np.mean(y)
    y.fill(mean)
    model = lightgbm.sklearn.LGBMRegressor(n_estimators=1)
    model.fit(X, y)

    # explain the model's predictions using SHAP values
    shap.TreeExplainer(model).shap_values(X)


def test_lightgbm_constant_multiclass():
    # note: this test used to fail with lightgbm 2.2.1 with error:
    # ValueError: zero-size array to reduction operation maximum which has no identity
    # on TreeExplainer when trying to compute max nodes:
    # max_nodes = np.max([len(t.values) for t in self.trees])
    # The test does not fail with latest lightgbm 2.2.3 however
    lightgbm = pytest.importorskip("lightgbm")

    # train lightgbm model
    X, Y = shap.datasets.iris()
    Y.fill(1)
    model = lightgbm.sklearn.LGBMClassifier(num_classes=3, objective="multiclass")
    model.fit(X, Y)

    # explain the model's predictions using SHAP values
    shap.TreeExplainer(model).shap_values(X)


def test_lightgbm_multiclass():
    lightgbm = pytest.importorskip("lightgbm")
    # train lightgbm model
    X, Y = shap.datasets.iris()
    model = lightgbm.sklearn.LGBMClassifier()
    model.fit(X, Y)

    # explain the model's predictions using SHAP values
    shap_values = shap.TreeExplainer(model).shap_values(X)

    # ensure plot works for first class
    shap.dependence_plot(0, shap_values[0], X, show=False)


def test_lightgbm_binary():
    lightgbm = pytest.importorskip("lightgbm")
    # train lightgbm model
    X_train, X_test, Y_train, _ = sklearn.model_selection.train_test_split(*shap.datasets.adult(),
                                                                           test_size=0.2,
                                                                           random_state=0)
    model = lightgbm.sklearn.LGBMClassifier()
    model.fit(X_train, Y_train)

    # explain the model's predictions using SHAP values
    shap_values = shap.TreeExplainer(model).shap_values(X_test)

    # validate structure of shap values, must be a list of ndarray for both classes
    assert isinstance(shap_values, list)
    assert len(shap_values) == 2

    # ensure plot works for first class
    shap.dependence_plot(0, shap_values[0], X_test, show=False)


# def test_lightgbm_ranking():
#     try:
#         import lightgbm
#     except:
#         print("Skipping test_lightgbm_ranking!")
#         return
#
#

#     # train lightgbm ranker model
#     x_train, y_train, x_test, y_test, q_train, q_test = shap.datasets.rank()
#     model = lightgbm.LGBMRanker()
#     model.fit(x_train, y_train, group=q_train, eval_set=[(x_test, y_test)],
#               eval_group=[q_test], eval_at=[1, 3], early_stopping_rounds=5, verbose=False,
#               callbacks=[lightgbm.reset_parameter(learning_rate=lambda x: 0.95 ** x * 0.1)])
#     _validate_shap_values(model, x_test)

# TODO: Test tree_limit argument

def test_sklearn_interaction():
    # train a simple sklean RF model on the iris dataset
    X, _ = shap.datasets.iris()
    X_train, _, Y_train, _ = sklearn.model_selection.train_test_split(*shap.datasets.iris(),
                                                                      test_size=0.2, random_state=0)
    rforest = sklearn.ensemble.RandomForestClassifier(n_estimators=100, max_depth=None,
                                                      min_samples_split=2,
                                                      random_state=0)
    model = rforest.fit(X_train, Y_train)

    # verify symmetry of the interaction values (this typically breaks if anything is wrong)
    interaction_vals = shap.TreeExplainer(model).shap_interaction_values(X)
    for i, _ in enumerate(interaction_vals):
        for j, _ in enumerate(interaction_vals[i]):
            for k, _ in enumerate(interaction_vals[i][j]):
                for l, _ in enumerate(interaction_vals[i][j][k]):
                    assert abs(interaction_vals[i][j][k][l] - interaction_vals[i][j][l][k]) < 1e-4

    # ensure the interaction plot works
    shap.summary_plot(interaction_vals[0], X, show=False)


def test_lightgbm_interaction():
    lightgbm = pytest.importorskip("lightgbm")

    # train XGBoost model
    X, y = shap.datasets.boston()
    model = lightgbm.sklearn.LGBMRegressor()
    model.fit(X, y)

    # verify symmetry of the interaction values (this typically breaks if anything is wrong)
    interaction_vals = shap.TreeExplainer(model).shap_interaction_values(X)
    for j, _ in enumerate(interaction_vals):
        for k, _ in enumerate(interaction_vals[j]):
            for l, _ in enumerate(interaction_vals[j][k]):
                assert abs(interaction_vals[j][k][l] - interaction_vals[j][l][k]) < 1e-4


def test_sum_match_random_forest():
    X_train, X_test, Y_train, _ = sklearn.model_selection.train_test_split(*shap.datasets.adult(),
                                                                           test_size=0.2,
                                                                           random_state=0)
    clf = sklearn.ensemble.RandomForestClassifier(random_state=202, n_estimators=10, max_depth=10)
    clf.fit(X_train, Y_train)
    predicted = clf.predict_proba(X_test)
    ex = shap.TreeExplainer(clf)
    shap_values = ex.shap_values(X_test)
    assert np.abs(shap_values[0].sum(1) + ex.expected_value[0] - predicted[:, 0]).max() < 1e-4, \
        "SHAP values don't sum to model output!"


def test_sum_match_extra_trees():
    X_train, X_test, Y_train, _ = sklearn.model_selection.train_test_split(*shap.datasets.adult(),
                                                                           test_size=0.2,
                                                                           random_state=0)
    clf = sklearn.ensemble.ExtraTreesRegressor(random_state=202, n_estimators=10, max_depth=10)
    clf.fit(X_train, Y_train)
    predicted = clf.predict(X_test)
    ex = shap.TreeExplainer(clf)
    shap_values = ex.shap_values(X_test)
    assert np.abs(shap_values.sum(1) + ex.expected_value - predicted).max() < 1e-4, \
        "SHAP values don't sum to model output!"


def test_single_row_random_forest():
    X_train, X_test, Y_train, _ = sklearn.model_selection.train_test_split(*shap.datasets.adult(),
                                                                           test_size=0.2,
                                                                           random_state=0)
    clf = sklearn.ensemble.RandomForestClassifier(random_state=202, n_estimators=10, max_depth=10)
    clf.fit(X_train, Y_train)
    predicted = clf.predict_proba(X_test)
    ex = shap.TreeExplainer(clf)
    shap_values = ex.shap_values(X_test.iloc[0, :])
    assert np.abs(shap_values[0].sum() + ex.expected_value[0] - predicted[0, 0]) < 1e-4, \
        "SHAP values don't sum to model output!"


def test_sum_match_gradient_boosting_classifier():
    X_train, X_test, Y_train, _ = sklearn.model_selection.train_test_split(*shap.datasets.adult(),
                                                                           test_size=0.2,
                                                                           random_state=0)
    clf = sklearn.ensemble.GradientBoostingClassifier(random_state=202, n_estimators=10,
                                                      max_depth=10)
    clf.fit(X_train, Y_train)

    # Use decision function to get prediction before it is mapped to a probability
    predicted = clf.decision_function(X_test)

    # check SHAP values
    ex = shap.TreeExplainer(clf)
    initial_ex_value = ex.expected_value
    shap_values = ex.shap_values(X_test)
    assert np.abs(shap_values.sum(1) + ex.expected_value - predicted).max() < 1e-4, \
        "SHAP values don't sum to model output!"

    # check initial expected value
    assert np.abs(initial_ex_value - ex.expected_value) < 1e-4, "Inital expected value is wrong!"

    # check SHAP interaction values
    shap_interaction_values = ex.shap_interaction_values(X_test.iloc[:10, :])
    assert np.abs(
        shap_interaction_values.sum(1).sum(1) + ex.expected_value - predicted[:10]).max() < 1e-4, \
        "SHAP interaction values don't sum to model output!"


def test_single_row_gradient_boosting_classifier():
    X_train, X_test, Y_train, _ = sklearn.model_selection.train_test_split(*shap.datasets.adult(),
                                                                           test_size=0.2,
                                                                           random_state=0)
    clf = sklearn.ensemble.GradientBoostingClassifier(random_state=202, n_estimators=10,
                                                      max_depth=10)
    clf.fit(X_train, Y_train)
    predicted = clf.decision_function(X_test)
    ex = shap.TreeExplainer(clf)
    shap_values = ex.shap_values(X_test.iloc[0, :])
    assert np.abs(shap_values.sum() + ex.expected_value - predicted[0]) < 1e-4, \
        "SHAP values don't sum to model output!"


def test_HistGradientBoostingRegressor():
    # train a tree-based model
    X, y = shap.datasets.diabetes()
    model = sklearn.ensemble.HistGradientBoostingRegressor(max_iter=1000, max_depth=6).fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    assert np.max(np.abs(shap_values.sum(1) + explainer.expected_value - model.predict(X))) < 1e-4


def test_HistGradientBoostingClassifier_proba():
    # train a tree-based model
    X, y = shap.datasets.adult()
    model = sklearn.ensemble.HistGradientBoostingClassifier(max_iter=10, max_depth=6).fit(X, y)
    explainer = shap.TreeExplainer(model, shap.sample(X, 10), model_output="predict_proba")
    shap_values = explainer.shap_values(X)
    assert np.max(np.abs(
        shap_values[0].sum(1) + explainer.expected_value[0] - model.predict_proba(X)[:, 0])) < 1e-4


def test_HistGradientBoostingClassifier_multidim():
    # train a tree-based model
    X, y = shap.datasets.adult()
    X = X[:100]
    y = y[:100]
    y = np.random.randint(0, 3, len(y))
    model = sklearn.ensemble.HistGradientBoostingClassifier(max_iter=10, max_depth=6).fit(X, y)
    explainer = shap.TreeExplainer(model, shap.sample(X, 10), model_output="raw")
    shap_values = explainer.shap_values(X)
    assert np.max(np.abs(shap_values[0].sum(1) +
                         explainer.expected_value[0] - model.decision_function(X)[:, 0])) < 1e-4


def test_sum_match_gradient_boosting_regressor():
    X_train, X_test, Y_train, _ = sklearn.model_selection.train_test_split(*shap.datasets.adult(),
                                                                           test_size=0.2,
                                                                           random_state=0)
    clf = sklearn.ensemble.GradientBoostingRegressor(random_state=202, n_estimators=10,
                                                     max_depth=10)
    clf.fit(X_train, Y_train)

    predicted = clf.predict(X_test)
    ex = shap.TreeExplainer(clf)
    shap_values = ex.shap_values(X_test)
    assert np.abs(shap_values.sum(1) + ex.expected_value - predicted).max() < 1e-4, \
        "SHAP values don't sum to model output!"


def test_single_row_gradient_boosting_regressor():
    X_train, X_test, Y_train, _ = sklearn.model_selection.train_test_split(*shap.datasets.adult(),
                                                                           test_size=0.2,
                                                                           random_state=0)
    clf = sklearn.ensemble.GradientBoostingRegressor(random_state=202, n_estimators=10,
                                                     max_depth=10)
    clf.fit(X_train, Y_train)

    predicted = clf.predict(X_test)
    ex = shap.TreeExplainer(clf)
    shap_values = ex.shap_values(X_test.iloc[0, :])
    assert np.abs(shap_values.sum() + ex.expected_value - predicted[0]) < 1e-4, \
        "SHAP values don't sum to model output!"


def test_multi_target_random_forest():
    X_train, X_test, Y_train, _ = sklearn.model_selection.train_test_split(
        *shap.datasets.linnerud(), test_size=0.2,
        random_state=0)
    est = sklearn.ensemble.RandomForestRegressor(random_state=202, n_estimators=10, max_depth=10)
    est.fit(X_train, Y_train)
    predicted = est.predict(X_test)

    explainer = shap.TreeExplainer(est)
    expected_values = np.asarray(explainer.expected_value)
    assert len(
        expected_values) == est.n_outputs_, "Length of expected_values doesn't match n_outputs_"
    shap_values = np.asarray(explainer.shap_values(X_test)).reshape(
        est.n_outputs_ * X_test.shape[0], X_test.shape[1])
    phi = np.hstack((shap_values, np.repeat(expected_values, X_test.shape[0]).reshape(-1, 1)))
    assert np.allclose(phi.sum(1), predicted.flatten(order="F"), atol=1e-4)


def test_isolation_forest():
    IsolationForest = pytest.importorskip("sklearn.ensemble.IsolationForest")
    _average_path_length = pytest.importorskip("sklearn.ensemble._iforest._average_path_length")
    X, _ = shap.datasets.boston()
    for max_features in [1.0, 0.75]:
        iso = IsolationForest(max_features=max_features)
        iso.fit(X)

        explainer = shap.TreeExplainer(iso)
        shap_values = explainer.shap_values(X)

        l = _average_path_length(  # pylint: disable=protected-access
            np.array([iso.max_samples_]))[0]
        score_from_shap = - 2 ** (- (np.sum(shap_values, axis=1) + explainer.expected_value) / l)
        assert np.allclose(iso.score_samples(X), score_from_shap, atol=1e-7)


def test_pyod_isolation_forest():
    try:
        IForest = pytest.importorskip("pyod.models.iforest.IForest")
    except Exception: # pylint: disable=broad-except
        pytest.skip("Failed to import pyod.models.iforest.IForest")
    _average_path_length = pytest.importorskip("sklearn.ensemble._iforest._average_path_length")

    X, _ = shap.datasets.boston()
    for max_features in [1.0, 0.75]:
        iso = IForest(max_features=max_features)
        iso.fit(X)

        explainer = shap.TreeExplainer(iso)
        shap_values = explainer.shap_values(X)

        l = _average_path_length(np.array([iso.max_samples_]))[0]
        score_from_shap = - 2 ** (- (np.sum(shap_values, axis=1) + explainer.expected_value) / l)
        assert np.allclose(iso.detector_.score_samples(X), score_from_shap, atol=1e-7)


# TODO: this has sometimes failed with strange answers, should run memcheck on this for any
#  memory issues at some point...
def test_multi_target_extra_trees():
    X_train, X_test, Y_train, _ = sklearn.model_selection.train_test_split(
        *shap.datasets.linnerud(), test_size=0.2,
        random_state=0)
    est = sklearn.ensemble.ExtraTreesRegressor(random_state=202, n_estimators=10, max_depth=10)
    est.fit(X_train, Y_train)
    predicted = est.predict(X_test)

    explainer = shap.TreeExplainer(est)
    expected_values = np.asarray(explainer.expected_value)
    assert len(
        expected_values) == est.n_outputs_, "Length of expected_values doesn't match n_outputs_"
    shap_values = np.asarray(explainer.shap_values(X_test)).reshape(
        est.n_outputs_ * X_test.shape[0], X_test.shape[1])
    phi = np.hstack((shap_values, np.repeat(expected_values, X_test.shape[0]).reshape(-1, 1)))
    assert np.allclose(phi.sum(1), predicted.flatten(order="F"), atol=1e-4)


def test_provided_background_tree_path_dependent():
    xgboost = pytest.importorskip("xgboost")
    np.random.seed(10)

    X, y = shap.datasets.iris()
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
        'silent': 1
    }

    bst = xgboost.train(params=params, dtrain=dtrain, num_boost_round=100)

    explainer = shap.TreeExplainer(bst, test_x, feature_perturbation="tree_path_dependent")
    diffs = explainer.expected_value + \
            explainer.shap_values(test_x).sum(1) - bst.predict(dtest, output_margin=True)
    assert np.max(np.abs(diffs)) < 1e-4, "SHAP values don't sum to model output!"
    assert np.abs(explainer.expected_value - bst.predict(dtest,
                                                         output_margin=True).mean()) < 1e-6, \
        "Bad expected_value!"


def test_provided_background_independent():
    xgboost = pytest.importorskip("xgboost")

    np.random.seed(10)

    X, y = shap.datasets.iris()
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
        'silent': 1
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

    np.random.seed(10)

    X, y = shap.datasets.iris()
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
            'silent': 1
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
    np.random.seed(10)

    n = 100
    X = np.random.normal(size=(n, 7))
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
        x_ind = np.random.choice(X.shape[1])
        x = X[x_ind:x_ind + 1, :]

        expl = shap.TreeExplainer(model, X, feature_perturbation="interventional")
        f = lambda inp: model.predict(xgboost.DMatrix(inp))
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
    xgboost = pytest.importorskip("xgboost")
    np.random.seed(10)

    n = 1000
    X = np.random.normal(size=(n, 7))
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
        x_ind = np.random.choice(X.shape[1])
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

    xgboost = pytest.importorskip("xgboost")
    np.random.seed(10)

    n = 100
    X = np.random.normal(size=(n, 7))
    y = np.matmul(X, [-2, 1, 3, 5, 2, 20, -5])
    y = y + abs(min(y))
    y = np.random.binomial(n=1, p=y / max(y))

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
    f = lambda inp: model.predict(xgboost.DMatrix(inp), output_margin=True)
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


def test_xgboost_classifier_independent_margin():
    xgboost = pytest.importorskip("xgboost")
    # train XGBoost model
    np.random.seed(10)
    n = 1000
    X = np.random.normal(size=(n, 7))
    y = np.matmul(X, [-2, 1, 3, 5, 2, 20, -5])
    y = y + abs(min(y))
    y = np.random.binomial(n=1, p=y / max(y))

    model = xgboost.XGBClassifier(n_estimators=10, max_depth=5)
    model.fit(X, y)

    # explain the model's predictions using SHAP values
    e = shap.TreeExplainer(model, X, feature_perturbation="interventional", model_output="margin")
    shap_values = e.shap_values(X)

    assert np.allclose(shap_values.sum(1) + e.expected_value, model.predict(X, output_margin=True))


def test_xgboost_classifier_independent_probability():
    xgboost = pytest.importorskip("xgboost")

    # train XGBoost model
    np.random.seed(10)
    n = 1000
    X = np.random.normal(size=(n, 7))
    b = np.array([-2, 1, 3, 5, 2, 20, -5])
    y = np.matmul(X, b)
    y = y + abs(min(y))
    y = np.random.binomial(n=1, p=y / max(y))

    model = xgboost.XGBClassifier(n_estimators=10, max_depth=5)
    model.fit(X, y)

    # explain the model's predictions using SHAP values
    e = shap.TreeExplainer(model, X, feature_perturbation="interventional",
                           model_output="probability")
    shap_values = e.shap_values(X)

    assert np.allclose(shap_values.sum(1) + e.expected_value, model.predict_proba(X)[:, 1])


# def test_front_page_xgboost_global_path_dependent():
#     try:
#         xgboost = pytest.importorskip("xgboost")
#     except:
#         print("Skipping test_front_page_xgboost!")
#         return
#
#

#     # train XGBoost model
#     X, y = shap.datasets.boston()
#     model = xgboost.XGBRegressor()
#     model.fit(X, y)

#     # explain the model's predictions using SHAP values
#     explainer = shap.TreeExplainer(model, X, feature_perturbation="global_path_dependent")
#     shap_values = explainer.shap_values(X)

#     assert np.allclose(shap_values.sum(1) + explainer.expected_value, model.predict(X))

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

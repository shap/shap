import matplotlib
matplotlib.use('Agg')


def test_front_page_xgboost():
    try:
        import xgboost
    except:
        print("Skipping test_front_page_xgboost!")
        return
    import shap
    import numpy as np

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
    import sklearn.ensemble
    import shap
    import numpy as np

    # load JS visualization code to notebook
    shap.initjs()

    # train model
    X, y = shap.datasets.boston()
    models = [
        sklearn.ensemble.RandomForestRegressor(n_estimators=100),
        sklearn.ensemble.ExtraTreesRegressor(n_estimators=100),
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

        # create a SHAP dependence plot to show the effect of a single feature across the whole dataset
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
        elif f in S:
            if x[f] <= tree.thresholds[tree_ind, node_ind]:
                return R(lc)
            else:
                return R(rc)
        else:
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
    import itertools
    import math
    import numpy as np

    m = len(x)
    phi = np.zeros(m)
    for p in itertools.permutations(list(range(m))):
        for i in range(m):
            phi[p[i]] += _conditional_expectation(tree, p[:i+1], x) - _conditional_expectation(tree, p[:i], x)
    return phi / math.factorial(m)

def test_xgboost_direct():
    try:
        import xgboost
    except:
        print("Skipping test_xgboost_direct!")
        return
    import shap
    import numpy as np

    N = 100
    M = 4
    X = np.random.randn(N,M)
    y = np.random.randn(N)  

    model = xgboost.XGBRegressor()
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    assert np.allclose(shap_values[0,:], _brute_force_tree_shap(explainer.model, X[0,:]))

def test_xgboost_multiclass():
    try:
        import xgboost
    except:
        print("Skipping test_xgboost_multiclass!")
        return
    import shap
    import numpy as np

    # train XGBoost model
    X, Y = shap.datasets.iris()
    model = xgboost.XGBClassifier(objective="binary:logistic", max_depth=4)
    model.fit(X, Y)

    # explain the model's predictions using SHAP values (use pred_contrib in LightGBM)
    shap_values = shap.TreeExplainer(model).shap_values(X)

    # ensure plot works for first class
    shap.dependence_plot(0, shap_values[0], X, show=False)

def _validate_shap_values(model, x_test):
    import shap
    import numpy as np

    # explain the model's predictions using SHAP values
    tree_explainer = shap.TreeExplainer(model)
    shap_values = tree_explainer.shap_values(x_test)
    expected_values = tree_explainer.expected_value
    # validate values sum to the margin prediction of the model plus expected_value
    assert(np.allclose(np.sum(shap_values, axis=1) + expected_values, model.predict(x_test)))

def test_xgboost_ranking():
    try:
        import xgboost
    except:
        print("Skipping test_xgboost_ranking!")
        return
    import shap
    import numpy as np

    # train lightgbm ranker model
    x_train, y_train, x_test, y_test, q_train, q_test = shap.datasets.rank()
    params = {'objective': 'rank:pairwise', 'learning_rate': 0.1,
              'gamma': 1.0, 'min_child_weight': 0.1,
              'max_depth': 4, 'n_estimators': 4}
    model = xgboost.sklearn.XGBRanker(**params)
    model.fit(x_train, y_train, q_train.astype(int),
              eval_set=[(x_test, y_test)], eval_group=[q_test.astype(int)])
    _validate_shap_values(model, x_test)

def test_xgboost_mixed_types():
    try:
        import xgboost
    except:
        print("Skipping test_xgboost_mixed_types!")
        return
    import shap
    import numpy as np

    X,y = shap.datasets.boston()
    X["LSTAT"] = X["LSTAT"].astype(np.int64)
    X["B"] = X["B"].astype(np.bool)
    bst = xgboost.train({"learning_rate": 0.01, "silent": 1}, xgboost.DMatrix(X, label=y), 1000)
    shap_values = shap.TreeExplainer(bst).shap_values(X)
    shap.dependence_plot(0, shap_values, X, show=False)

def test_ngboost():
    try:
        import ngboost
    except:
        print("Skipping test_ngboost!")
        return
    import shap
    import numpy as np

    X,y = shap.datasets.boston()
    model = ngboost.NGBRegressor(n_estimators=20).fit(X, y)
    explainer = shap.TreeExplainer(model, model_output=0)
    assert np.max(np.abs(explainer.shap_values(X).sum(1) + explainer.expected_value - model.predict(X))) < 1e-5

def test_pyspark_classifier_decision_tree():
    try:
        import pyspark
        import sklearn.datasets
        from pyspark.sql import SparkSession
        from pyspark import SparkContext, SparkConf
        from pyspark.ml.feature import VectorAssembler, StringIndexer
        from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier, GBTClassifier
        import pandas as pd
        import pickle

        iris_sk = sklearn.datasets.load_iris()
        iris = pd.DataFrame(data= np.c_[iris_sk['data'], iris_sk['target']], columns= iris_sk['feature_names'] + ['target'])[:100]
        spark = SparkSession.builder.config(conf=SparkConf().set("spark.master", "local[*]")).getOrCreate()
    except:
        print("Skipping test_pyspark_classifier_decision_tree!")
        return
    import shap
    import numpy as np

    col = ["sepal_length","sepal_width","petal_length","petal_width","type"]
    iris = spark.createDataFrame(iris, col)
    iris = VectorAssembler(inputCols=col[:-1],outputCol="features").transform(iris)
    iris = StringIndexer(inputCol="type", outputCol="label").fit(iris).transform(iris)

    classifiers = [GBTClassifier(labelCol="label", featuresCol="features"),
                   RandomForestClassifier(labelCol="label", featuresCol="features"),
                   DecisionTreeClassifier(labelCol="label", featuresCol="features")]
    for classifier in classifiers:
        model = classifier.fit(iris)
        explainer = shap.TreeExplainer(model)
        #Make sure the model can be serializable to run shap values with spark
        pickle.dumps(explainer)
        X = pd.DataFrame(data=iris_sk.data, columns=iris_sk.feature_names)[:100] # pylint: disable=E1101

        shap_values = explainer.shap_values(X)
        expected_values = explainer.expected_value

        predictions = model.transform(iris).select("rawPrediction")\
            .rdd.map(lambda x:[float(y) for y in x['rawPrediction']]).toDF(['class0','class1']).toPandas()

        if str(type(model)).endswith("GBTClassificationModel'>"):
            diffs = expected_values + shap_values.sum(1) - predictions.class1
            assert np.max(np.abs(diffs)) < 1e-4, "SHAP values don't sum to model output for class0!"
        else:
            normalizedPredictions = (predictions.T / predictions.sum(1)).T
            diffs = expected_values[0] + shap_values[0].sum(1) - normalizedPredictions.class0
            assert np.max(np.abs(diffs)) < 1e-4, "SHAP values don't sum to model output for class0!"+model
            diffs = expected_values[1] + shap_values[1].sum(1) - normalizedPredictions.class1
            assert np.max(np.abs(diffs)) < 1e-4, "SHAP values don't sum to model output for class1!"+model
            assert (np.abs(expected_values - normalizedPredictions.mean()) < 1e-1).all(), "Bad expected_value!"+model
    spark.stop()

def test_pyspark_regression_decision_tree():
    try:
        import pyspark
        import sklearn.datasets
        from pyspark.sql import SparkSession
        from pyspark import SparkContext, SparkConf
        from pyspark.ml.feature import VectorAssembler, StringIndexer
        from pyspark.ml.regression import DecisionTreeRegressor, GBTRegressor, RandomForestRegressor
        import pandas as pd

        iris_sk = sklearn.datasets.load_iris()
        iris = pd.DataFrame(data= np.c_[iris_sk['data'], iris_sk['target']], columns= iris_sk['feature_names'] + ['target'])[:100]
        spark = SparkSession.builder.config(conf=SparkConf().set("spark.master", "local[*]")).getOrCreate()
    except:
        print("Skipping test_pyspark_regression_decision_tree!")
        return
    import shap
    import numpy as np

    # Simple regressor: try to predict sepal length based on the other features
    col = ["sepal_length","sepal_width","petal_length","petal_width","type"]
    iris = spark.createDataFrame(iris, col).drop("type")
    iris = VectorAssembler(inputCols=col[1:-1],outputCol="features").transform(iris)

    regressors = [GBTRegressor(labelCol="sepal_length", featuresCol="features"),
                  RandomForestRegressor(labelCol="sepal_length", featuresCol="features"),
                  DecisionTreeRegressor(labelCol="sepal_length", featuresCol="features")]
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
    import shap
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np

    X, y = shap.datasets.iris()
    y[y == 2] = 1
    model = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    assert np.abs(shap_values[0][0,0] - 0.05) < 1e-3
    assert np.abs(shap_values[1][0,0] + 0.05) < 1e-3

def create_binary_newsgroups_data():
    from sklearn.datasets import fetch_20newsgroups

    categories = ['alt.atheism', 'soc.religion.christian']
    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
    class_names = ['atheism', 'christian']
    return newsgroups_train, newsgroups_test, class_names

def create_random_forest_vectorizer():
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.base import TransformerMixin

    vectorizer = CountVectorizer(lowercase=False, min_df=0.0, binary=True)

    class DenseTransformer(TransformerMixin):
        def fit(self, X, y=None, **fit_params):
            return self
        def transform(self, X, y=None, **fit_params):
            return X.toarray()

    rf = RandomForestClassifier(n_estimators=500, random_state=777)
    return Pipeline([('vectorizer', vectorizer), ('to_dense', DenseTransformer()), ('rf', rf)])

def test_sklearn_random_forest_newsgroups():
    import shap
    import numpy as np
    #from sklearn.ensemble import RandomForestClassifier

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
    import shap
    from sklearn.tree import DecisionTreeClassifier
    import numpy as np

    X, y = shap.datasets.iris()
    y[y == 2] = 1
    model = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    assert np.abs(shap_values[0][0,0] - 0.05) < 1e-1
    assert np.abs(shap_values[1][0,0] + 0.05) < 1e-1

def test_lightgbm():
    try:
        import lightgbm
    except:
        print("Skipping test_lightgbm!")
        return
    import shap
    import numpy as np

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
    try:
        import gpboost
    except:
        print("Skipping test_gpboost!")
        return
    import shap

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
    try:
        import catboost
        from catboost.datasets import amazon
    except:
        print("Skipping test_catboost!")
        return
    import shap
    import numpy as np

    # train catboost model
    X, y = shap.datasets.boston()
    X["RAD"] = X["RAD"].astype(np.int)
    model = catboost.CatBoostRegressor(iterations=300, learning_rate=0.1, random_seed=123)
    p = catboost.Pool(X, y, cat_features=["RAD"])
    model.fit(p, verbose=False, plot=False)

    # explain the model's predictions using SHAP values
    ex = shap.TreeExplainer(model)
    shap_values = ex.shap_values(p)

    predicted = model.predict(X)

    assert np.abs(shap_values.sum(1) + ex.expected_value - predicted).max() < 1e-4, \
        "SHAP values don't sum to model output!"
    
    train_df, _ = amazon()
    ix = 100
    X_train = train_df.drop('ACTION', axis=1)[:ix]
    y_train = train_df.ACTION[:ix]
    X_val = train_df.drop('ACTION', axis=1)[ix:ix+20]
    y_val = train_df.ACTION[ix:ix+20]
    model = catboost.CatBoostClassifier(iterations=100, learning_rate=0.5, random_seed=12)
    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),        
        verbose=False,
        plot=False
    )
    shap.TreeExplainer(model)

def test_catboost_categorical():
    try:
        import catboost
        from catboost.datasets import amazon
    except:
        print("Skipping test_catboost!")
        return
    import shap
    import pandas as pd
    from sklearn.datasets import load_boston
    import numpy as np

    bunch = load_boston()
    X, y = load_boston(True)
    X = pd.DataFrame(X, columns=bunch.feature_names) # pylint: disable=no-member
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
    try:
        import lightgbm
    except:
        print("Skipping test_lightgbm_constant_prediction!")
        return
    import shap
    import numpy as np

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
    try:
        import lightgbm
    except:
        print("Skipping test_lightgbm_constant_multiclass!")
        return
    import shap
    import numpy as np

    # train lightgbm model
    X, Y = shap.datasets.iris()
    Y.fill(1)
    model = lightgbm.sklearn.LGBMClassifier(num_classes=3, objective="multiclass")
    model.fit(X, Y)

    # explain the model's predictions using SHAP values
    shap.TreeExplainer(model).shap_values(X)

def test_lightgbm_multiclass():
    try:
        import lightgbm
    except:
        print("Skipping test_lightgbm_multiclass!")
        return
    import shap
    import numpy as np

    # train lightgbm model
    X, Y = shap.datasets.iris()
    model = lightgbm.sklearn.LGBMClassifier()
    model.fit(X, Y)

    # explain the model's predictions using SHAP values
    shap_values = shap.TreeExplainer(model).shap_values(X)

    # ensure plot works for first class
    shap.dependence_plot(0, shap_values[0], X, show=False)

def test_lightgbm_binary():
    try:
        import lightgbm
    except:
        print("Skipping test_lightgbm_binary!")
        return
    import shap
    from sklearn.model_selection import train_test_split

    # train lightgbm model
    X_train,X_test,Y_train,_ = train_test_split(*shap.datasets.adult(), test_size=0.2, random_state=0)
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
#     import shap
#     import numpy as np

#     # train lightgbm ranker model
#     x_train, y_train, x_test, y_test, q_train, q_test = shap.datasets.rank()
#     model = lightgbm.LGBMRanker()
#     model.fit(x_train, y_train, group=q_train, eval_set=[(x_test, y_test)],
#               eval_group=[q_test], eval_at=[1, 3], early_stopping_rounds=5, verbose=False,
#               callbacks=[lightgbm.reset_parameter(learning_rate=lambda x: 0.95 ** x * 0.1)])
#     _validate_shap_values(model, x_test)

# TODO: Test tree_limit argument

def test_sklearn_interaction():
    import sklearn
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    import shap

    # train a simple sklean RF model on the iris dataset
    X, _ = shap.datasets.iris()
    X_train,_,Y_train,_ = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)
    rforest = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0)
    model = rforest.fit(X_train, Y_train)

    # verify symmetry of the interaction values (this typically breaks if anything is wrong)
    interaction_vals = shap.TreeExplainer(model).shap_interaction_values(X)
    for i in range(len(interaction_vals)):
        for j in range(len(interaction_vals[i])):
            for k in range(len(interaction_vals[i][j])):
                for l in range(len(interaction_vals[i][j][k])):
                    assert abs(interaction_vals[i][j][k][l] - interaction_vals[i][j][l][k]) < 1e-4

    # ensure the interaction plot works
    shap.summary_plot(interaction_vals[0], X, show=False)

def test_lightgbm_interaction():
    try:
        import lightgbm
    except:
        print("Skipping test_lightgbm_interaction!")
        return
    import shap
    import numpy as np

    # train XGBoost model
    X, y = shap.datasets.boston()
    model = lightgbm.sklearn.LGBMRegressor()
    model.fit(X, y)

    # verify symmetry of the interaction values (this typically breaks if anything is wrong)
    interaction_vals = shap.TreeExplainer(model).shap_interaction_values(X)
    for j in range(len(interaction_vals)):
        for k in range(len(interaction_vals[j])):
            for l in range(len(interaction_vals[j][k])):
                assert abs(interaction_vals[j][k][l] - interaction_vals[j][l][k]) < 1e-4

def test_sum_match_random_forest():
    import shap
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    import sklearn

    X_train,X_test,Y_train,_ = train_test_split(*shap.datasets.adult(), test_size=0.2, random_state=0)
    clf = RandomForestClassifier(random_state=202, n_estimators=10, max_depth=10)
    clf.fit(X_train, Y_train)
    predicted = clf.predict_proba(X_test)
    ex = shap.TreeExplainer(clf)
    shap_values = ex.shap_values(X_test)
    assert np.abs(shap_values[0].sum(1) + ex.expected_value[0] - predicted[:,0]).max() < 1e-4, \
        "SHAP values don't sum to model output!"
    
def test_sum_match_extra_trees():
    import shap
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import ExtraTreesRegressor
    import sklearn

    X_train,X_test,Y_train,_ = train_test_split(*shap.datasets.adult(), test_size=0.2, random_state=0)
    clf = ExtraTreesRegressor(random_state=202, n_estimators=10, max_depth=10)
    clf.fit(X_train, Y_train)
    predicted = clf.predict(X_test)
    ex = shap.TreeExplainer(clf)
    shap_values = ex.shap_values(X_test)
    assert np.abs(shap_values.sum(1) + ex.expected_value - predicted).max() < 1e-4, \
        "SHAP values don't sum to model output!"

def test_single_row_random_forest():
    import shap
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    import sklearn

    X_train,X_test,Y_train,_ = train_test_split(*shap.datasets.adult(), test_size=0.2, random_state=0)
    clf = RandomForestClassifier(random_state=202, n_estimators=10, max_depth=10)
    clf.fit(X_train, Y_train)
    predicted = clf.predict_proba(X_test)
    ex = shap.TreeExplainer(clf)
    shap_values = ex.shap_values(X_test.iloc[0,:])
    assert np.abs(shap_values[0].sum() + ex.expected_value[0] - predicted[0,0]) < 1e-4, \
        "SHAP values don't sum to model output!"

def test_sum_match_gradient_boosting_classifier():
    import shap
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingClassifier
    import sklearn

    X_train,X_test,Y_train,_ = train_test_split(*shap.datasets.adult(), test_size=0.2, random_state=0)
    clf = GradientBoostingClassifier(random_state=202, n_estimators=10, max_depth=10)
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
    shap_interaction_values = ex.shap_interaction_values(X_test.iloc[:10,:])
    assert np.abs(shap_interaction_values.sum(1).sum(1) + ex.expected_value - predicted[:10]).max() < 1e-4, \
        "SHAP interaction values don't sum to model output!"

def test_sum_match_adaboost_classifier(): 
    import shap 
    import numpy as np 
    from sklearn.model_selection import train_test_split 
    from sklearn.ensemble import AdaBoostClassifier 
    import sklearn 

    X_train,X_test,Y_train,Y_test = train_test_split(*shap.datasets.adult(), test_size=0.2, random_state=0) 
    clf = AdaBoostClassifier(random_state=202, n_estimators=10) 
    clf.fit(X_train, Y_train) 
    predicted = clf.predict_proba(X_test) 
    ex = shap.TreeExplainer(clf) 
    shap_values = ex.shap_values(X_test) 
    assert np.abs(shap_values[0].sum(1) + ex.expected_value[0] - predicted[:,0]).max() < 1e-4, \
        "SHAP values don't sum to model output!" 
    
def test_single_row_gradient_boosting_classifier():
    import shap
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingClassifier
    import sklearn

    X_train,X_test,Y_train,_ = train_test_split(*shap.datasets.adult(), test_size=0.2, random_state=0)
    clf = GradientBoostingClassifier(random_state=202, n_estimators=10, max_depth=10)
    clf.fit(X_train, Y_train)
    predicted = clf.decision_function(X_test)
    ex = shap.TreeExplainer(clf)
    shap_values = ex.shap_values(X_test.iloc[0,:])
    assert np.abs(shap_values.sum() + ex.expected_value - predicted[0]) < 1e-4, \
        "SHAP values don't sum to model output!"

def test_HistGradientBoostingRegressor():
    from sklearn.experimental import enable_hist_gradient_boosting
    from sklearn.ensemble import HistGradientBoostingRegressor
    import numpy as np
    import shap

    # train a tree-based model
    X, y = shap.datasets.diabetes()
    model = HistGradientBoostingRegressor(max_iter=1000, max_depth=6).fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    assert np.max(np.abs(shap_values.sum(1) + explainer.expected_value - model.predict(X))) < 1e-4

def test_HistGradientBoostingClassifier_proba():
    from sklearn.experimental import enable_hist_gradient_boosting
    from sklearn.ensemble import HistGradientBoostingClassifier
    import numpy as np
    import shap

    # train a tree-based model
    X, y = shap.datasets.adult()
    model = HistGradientBoostingClassifier(max_iter=10, max_depth=6).fit(X, y)
    explainer = shap.TreeExplainer(model, shap.sample(X, 10), model_output="predict_proba")
    shap_values = explainer.shap_values(X)
    assert np.max(np.abs(shap_values[0].sum(1) + explainer.expected_value[0] - model.predict_proba(X)[:,0])) < 1e-4

def test_HistGradientBoostingClassifier_multidim():
    from sklearn.experimental import enable_hist_gradient_boosting
    from sklearn.ensemble import HistGradientBoostingClassifier
    import numpy as np
    import shap

    # train a tree-based model
    X, y = shap.datasets.adult()
    X = X[:100]
    y = y[:100]
    y = np.random.randint(0, 3, len(y))
    model = HistGradientBoostingClassifier(max_iter=10, max_depth=6).fit(X, y)
    explainer = shap.TreeExplainer(model, shap.sample(X, 10), model_output="raw")
    shap_values = explainer.shap_values(X)
    assert np.max(np.abs(shap_values[0].sum(1) + explainer.expected_value[0] - model.decision_function(X)[:,0])) < 1e-4

def test_sum_match_gradient_boosting_regressor():
    import shap
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingRegressor
    import sklearn

    X_train,X_test,Y_train,_ = train_test_split(*shap.datasets.adult(), test_size=0.2, random_state=0)
    clf = GradientBoostingRegressor(random_state=202, n_estimators=10, max_depth=10)
    clf.fit(X_train, Y_train)

    predicted = clf.predict(X_test)
    ex = shap.TreeExplainer(clf)
    shap_values = ex.shap_values(X_test)
    assert np.abs(shap_values.sum(1) + ex.expected_value - predicted).max() < 1e-4, \
        "SHAP values don't sum to model output!"

def test_single_row_gradient_boosting_regressor():
    import shap
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingRegressor
    import sklearn

    X_train,X_test,Y_train,_ = train_test_split(*shap.datasets.adult(), test_size=0.2, random_state=0)
    clf = GradientBoostingRegressor(random_state=202, n_estimators=10, max_depth=10)
    clf.fit(X_train, Y_train)
    
    predicted = clf.predict(X_test)
    ex = shap.TreeExplainer(clf)
    shap_values = ex.shap_values(X_test.iloc[0,:])
    assert np.abs(shap_values.sum() + ex.expected_value - predicted[0]) < 1e-4, \
        "SHAP values don't sum to model output!"


def test_multi_target_random_forest():
    import shap
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor

    X_train, X_test, Y_train, _ = train_test_split(*shap.datasets.linnerud(), test_size=0.2, random_state=0)
    est = RandomForestRegressor(random_state=202, n_estimators=10, max_depth=10)
    est.fit(X_train, Y_train)
    predicted = est.predict(X_test)

    explainer = shap.TreeExplainer(est)
    expected_values = np.asarray(explainer.expected_value)
    assert len(expected_values) == est.n_outputs_, "Length of expected_values doesn't match n_outputs_"
    shap_values = np.asarray(explainer.shap_values(X_test)).reshape(est.n_outputs_ * X_test.shape[0], X_test.shape[1])
    phi = np.hstack((shap_values, np.repeat(expected_values, X_test.shape[0]).reshape(-1, 1)))
    assert np.allclose(phi.sum(1), predicted.flatten(order="F"), atol=1e-4)


def test_isolation_forest():
    import shap
    import numpy as np
    from sklearn.ensemble import IsolationForest
    from sklearn.ensemble.iforest import _average_path_length

    X,_ = shap.datasets.boston()
    for max_features in [1.0, 0.75]:
        iso = IsolationForest(max_features=max_features)
        iso.fit(X)

        explainer = shap.TreeExplainer(iso)
        shap_values = explainer.shap_values(X)

        score_from_shap = - 2**(
            - (np.sum(shap_values, axis=1) + explainer.expected_value) /
            _average_path_length(np.array([iso.max_samples_]))[0]
            )
        assert np.allclose(iso.score_samples(X), score_from_shap, atol=1e-7)


# TODO: this has sometimes failed with strange answers, should run memcheck on this for any memory issues at some point...
def test_multi_target_extra_trees():
    import shap
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import ExtraTreesRegressor

    X_train, X_test, Y_train, _ = train_test_split(*shap.datasets.linnerud(), test_size=0.2, random_state=0)
    est = ExtraTreesRegressor(random_state=202, n_estimators=10, max_depth=10)
    est.fit(X_train, Y_train)
    predicted = est.predict(X_test)

    explainer = shap.TreeExplainer(est)
    expected_values = np.asarray(explainer.expected_value)
    assert len(expected_values) == est.n_outputs_, "Length of expected_values doesn't match n_outputs_"
    shap_values = np.asarray(explainer.shap_values(X_test)).reshape(est.n_outputs_ * X_test.shape[0], X_test.shape[1])
    phi = np.hstack((shap_values, np.repeat(expected_values, X_test.shape[0]).reshape(-1, 1)))
    assert np.allclose(phi.sum(1), predicted.flatten(order="F"), atol=1e-4)


def test_provided_background_tree_path_dependent():
    try:
        import xgboost
    except:
        print("Skipping test_provided_background_tree_path_dependent!")
        return

    from sklearn.model_selection import train_test_split
    import numpy as np
    import shap
    np.random.seed(10)

    X,y = shap.datasets.iris()
    X = X[:100]
    y = y[:100]
    train_x, test_x, train_y, _ = train_test_split(X, y, random_state=1)
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
    diffs = explainer.expected_value + explainer.shap_values(test_x).sum(1) - bst.predict(dtest, output_margin=True)
    assert np.max(np.abs(diffs)) < 1e-4, "SHAP values don't sum to model output!"
    assert np.abs(explainer.expected_value - bst.predict(dtest, output_margin=True).mean()) < 1e-6, "Bad expected_value!"

def test_provided_background_independent():
    try:
        import xgboost
    except:
        print("Skipping test_provided_background_independent!")
        return

    from sklearn.model_selection import train_test_split
    import numpy as np
    import shap
    np.random.seed(10)

    X,y = shap.datasets.iris()
    X = X[:100]
    y = y[:100]
    train_x, test_x, train_y, _ = train_test_split(X, y, random_state=1)
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
    diffs = explainer.expected_value + explainer.shap_values(test_x).sum(1) - bst.predict(dtest, output_margin=True)
    assert np.max(np.abs(diffs)) < 1e-4, "SHAP values don't sum to model output!"
    assert np.abs(explainer.expected_value - bst.predict(dtest, output_margin=True).mean()) < 1e-4, "Bad expected_value!"

def test_provided_background_independent_prob_output():
    try:
        import xgboost
    except:
        print("Skipping test_provided_background_independent_prob_output!")
        return

    from sklearn.model_selection import train_test_split
    import numpy as np
    import shap
    np.random.seed(10)

    X,y = shap.datasets.iris()
    X = X[:100]
    y = y[:100]
    train_x, test_x, train_y, _ = train_test_split(X, y, random_state=1)
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

        explainer = shap.TreeExplainer(bst, test_x, feature_perturbation="interventional", model_output="probability")
        diffs = explainer.expected_value + explainer.shap_values(test_x).sum(1) - bst.predict(dtest)
        assert np.max(np.abs(diffs)) < 1e-4, "SHAP values don't sum to model output!"
        assert np.abs(explainer.expected_value - bst.predict(dtest).mean()) < 1e-4, "Bad expected_value!"

def test_single_tree_compare_with_kernel_shap():
    """ Compare with Kernel SHAP, which makes the same independence assumptions
    as Independent Tree SHAP.  Namely, they both assume independence between the 
    set being conditioned on, and the remainder set.
    """
    try:
        import xgboost
    except:
        print("Skipping test_single_tree_compare_with_kernel_shap!")
        return
    import numpy as np
    import shap
    np.random.seed(10)

    n = 100
    X = np.random.normal(size=(n,7))
    b = np.array([-2,1,3,5,2,20,-5])
    y = np.matmul(X,b)
    max_depth = 6

    # train a model with single tree
    Xd = xgboost.DMatrix(X, label=y)
    model = xgboost.train({'eta':1, 
                       'max_depth':max_depth, 
                       'base_score': 0, 
                       "lambda": 0}, 
                      Xd, 1)
    ypred = model.predict(Xd)

    # Compare for five random samples
    for _ in range(5):
        x_ind = np.random.choice(X.shape[1]); x = X[x_ind:x_ind+1,:]

        expl = shap.TreeExplainer(model, X, feature_perturbation="interventional")
        f = lambda inp : model.predict(xgboost.DMatrix(inp))
        expl_kern = shap.KernelExplainer(f, X)

        itshap = expl.shap_values(x)
        kshap = expl_kern.shap_values(x, nsamples=150)
        assert np.allclose(itshap,kshap), \
        "Kernel SHAP doesn't match Independent Tree SHAP!"
        assert np.allclose(itshap.sum() + expl.expected_value, ypred[x_ind]), \
        "SHAP values don't sum to model output!"    

def test_several_trees():
    """ Make sure Independent Tree SHAP sums up to the correct value for
    larger models (20 trees).
    """    
    try:
        import xgboost
    except:
        print("Skipping test_several_trees!")
        return
    import shap
    import numpy as np
    np.random.seed(10)

    n = 1000
    X = np.random.normal(size=(n,7))
    b = np.array([-2,1,3,5,2,20,-5])
    y = np.matmul(X,b)
    max_depth = 6

    # train a model with single tree
    Xd = xgboost.DMatrix(X, label=y)
    model = xgboost.train({'eta':1, 
                       'max_depth':max_depth, 
                       'base_score': 0, 
                       "lambda": 0}, 
                      Xd, 20)
    ypred = model.predict(Xd)

    # Compare for five random samples
    for _ in range(5):
        x_ind = np.random.choice(X.shape[1]); x = X[x_ind:x_ind+1,:]
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

    try:
        import xgboost
    except:
        print("Skipping test_several_trees!")
        return
    import shap
    import numpy as np
    np.random.seed(10)

    n = 100
    X = np.random.normal(size=(n,7))
    b = np.array([-2,1,3,5,2,20,-5])
    y = np.matmul(X,b)
    y = y + abs(min(y))
    y = np.random.binomial(n=1,p=y/max(y))
    max_depth = 6

    # train a model with single tree
    Xd = xgboost.DMatrix(X, label=y)
    model = xgboost.train({'eta':1, 
                       'max_depth':max_depth, 
                       'base_score': y.mean(), 
                       "lambda": 0,
                       "objective": "binary:logistic"}, 
                      Xd, 1)
    pred = model.predict(Xd,output_margin=True) # In margin space (log odds)
    trans_pred = model.predict(Xd) # In probability space

    expl = shap.TreeExplainer(model, X, feature_perturbation="interventional")
    f = lambda inp : model.predict(xgboost.DMatrix(inp), output_margin=True)
    expl_kern = shap.KernelExplainer(f, X)

    x_ind = 0; x = X[x_ind:x_ind+1,:]
    itshap = expl.shap_values(x)
    kshap = expl_kern.shap_values(x, nsamples=300)
    assert np.allclose(itshap.sum() + expl.expected_value, pred[x_ind]), \
    "SHAP values don't sum to model output on explaining margin!"
    assert np.allclose(itshap, kshap), \
    "Independent Tree SHAP doesn't match Kernel SHAP on explaining margin!"

    model.set_attr(objective="binary:logistic")
    expl = shap.TreeExplainer(model, X, feature_perturbation="interventional", model_output="probability")
    itshap = expl.shap_values(x)
    assert np.allclose(itshap.sum() + expl.expected_value, trans_pred[x_ind]), \
    "SHAP values don't sum to model output on explaining logistic!"

    # expl = shap.TreeExplainer(model, X, feature_perturbation="interventional", model_output="logloss")
    # itshap = expl.shap_values(x,y=y[x_ind])
    # margin_pred = model.predict(xgb.DMatrix(x),output_margin=True)
    # currpred = log_loss(y[x_ind],sigmoid(margin_pred))
    # assert np.allclose(itshap.sum(), currpred - expl.expected_value), \
    # "SHAP values don't sum to model output on explaining logloss!"

def test_xgboost_classifier_independent_margin():
    try:
        import xgboost
    except:
        print("Skipping test_several_trees!")
        return
    import numpy as np
    import shap

    # train XGBoost model
    np.random.seed(10)
    n = 1000
    X = np.random.normal(size=(n,7))
    b = np.array([-2,1,3,5,2,20,-5])
    y = np.matmul(X,b)
    y = y + abs(min(y))
    y = np.random.binomial(n=1,p=y/max(y))

    model = xgboost.XGBClassifier(n_estimators=10, max_depth=5)
    model.fit(X, y)

    # explain the model's predictions using SHAP values
    e = shap.TreeExplainer(model, X, feature_perturbation="interventional", model_output="margin")
    shap_values = e.shap_values(X)

    assert np.allclose(shap_values.sum(1) + e.expected_value, model.predict(X, output_margin=True))


def test_xgboost_classifier_independent_probability():
    try:
        import xgboost
    except:
        print("Skipping test_several_trees!")
        return
    import shap
    import numpy as np

    # train XGBoost model
    np.random.seed(10)
    n = 1000
    X = np.random.normal(size=(n,7))
    b = np.array([-2,1,3,5,2,20,-5])
    y = np.matmul(X,b)
    y = y + abs(min(y))
    y = np.random.binomial(n=1,p=y/max(y))

    model = xgboost.XGBClassifier(n_estimators=10, max_depth=5)
    model.fit(X, y)

    # explain the model's predictions using SHAP values
    e = shap.TreeExplainer(model, X, feature_perturbation="interventional", model_output="probability")
    shap_values = e.shap_values(X)

    assert np.allclose(shap_values.sum(1) + e.expected_value, model.predict_proba(X)[:,1])

# def test_front_page_xgboost_global_path_dependent():
#     try:
#         import xgboost
#     except:
#         print("Skipping test_front_page_xgboost!")
#         return
#     import shap
#     import numpy as np

#     # train XGBoost model
#     X, y = shap.datasets.boston()
#     model = xgboost.XGBRegressor()
#     model.fit(X, y)

#     # explain the model's predictions using SHAP values
#     explainer = shap.TreeExplainer(model, X, feature_perturbation="global_path_dependent")
#     shap_values = explainer.shap_values(X)

#     assert np.allclose(shap_values.sum(1) + explainer.expected_value, model.predict(X))

def test_skopt_rf_et():
    try:
        import skopt
        import pandas as pd
    except:
        print("Skipping test_skopt_rf_et!")
        return
    import shap
    import numpy as np

    # Define an objective function for skopt to optimise.
    def objective_function(x):
        return x[0]**2 - x[1]**2 + x[1]*x[0]

    # Uneven bounds to prevent "objective has been evaluated" warnings.
    problem_bounds = [(-1e6, 3e6), (-1e6, 3e6)]

    # Don't worry about "objective has been evaluated" warnings.
    result_et = skopt.forest_minimize(objective_function, problem_bounds, n_calls = 100, base_estimator = "ET")
    result_rf = skopt.forest_minimize(objective_function, problem_bounds, n_calls = 100, base_estimator = "RF")

    et_df = pd.DataFrame(result_et.x_iters, columns = ["X0", "X1"])

    # Explain the model's predictions.
    explainer_et = shap.TreeExplainer(result_et.models[-1], et_df)
    shap_values_et = explainer_et.shap_values(et_df)

    rf_df = pd.DataFrame(result_rf.x_iters, columns = ["X0", "X1"])

    # Explain the model's predictions (Random forest).
    explainer_rf = shap.TreeExplainer(result_rf.models[-1], rf_df)
    shap_values_rf = explainer_rf.shap_values(rf_df)

    assert np.allclose(shap_values_et.sum(1) + explainer_et.expected_value, result_et.models[-1].predict(et_df))
    assert np.allclose(shap_values_rf.sum(1) + explainer_rf.expected_value, result_rf.models[-1].predict(rf_df))

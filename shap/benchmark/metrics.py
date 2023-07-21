import hashlib
import os
import time

import numpy as np
import sklearn

from .. import __version__
from . import measures, methods

try:
    import dill as pickle
except Exception:
    pass

try:
    from sklearn.model_selection import train_test_split
except Exception:
    from sklearn.cross_validation import train_test_split


def runtime(X, y, model_generator, method_name):
    """ Runtime (sec / 1k samples)
    transform = "negate_log"
    sort_order = 2
    """

    old_seed = np.random.seed()
    np.random.seed(3293)

    # average the method scores over several train/test splits
    method_reps = []
    for i in range(3):
        X_train, X_test, y_train, _ = train_test_split(__toarray(X), y, test_size=100, random_state=i)

        # define the model we are going to explain
        model = model_generator()
        model.fit(X_train, y_train)

        # evaluate each method
        start = time.time()
        explainer = getattr(methods, method_name)(model, X_train)
        build_time = time.time() - start

        start = time.time()
        explainer(X_test)
        explain_time = time.time() - start

        # we always normalize the explain time as though we were explaining 1000 samples
        # even if to reduce the runtime of the benchmark we do less (like just 100)
        method_reps.append(build_time + explain_time * 1000.0 / X_test.shape[0])
    np.random.seed(old_seed)

    return None, np.mean(method_reps)

def local_accuracy(X, y, model_generator, method_name):
    """ Local Accuracy
    transform = "identity"
    sort_order = 0
    """

    def score_map(true, pred):
        """ Computes local accuracy as the normalized standard deviation of numerical scores.
        """
        return np.std(pred - true) / (np.std(true) + 1e-6)

    def score_function(X_train, X_test, y_train, y_test, attr_function, trained_model, random_state):
        return measures.local_accuracy(
            X_train, y_train, X_test, y_test, attr_function(X_test),
            model_generator, score_map, trained_model
        )
    return None, __score_method(X, y, None, model_generator, score_function, method_name)

def consistency_guarantees(X, y, model_generator, method_name):
    """ Consistency Guarantees
    transform = "identity"
    sort_order = 1
    """

    # 1.0 - perfect consistency
    # 0.8 - guarantees depend on sampling
    # 0.6 - guarantees depend on approximation
    # 0.0 - no garuntees
    guarantees = {
        "linear_shap_corr": 1.0,
        "linear_shap_ind": 1.0,
        "coef": 0.0,
        "kernel_shap_1000_meanref": 0.8,
        "sampling_shap_1000": 0.8,
        "random": 0.0,
        "saabas": 0.0,
        "tree_gain": 0.0,
        "tree_shap_tree_path_dependent": 1.0,
        "tree_shap_independent_200": 1.0,
        "mean_abs_tree_shap": 1.0,
        "lime_tabular_regression_1000": 0.8,
        "lime_tabular_classification_1000": 0.8,
        "maple": 0.8,
        "tree_maple": 0.8,
        "deep_shap": 0.6,
        "expected_gradients": 0.6
    }

    return None, guarantees[method_name]

def __mean_pred(true, pred):
    """ A trivial metric that is just is the output of the model.
    """
    return np.mean(pred)

def keep_positive_mask(X, y, model_generator, method_name, num_fcounts=11):
    """ Keep Positive (mask)
    xlabel = "Max fraction of features kept"
    ylabel = "Mean model output"
    transform = "identity"
    sort_order = 4
    """
    return __run_measure(measures.keep_mask, X, y, model_generator, method_name, 1, num_fcounts, __mean_pred)

def keep_negative_mask(X, y, model_generator, method_name, num_fcounts=11):
    """ Keep Negative (mask)
    xlabel = "Max fraction of features kept"
    ylabel = "Negative mean model output"
    transform = "negate"
    sort_order = 5
    """
    return __run_measure(measures.keep_mask, X, y, model_generator, method_name, -1, num_fcounts, __mean_pred)

def keep_absolute_mask__r2(X, y, model_generator, method_name, num_fcounts=11):
    """ Keep Absolute (mask)
    xlabel = "Max fraction of features kept"
    ylabel = "R^2"
    transform = "identity"
    sort_order = 6
    """
    return __run_measure(measures.keep_mask, X, y, model_generator, method_name, 0, num_fcounts, sklearn.metrics.r2_score)

def keep_absolute_mask__roc_auc(X, y, model_generator, method_name, num_fcounts=11):
    """ Keep Absolute (mask)
    xlabel = "Max fraction of features kept"
    ylabel = "ROC AUC"
    transform = "identity"
    sort_order = 6
    """
    return __run_measure(measures.keep_mask, X, y, model_generator, method_name, 0, num_fcounts, sklearn.metrics.roc_auc_score)

def remove_positive_mask(X, y, model_generator, method_name, num_fcounts=11):
    """ Remove Positive (mask)
    xlabel = "Max fraction of features removed"
    ylabel = "Negative mean model output"
    transform = "negate"
    sort_order = 7
    """
    return __run_measure(measures.remove_mask, X, y, model_generator, method_name, 1, num_fcounts, __mean_pred)

def remove_negative_mask(X, y, model_generator, method_name, num_fcounts=11):
    """ Remove Negative (mask)
    xlabel = "Max fraction of features removed"
    ylabel = "Mean model output"
    transform = "identity"
    sort_order = 8
    """
    return __run_measure(measures.remove_mask, X, y, model_generator, method_name, -1, num_fcounts, __mean_pred)

def remove_absolute_mask__r2(X, y, model_generator, method_name, num_fcounts=11):
    """ Remove Absolute (mask)
    xlabel = "Max fraction of features removed"
    ylabel = "1 - R^2"
    transform = "one_minus"
    sort_order = 9
    """
    return __run_measure(measures.remove_mask, X, y, model_generator, method_name, 0, num_fcounts, sklearn.metrics.r2_score)

def remove_absolute_mask__roc_auc(X, y, model_generator, method_name, num_fcounts=11):
    """ Remove Absolute (mask)
    xlabel = "Max fraction of features removed"
    ylabel = "1 - ROC AUC"
    transform = "one_minus"
    sort_order = 9
    """
    return __run_measure(measures.remove_mask, X, y, model_generator, method_name, 0, num_fcounts, sklearn.metrics.roc_auc_score)

def keep_positive_resample(X, y, model_generator, method_name, num_fcounts=11):
    """ Keep Positive (resample)
    xlabel = "Max fraction of features kept"
    ylabel = "Mean model output"
    transform = "identity"
    sort_order = 10
    """
    return __run_measure(measures.keep_resample, X, y, model_generator, method_name, 1, num_fcounts, __mean_pred)

def keep_negative_resample(X, y, model_generator, method_name, num_fcounts=11):
    """ Keep Negative (resample)
    xlabel = "Max fraction of features kept"
    ylabel = "Negative mean model output"
    transform = "negate"
    sort_order = 11
    """
    return __run_measure(measures.keep_resample, X, y, model_generator, method_name, -1, num_fcounts, __mean_pred)

def keep_absolute_resample__r2(X, y, model_generator, method_name, num_fcounts=11):
    """ Keep Absolute (resample)
    xlabel = "Max fraction of features kept"
    ylabel = "R^2"
    transform = "identity"
    sort_order = 12
    """
    return __run_measure(measures.keep_resample, X, y, model_generator, method_name, 0, num_fcounts, sklearn.metrics.r2_score)

def keep_absolute_resample__roc_auc(X, y, model_generator, method_name, num_fcounts=11):
    """ Keep Absolute (resample)
    xlabel = "Max fraction of features kept"
    ylabel = "ROC AUC"
    transform = "identity"
    sort_order = 12
    """
    return __run_measure(measures.keep_resample, X, y, model_generator, method_name, 0, num_fcounts, sklearn.metrics.roc_auc_score)

def remove_positive_resample(X, y, model_generator, method_name, num_fcounts=11):
    """ Remove Positive (resample)
    xlabel = "Max fraction of features removed"
    ylabel = "Negative mean model output"
    transform = "negate"
    sort_order = 13
    """
    return __run_measure(measures.remove_resample, X, y, model_generator, method_name, 1, num_fcounts, __mean_pred)

def remove_negative_resample(X, y, model_generator, method_name, num_fcounts=11):
    """ Remove Negative (resample)
    xlabel = "Max fraction of features removed"
    ylabel = "Mean model output"
    transform = "identity"
    sort_order = 14
    """
    return __run_measure(measures.remove_resample, X, y, model_generator, method_name, -1, num_fcounts, __mean_pred)

def remove_absolute_resample__r2(X, y, model_generator, method_name, num_fcounts=11):
    """ Remove Absolute (resample)
    xlabel = "Max fraction of features removed"
    ylabel = "1 - R^2"
    transform = "one_minus"
    sort_order = 15
    """
    return __run_measure(measures.remove_resample, X, y, model_generator, method_name, 0, num_fcounts, sklearn.metrics.r2_score)

def remove_absolute_resample__roc_auc(X, y, model_generator, method_name, num_fcounts=11):
    """ Remove Absolute (resample)
    xlabel = "Max fraction of features removed"
    ylabel = "1 - ROC AUC"
    transform = "one_minus"
    sort_order = 15
    """
    return __run_measure(measures.remove_resample, X, y, model_generator, method_name, 0, num_fcounts, sklearn.metrics.roc_auc_score)

def keep_positive_impute(X, y, model_generator, method_name, num_fcounts=11):
    """ Keep Positive (impute)
    xlabel = "Max fraction of features kept"
    ylabel = "Mean model output"
    transform = "identity"
    sort_order = 16
    """
    return __run_measure(measures.keep_impute, X, y, model_generator, method_name, 1, num_fcounts, __mean_pred)

def keep_negative_impute(X, y, model_generator, method_name, num_fcounts=11):
    """ Keep Negative (impute)
    xlabel = "Max fraction of features kept"
    ylabel = "Negative mean model output"
    transform = "negate"
    sort_order = 17
    """
    return __run_measure(measures.keep_impute, X, y, model_generator, method_name, -1, num_fcounts, __mean_pred)

def keep_absolute_impute__r2(X, y, model_generator, method_name, num_fcounts=11):
    """ Keep Absolute (impute)
    xlabel = "Max fraction of features kept"
    ylabel = "R^2"
    transform = "identity"
    sort_order = 18
    """
    return __run_measure(measures.keep_impute, X, y, model_generator, method_name, 0, num_fcounts, sklearn.metrics.r2_score)

def keep_absolute_impute__roc_auc(X, y, model_generator, method_name, num_fcounts=11):
    """ Keep Absolute (impute)
    xlabel = "Max fraction of features kept"
    ylabel = "ROC AUC"
    transform = "identity"
    sort_order = 19
    """
    return __run_measure(measures.keep_mask, X, y, model_generator, method_name, 0, num_fcounts, sklearn.metrics.roc_auc_score)

def remove_positive_impute(X, y, model_generator, method_name, num_fcounts=11):
    """ Remove Positive (impute)
    xlabel = "Max fraction of features removed"
    ylabel = "Negative mean model output"
    transform = "negate"
    sort_order = 7
    """
    return __run_measure(measures.remove_impute, X, y, model_generator, method_name, 1, num_fcounts, __mean_pred)

def remove_negative_impute(X, y, model_generator, method_name, num_fcounts=11):
    """ Remove Negative (impute)
    xlabel = "Max fraction of features removed"
    ylabel = "Mean model output"
    transform = "identity"
    sort_order = 8
    """
    return __run_measure(measures.remove_impute, X, y, model_generator, method_name, -1, num_fcounts, __mean_pred)

def remove_absolute_impute__r2(X, y, model_generator, method_name, num_fcounts=11):
    """ Remove Absolute (impute)
    xlabel = "Max fraction of features removed"
    ylabel = "1 - R^2"
    transform = "one_minus"
    sort_order = 9
    """
    return __run_measure(measures.remove_impute, X, y, model_generator, method_name, 0, num_fcounts, sklearn.metrics.r2_score)

def remove_absolute_impute__roc_auc(X, y, model_generator, method_name, num_fcounts=11):
    """ Remove Absolute (impute)
    xlabel = "Max fraction of features removed"
    ylabel = "1 - ROC AUC"
    transform = "one_minus"
    sort_order = 9
    """
    return __run_measure(measures.remove_mask, X, y, model_generator, method_name, 0, num_fcounts, sklearn.metrics.roc_auc_score)

def keep_positive_retrain(X, y, model_generator, method_name, num_fcounts=11):
    """ Keep Positive (retrain)
    xlabel = "Max fraction of features kept"
    ylabel = "Mean model output"
    transform = "identity"
    sort_order = 6
    """
    return __run_measure(measures.keep_retrain, X, y, model_generator, method_name, 1, num_fcounts, __mean_pred)

def keep_negative_retrain(X, y, model_generator, method_name, num_fcounts=11):
    """ Keep Negative (retrain)
    xlabel = "Max fraction of features kept"
    ylabel = "Negative mean model output"
    transform = "negate"
    sort_order = 7
    """
    return __run_measure(measures.keep_retrain, X, y, model_generator, method_name, -1, num_fcounts, __mean_pred)

def remove_positive_retrain(X, y, model_generator, method_name, num_fcounts=11):
    """ Remove Positive (retrain)
    xlabel = "Max fraction of features removed"
    ylabel = "Negative mean model output"
    transform = "negate"
    sort_order = 11
    """
    return __run_measure(measures.remove_retrain, X, y, model_generator, method_name, 1, num_fcounts, __mean_pred)

def remove_negative_retrain(X, y, model_generator, method_name, num_fcounts=11):
    """ Remove Negative (retrain)
    xlabel = "Max fraction of features removed"
    ylabel = "Mean model output"
    transform = "identity"
    sort_order = 12
    """
    return __run_measure(measures.remove_retrain, X, y, model_generator, method_name, -1, num_fcounts, __mean_pred)

def __run_measure(measure, X, y, model_generator, method_name, attribution_sign, num_fcounts, summary_function):

    def score_function(fcount, X_train, X_test, y_train, y_test, attr_function, trained_model, random_state):
        if attribution_sign == 0:
            A = np.abs(__strip_list(attr_function(X_test)))
        else:
            A = attribution_sign * __strip_list(attr_function(X_test))
        nmask = np.ones(len(y_test)) * fcount
        nmask = np.minimum(nmask, np.array(A >= 0).sum(1)).astype(int)
        return measure(
            nmask, X_train, y_train, X_test, y_test, A,
            model_generator, summary_function, trained_model, random_state
        )
    fcounts = __intlogspace(0, X.shape[1], num_fcounts)
    return fcounts, __score_method(X, y, fcounts, model_generator, score_function, method_name)

def batch_remove_absolute_retrain__r2(X, y, model_generator, method_name, num_fcounts=11):
    """ Batch Remove Absolute (retrain)
    xlabel = "Fraction of features removed"
    ylabel = "1 - R^2"
    transform = "one_minus"
    sort_order = 13
    """
    return __run_batch_abs_metric(measures.batch_remove_retrain, X, y, model_generator, method_name, sklearn.metrics.r2_score, num_fcounts)

def batch_keep_absolute_retrain__r2(X, y, model_generator, method_name, num_fcounts=11):
    """ Batch Keep Absolute (retrain)
    xlabel = "Fraction of features kept"
    ylabel = "R^2"
    transform = "identity"
    sort_order = 13
    """
    return __run_batch_abs_metric(measures.batch_keep_retrain, X, y, model_generator, method_name, sklearn.metrics.r2_score, num_fcounts)

def batch_remove_absolute_retrain__roc_auc(X, y, model_generator, method_name, num_fcounts=11):
    """ Batch Remove Absolute (retrain)
    xlabel = "Fraction of features removed"
    ylabel = "1 - ROC AUC"
    transform = "one_minus"
    sort_order = 13
    """
    return __run_batch_abs_metric(measures.batch_remove_retrain, X, y, model_generator, method_name, sklearn.metrics.roc_auc_score, num_fcounts)

def batch_keep_absolute_retrain__roc_auc(X, y, model_generator, method_name, num_fcounts=11):
    """ Batch Keep Absolute (retrain)
    xlabel = "Fraction of features kept"
    ylabel = "ROC AUC"
    transform = "identity"
    sort_order = 13
    """
    return __run_batch_abs_metric(measures.batch_keep_retrain, X, y, model_generator, method_name, sklearn.metrics.roc_auc_score, num_fcounts)

def __run_batch_abs_metric(metric, X, y, model_generator, method_name, loss, num_fcounts):
    def score_function(fcount, X_train, X_test, y_train, y_test, attr_function, trained_model):
        A_train = np.abs(__strip_list(attr_function(X_train)))
        nkeep_train = (np.ones(len(y_train)) * fcount).astype(int)
        #nkeep_train = np.minimum(nkeep_train, np.array(A_train > 0).sum(1)).astype(int)
        A_test = np.abs(__strip_list(attr_function(X_test)))
        nkeep_test = (np.ones(len(y_test)) * fcount).astype(int)
        #nkeep_test = np.minimum(nkeep_test, np.array(A_test >= 0).sum(1)).astype(int)
        return metric(
            nkeep_train, nkeep_test, X_train, y_train, X_test, y_test, A_train, A_test,
            model_generator, loss
        )
    fcounts = __intlogspace(0, X.shape[1], num_fcounts)
    return fcounts, __score_method(X, y, fcounts, model_generator, score_function, method_name)

_attribution_cache = {}
def __score_method(X, y, fcounts, model_generator, score_function, method_name, nreps=10, test_size=100, cache_dir="/tmp"):
    """ Test an explanation method.
    """

    try:
        pickle
    except NameError:
        raise ImportError("The 'dill' package could not be loaded and is needed for the benchmark!")

    old_seed = np.random.seed()
    np.random.seed(3293)

    # average the method scores over several train/test splits
    method_reps = []

    data_hash = hashlib.sha256(__toarray(X).flatten()).hexdigest() + hashlib.sha256(__toarray(y)).hexdigest()
    for i in range(nreps):
        X_train, X_test, y_train, y_test = train_test_split(__toarray(X), y, test_size=test_size, random_state=i)

        # define the model we are going to explain, caching so we onlu build it once
        model_id = "model_cache__v" + "__".join([__version__, data_hash, model_generator.__name__])+".pickle"
        cache_file = os.path.join(cache_dir, model_id + ".pickle")
        if os.path.isfile(cache_file):
            with open(cache_file, "rb") as f:
                model = pickle.load(f)
        else:
            model = model_generator()
            model.fit(X_train, y_train)
            with open(cache_file, "wb") as f:
                pickle.dump(model, f)

        attr_key = "_".join([model_generator.__name__, method_name, str(test_size), str(nreps), str(i), data_hash])
        def score(attr_function):
            def cached_attr_function(X_inner):
                if attr_key not in _attribution_cache:
                    _attribution_cache[attr_key] = attr_function(X_inner)
                return _attribution_cache[attr_key]

            #cached_attr_function = lambda X: __check_cache(attr_function, X)
            if fcounts is None:
                return score_function(X_train, X_test, y_train, y_test, cached_attr_function, model, i)
            else:
                scores = []
                for f in fcounts:
                    scores.append(score_function(f, X_train, X_test, y_train, y_test, cached_attr_function, model, i))
                return np.array(scores)

        # evaluate the method (only building the attribution function if we need to)
        if attr_key not in _attribution_cache:
            method_reps.append(score(getattr(methods, method_name)(model, X_train)))
        else:
            method_reps.append(score(None))

    np.random.seed(old_seed)
    return np.array(method_reps).mean(0)


# used to memoize explainer functions so we don't waste time re-explaining the same object
__cache0 = None
__cache_X0 = None
__cache_f0 = None
__cache1 = None
__cache_X1 = None
__cache_f1 = None
def __check_cache(f, X):
    global __cache0, __cache_X0, __cache_f0
    global __cache1, __cache_X1, __cache_f1
    if X is __cache_X0 and f is __cache_f0:
        return __cache0
    elif X is __cache_X1 and f is __cache_f1:
        return __cache1
    else:
        __cache_f1 = __cache_f0
        __cache_X1 = __cache_X0
        __cache1 = __cache0
        __cache_f0 = f
        __cache_X0 = X
        __cache0 = f(X)
        return __cache0

def __intlogspace(start, end, count):
    return np.unique(np.round(start + (end-start) * (np.logspace(0, 1, count, endpoint=True) - 1) / 9).astype(int))

def __toarray(X):
    """ Converts DataFrames to numpy arrays.
    """
    if hasattr(X, "values"):
        X = X.values
    return X

def __strip_list(attrs):
    """ This assumes that if you have a list of outputs you just want the second one (the second class).
    """
    if isinstance(attrs, list):
        return attrs[1]
    else:
        return attrs

def _fit_human(model_generator, val00, val01, val11):
    # force the model to fit a function with almost entirely zero background
    N = 1000000
    M = 3
    X = np.zeros((N,M))
    X.shape
    y = np.ones(N) * val00
    X[0:1000, 0] = 1
    y[0:1000] = val01
    for i in range(0,1000000,1000):
        X[i, 1] = 1
        y[i] = val01
    y[0] = val11
    model = model_generator()
    model.fit(X, y)
    return model

def _human_and(X, model_generator, method_name, fever, cough):
    assert np.abs(X).max() == 0, "Human agreement metrics are only for use with the human_agreement dataset!"

    # these are from the sickness_score mturk user study experiment
    X_test = np.zeros((100,3))
    if not fever and not cough:
        human_consensus = np.array([0., 0., 0.])
        X_test[0,:] = np.array([[0., 0., 1.]])
    elif not fever and cough:
        human_consensus = np.array([0., 2., 0.])
        X_test[0,:] = np.array([[0., 1., 1.]])
    elif fever and cough:
        human_consensus = np.array([5., 5., 0.])
        X_test[0,:] = np.array([[1., 1., 1.]])

    # force the model to fit an XOR function with almost entirely zero background
    model = _fit_human(model_generator, 0, 2, 10)

    attr_function = getattr(methods, method_name)(model, X)
    methods_attrs = attr_function(X_test)
    return "human", (human_consensus, methods_attrs[0,:])

def human_and_00(X, y, model_generator, method_name):
    """ AND (false/false)

    This tests how well a feature attribution method agrees with human intuition
    for an AND operation combined with linear effects. This metric deals
    specifically with the question of credit allocation for the following function
    when all three inputs are true:
    if fever: +2 points
    if cough: +2 points
    if fever and cough: +6 points

    transform = "identity"
    sort_order = 0
    """
    return _human_and(X, model_generator, method_name, False, False)

def human_and_01(X, y, model_generator, method_name):
    """ AND (false/true)

    This tests how well a feature attribution method agrees with human intuition
    for an AND operation combined with linear effects. This metric deals
    specifically with the question of credit allocation for the following function
    when all three inputs are true:
    if fever: +2 points
    if cough: +2 points
    if fever and cough: +6 points

    transform = "identity"
    sort_order = 1
    """
    return _human_and(X, model_generator, method_name, False, True)

def human_and_11(X, y, model_generator, method_name):
    """ AND (true/true)

    This tests how well a feature attribution method agrees with human intuition
    for an AND operation combined with linear effects. This metric deals
    specifically with the question of credit allocation for the following function
    when all three inputs are true:
    if fever: +2 points
    if cough: +2 points
    if fever and cough: +6 points

    transform = "identity"
    sort_order = 2
    """
    return _human_and(X, model_generator, method_name, True, True)


def _human_or(X, model_generator, method_name, fever, cough):
    assert np.abs(X).max() == 0, "Human agreement metrics are only for use with the human_agreement dataset!"

    # these are from the sickness_score mturk user study experiment
    X_test = np.zeros((100,3))
    if not fever and not cough:
        human_consensus = np.array([0., 0., 0.])
        X_test[0,:] = np.array([[0., 0., 1.]])
    elif not fever and cough:
        human_consensus = np.array([0., 8., 0.])
        X_test[0,:] = np.array([[0., 1., 1.]])
    elif fever and cough:
        human_consensus = np.array([5., 5., 0.])
        X_test[0,:] = np.array([[1., 1., 1.]])

    # force the model to fit an XOR function with almost entirely zero background
    model = _fit_human(model_generator, 0, 8, 10)

    attr_function = getattr(methods, method_name)(model, X)
    methods_attrs = attr_function(X_test)
    return "human", (human_consensus, methods_attrs[0,:])

def human_or_00(X, y, model_generator, method_name):
    """ OR (false/false)

    This tests how well a feature attribution method agrees with human intuition
    for an OR operation combined with linear effects. This metric deals
    specifically with the question of credit allocation for the following function
    when all three inputs are true:
    if fever: +2 points
    if cough: +2 points
    if fever or cough: +6 points

    transform = "identity"
    sort_order = 0
    """
    return _human_or(X, model_generator, method_name, False, False)

def human_or_01(X, y, model_generator, method_name):
    """ OR (false/true)

    This tests how well a feature attribution method agrees with human intuition
    for an OR operation combined with linear effects. This metric deals
    specifically with the question of credit allocation for the following function
    when all three inputs are true:
    if fever: +2 points
    if cough: +2 points
    if fever or cough: +6 points

    transform = "identity"
    sort_order = 1
    """
    return _human_or(X, model_generator, method_name, False, True)

def human_or_11(X, y, model_generator, method_name):
    """ OR (true/true)

    This tests how well a feature attribution method agrees with human intuition
    for an OR operation combined with linear effects. This metric deals
    specifically with the question of credit allocation for the following function
    when all three inputs are true:
    if fever: +2 points
    if cough: +2 points
    if fever or cough: +6 points

    transform = "identity"
    sort_order = 2
    """
    return _human_or(X, model_generator, method_name, True, True)


def _human_xor(X, model_generator, method_name, fever, cough):
    assert np.abs(X).max() == 0, "Human agreement metrics are only for use with the human_agreement dataset!"

    # these are from the sickness_score mturk user study experiment
    X_test = np.zeros((100,3))
    if not fever and not cough:
        human_consensus = np.array([0., 0., 0.])
        X_test[0,:] = np.array([[0., 0., 1.]])
    elif not fever and cough:
        human_consensus = np.array([0., 8., 0.])
        X_test[0,:] = np.array([[0., 1., 1.]])
    elif fever and cough:
        human_consensus = np.array([2., 2., 0.])
        X_test[0,:] = np.array([[1., 1., 1.]])

    # force the model to fit an XOR function with almost entirely zero background
    model = _fit_human(model_generator, 0, 8, 4)

    attr_function = getattr(methods, method_name)(model, X)
    methods_attrs = attr_function(X_test)
    return "human", (human_consensus, methods_attrs[0,:])

def human_xor_00(X, y, model_generator, method_name):
    """ XOR (false/false)

    This tests how well a feature attribution method agrees with human intuition
    for an eXclusive OR operation combined with linear effects. This metric deals
    specifically with the question of credit allocation for the following function
    when all three inputs are true:
    if fever: +2 points
    if cough: +2 points
    if fever or cough but not both: +6 points

    transform = "identity"
    sort_order = 3
    """
    return _human_xor(X, model_generator, method_name, False, False)

def human_xor_01(X, y, model_generator, method_name):
    """ XOR (false/true)

    This tests how well a feature attribution method agrees with human intuition
    for an eXclusive OR operation combined with linear effects. This metric deals
    specifically with the question of credit allocation for the following function
    when all three inputs are true:
    if fever: +2 points
    if cough: +2 points
    if fever or cough but not both: +6 points

    transform = "identity"
    sort_order = 4
    """
    return _human_xor(X, model_generator, method_name, False, True)

def human_xor_11(X, y, model_generator, method_name):
    """ XOR (true/true)

    This tests how well a feature attribution method agrees with human intuition
    for an eXclusive OR operation combined with linear effects. This metric deals
    specifically with the question of credit allocation for the following function
    when all three inputs are true:
    if fever: +2 points
    if cough: +2 points
    if fever or cough but not both: +6 points

    transform = "identity"
    sort_order = 5
    """
    return _human_xor(X, model_generator, method_name, True, True)


def _human_sum(X, model_generator, method_name, fever, cough):
    assert np.abs(X).max() == 0, "Human agreement metrics are only for use with the human_agreement dataset!"

    # these are from the sickness_score mturk user study experiment
    X_test = np.zeros((100,3))
    if not fever and not cough:
        human_consensus = np.array([0., 0., 0.])
        X_test[0,:] = np.array([[0., 0., 1.]])
    elif not fever and cough:
        human_consensus = np.array([0., 2., 0.])
        X_test[0,:] = np.array([[0., 1., 1.]])
    elif fever and cough:
        human_consensus = np.array([2., 2., 0.])
        X_test[0,:] = np.array([[1., 1., 1.]])

    # force the model to fit an XOR function with almost entirely zero background
    model = _fit_human(model_generator, 0, 2, 4)

    attr_function = getattr(methods, method_name)(model, X)
    methods_attrs = attr_function(X_test)
    return "human", (human_consensus, methods_attrs[0,:])

def human_sum_00(X, y, model_generator, method_name):
    """ SUM (false/false)

    This tests how well a feature attribution method agrees with human intuition
    for a SUM operation. This metric deals
    specifically with the question of credit allocation for the following function
    when all three inputs are true:
    if fever: +2 points
    if cough: +2 points

    transform = "identity"
    sort_order = 0
    """
    return _human_sum(X, model_generator, method_name, False, False)

def human_sum_01(X, y, model_generator, method_name):
    """ SUM (false/true)

    This tests how well a feature attribution method agrees with human intuition
    for a SUM operation. This metric deals
    specifically with the question of credit allocation for the following function
    when all three inputs are true:
    if fever: +2 points
    if cough: +2 points

    transform = "identity"
    sort_order = 1
    """
    return _human_sum(X, model_generator, method_name, False, True)

def human_sum_11(X, y, model_generator, method_name):
    """ SUM (true/true)

    This tests how well a feature attribution method agrees with human intuition
    for a SUM operation. This metric deals
    specifically with the question of credit allocation for the following function
    when all three inputs are true:
    if fever: +2 points
    if cough: +2 points

    transform = "identity"
    sort_order = 2
    """
    return _human_sum(X, model_generator, method_name, True, True)

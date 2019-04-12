from .. import LinearExplainer
from .. import KernelExplainer
from .. import SamplingExplainer
from ..explainers import other
from .. import __version__
from . import measures
from . import methods
import sklearn
import numpy as np
import copy
import functools
import time
import hashlib
import os
import pickle

try:
    from sklearn.model_selection import train_test_split
except Exception:
    from sklearn.cross_validation import train_test_split


def runtime(X, y, model_generator, method_name):
    """ Runtime
    transform = "negate"
    sort_order = 1
    """

    old_seed = np.random.seed()
    np.random.seed(3293)

    # average the method scores over several train/test splits
    method_reps = []
    for i in range(1):
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
    sort_order = 2
    """

    def score_map(true, pred):
        """ Converts local accuracy from % of standard deviation to numerical scores for coloring.
        """

        v = min(1.0, np.std(pred - true) / (np.std(true) + 1e-8))
        if v < 1e-6:
            return 1.0
        elif v < 0.01:
            return 0.9
        elif v < 0.05:
            return 0.75
        elif v < 0.1:
            return 0.6
        elif v < 0.2:
            return 0.4
        elif v < 0.3:
            return 0.3
        elif v < 0.5:
            return 0.2
        elif v < 0.7:
            return 0.1
        else:
            return 0.0
    def score_function(X_train, X_test, y_train, y_test, attr_function, trained_model, random_state):
        return measures.local_accuracy(
            X_train, y_train, X_test, y_test, attr_function(X_test),
            model_generator, score_map, trained_model
        )
    return None, __score_method(X, y, None, model_generator, score_function, method_name)

def consistency_guarantees(X, y, model_generator, method_name):
    """ Consistency Guarantees
    transform = "identity"
    sort_order = 3
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
        nmask = np.minimum(nmask, np.array(A >= 0).sum(1)).astype(np.int)
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
        nkeep_train = (np.ones(len(y_train)) * fcount).astype(np.int)
        #nkeep_train = np.minimum(nkeep_train, np.array(A_train > 0).sum(1)).astype(np.int)
        A_test = np.abs(__strip_list(attr_function(X_test)))
        nkeep_test = (np.ones(len(y_test)) * fcount).astype(np.int)
        #nkeep_test = np.minimum(nkeep_test, np.array(A_test >= 0).sum(1)).astype(np.int)
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

        def score(attr_function):
            def cached_attr_function(X_inner):
                key = "_".join([model_generator.__name__, method_name, str(test_size), str(nreps), str(i), data_hash])
                if key not in _attribution_cache:
                    _attribution_cache[key] = attr_function(X_inner)
                return _attribution_cache[key]

            #cached_attr_function = lambda X: __check_cache(attr_function, X)
            if fcounts is None:
                return score_function(X_train, X_test, y_train, y_test, cached_attr_function, model, i)
            else:
                scores = []
                for f in fcounts:
                    scores.append(score_function(f, X_train, X_test, y_train, y_test, cached_attr_function, model, i))
                return np.array(scores)

        # evaluate the method
        method_reps.append(score(getattr(methods, method_name)(model, X_train)))

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
    return np.unique(np.round(start + (end-start) * (np.logspace(0, 1, count, endpoint=True) - 1) / 9).astype(np.int))

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



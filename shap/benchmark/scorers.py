from .. import LinearExplainer
from .. import KernelExplainer
from .. import SamplingExplainer
from ..explainers import other
from . import metrics
from . import methods
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import copy
import functools
import time

def consistency_guarantees(X, y, model_generator, method_name):
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
        "tree_shap": 1.0,
        "mean_abs_tree_shap": 1.0,
        "lime_tabular_regression_1000": 0.8
    }
    
    return None, guarantees[method_name]

def local_accuracy(X, y, model_generator, method_name):
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
    def score_function(X_train, X_test, y_train, y_test, attr_function):
        return metrics.local_accuracy(
            X_train, y_train, X_test, y_test, attr_function(X_test),
            model_generator, score_map
        )
    return None, score_method(X, y, None, model_generator, score_function, method_name)

def runtime(X, y, model_generator, method_name):

    old_seed = np.random.seed()
    np.random.seed(3293)

    # average the method scores over several train/test splits
    method_reps = []
    for i in range(1):
        X_train, X_test, y_train, _ = train_test_split(toarray(X), y, test_size=0.1, random_state=i)

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

def remove_positive(X, y, model_generator, method_name, num_fcounts=11):
    return run_metric(metrics.remove, X, y, model_generator, method_name, 1, num_fcounts)

def remove_negative(X, y, model_generator, method_name, num_fcounts=11):
    return run_metric(metrics.remove, X, y, model_generator, method_name, -1, num_fcounts)

def mask_remove_positive(X, y, model_generator, method_name, num_fcounts=11):
    return run_metric(metrics.mask_remove, X, y, model_generator, method_name, 1, num_fcounts)

def mask_remove_negative(X, y, model_generator, method_name, num_fcounts=11):
    return run_metric(metrics.mask_remove, X, y, model_generator, method_name, -1, num_fcounts)

def keep_positive(X, y, model_generator, method_name, num_fcounts=11):
    return run_metric(metrics.keep, X, y, model_generator, method_name, 1, num_fcounts)

def keep_negative(X, y, model_generator, method_name, num_fcounts=11):
    return run_metric(metrics.keep, X, y, model_generator, method_name, -1, num_fcounts)

def mask_keep_positive(X, y, model_generator, method_name, num_fcounts=11):
    return run_metric(metrics.mask_keep, X, y, model_generator, method_name, 1, num_fcounts)

def mask_keep_negative(X, y, model_generator, method_name, num_fcounts=11):
    return run_metric(metrics.mask_keep, X, y, model_generator, method_name, -1, num_fcounts)

def run_metric(metric, X, y, model_generator, method_name, attribution_sign, num_fcounts):
    def metric_function(true, pred):
        return np.mean(pred)
    def score_function(fcount, X_train, X_test, y_train, y_test, attr_function):
        A = attribution_sign * attr_function(X_test)
        nmask = np.ones(len(y_test)) * fcount
        nmask = np.minimum(nmask, np.array(A > 0).sum(1)).astype(np.int)
        return metric(
            nmask, X_train, y_train, X_test, y_test, A,
            model_generator, metric_function
        )
    fcounts = intspace(0, X.shape[1], num_fcounts)
    return fcounts, score_method(X, y, fcounts, model_generator, score_function, method_name)

def batch_remove_absolute_r2(X, y, model_generator, method_name, num_fcounts=11):
    return run_batch_abs_metric(metrics.batch_remove, X, y, model_generator, method_name, sklearn.metrics.r2_score, num_fcounts)

def batch_keep_absolute_r2(X, y, model_generator, method_name, num_fcounts=11):
    return run_batch_abs_metric(metrics.batch_keep, X, y, model_generator, method_name, sklearn.metrics.r2_score, num_fcounts)

def run_batch_abs_metric(metric, X, y, model_generator, method_name, loss, num_fcounts):
    def score_function(fcount, X_train, X_test, y_train, y_test, attr_function):
        A_train = np.abs(attr_function(X_train))
        nkeep_train = (np.ones(len(y_train)) * fcount).astype(np.int)
        #nkeep_train = np.minimum(nkeep_train, np.array(A_train > 0).sum(1)).astype(np.int)
        A_test = np.abs(attr_function(X_test))
        nkeep_test = (np.ones(len(y_test)) * fcount).astype(np.int)
        #nkeep_test = np.minimum(nkeep_test, np.array(A_test >= 0).sum(1)).astype(np.int)
        return metric(
            nkeep_train, nkeep_test, X_train, y_train, X_test, y_test, A_train, A_test,
            model_generator, loss
        )
    fcounts = intspace(0, X.shape[1], num_fcounts)
    return fcounts, score_method(X, y, fcounts, model_generator, score_function, method_name)


def score_method(X, y, fcounts, model_generator, score_function, method_name):
    """ Test an explanation method.
    """

    old_seed = np.random.seed()
    np.random.seed(3293)

    # average the method scores over several train/test splits
    method_reps = []
    for i in range(1):
        X_train, X_test, y_train, y_test = train_test_split(toarray(X), y, test_size=0.1, random_state=i)

        # define the model we are going to explain
        model = model_generator()
        model.fit(X_train, y_train)

        def score(attr_function):
            cached_attr_function = lambda X: check_cache(attr_function, X)
            if fcounts is None:
                return score_function(X_train, X_test, y_train, y_test, cached_attr_function)
            else:
                scores = []
                for f in fcounts:
                    scores.append(score_function(f, X_train, X_test, y_train, y_test, cached_attr_function))
                return np.array(scores)

        # evaluate the method
        method_reps.append(score(getattr(methods, method_name)(model, X_train)))

    np.random.seed(old_seed)
    return np.array(method_reps).mean(0)


# used to memoize explainer functions so we don't waste time re-explaining the same object
cache0 = None
cache_X0 = None
cache_f0 = None
cache1 = None
cache_X1 = None
cache_f1 = None
def check_cache(f, X):
    global cache0, cache_X0, cache_f0
    global cache1, cache_X1, cache_f1
    if X is cache_X0 and f is cache_f0:
        return cache0
    elif X is cache_X1 and f is cache_f1:
        return cache1
    else:
        cache_f1 = cache_f0
        cache_X1 = cache_X0
        cache1 = cache0
        cache_f0 = f
        cache_X0 = X
        cache0 = f(X)
        return cache0

def intspace(start, end, count):
    return np.unique(np.round(np.linspace(start, end, count)).astype(np.int))

def toarray(X):
    """ Converts DataFrames to numpy arrays.
    """
    if hasattr(X, "values"):
        X = X.values
    return X
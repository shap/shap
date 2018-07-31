from .. import LinearExplainer
from .. import KernelExplainer
from .. import SamplingExplainer
from ..explainers import other
from . import metrics
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import copy
import functools


def remove_positive(X, y, model_generator, methods, num_fcounts=11):
    def score_function(fcount, X_train, X_test, y_train, y_test, attr_function):
        A = attr_function(X_test)
        nmask = np.ones(len(y_test)) * fcount
        nmask = np.minimum(nmask, np.array(A > 0).sum(1)).astype(np.int)
        return metrics.remove(
            nmask, X_train, y_train, X_test, y_test, A,
            model_generator, lambda true, pred: np.mean(pred)
        )
    fcounts = intspace(0, X.shape[1], num_fcounts)
    return "remove_positive", fcounts, score_methods(X, y, fcounts, model_generator, score_function, methods)

def remove_negative(X, y, model_generator, methods, num_fcounts=11):
    def score_function(fcount, X_train, X_test, y_train, y_test, attr_function):
        A = attr_function(X_test)
        nmask = np.ones(len(y_test)) * fcount
        nmask = np.minimum(nmask, np.array(A < 0).sum(1)).astype(np.int)
        return metrics.remove(
            nmask, X_train, y_train, X_test, y_test, -A,
            model_generator, lambda true, pred: np.mean(pred)
        )
    fcounts = intspace(0, X.shape[1], num_fcounts)
    return "remove_negative", fcounts, score_methods(X, y, fcounts, model_generator, score_function, methods)

def keep_positive(X, y, model_generator, methods, num_fcounts=11):
    def score_function(fcount, X_train, X_test, y_train, y_test, attr_function):
        A = attr_function(X_test)
        nmask = np.ones(len(y_test)) * fcount
        nmask = np.minimum(nmask, np.array(A > 0).sum(1)).astype(np.int)
        return metrics.keep(
            nmask, X_train, y_train, X_test, y_test, A,
            model_generator, lambda true, pred: np.mean(pred)
        )
    fcounts = intspace(0, X.shape[1], num_fcounts)
    return "keep_positive", fcounts, score_methods(X, y, fcounts, model_generator, score_function, methods)

def keep_negative(X, y, model_generator, methods, num_fcounts=11):
    def score_function(fcount, X_train, X_test, y_train, y_test, attr_function):
        A = attr_function(X_test)
        nmask = np.ones(len(y_test)) * fcount
        nmask = np.minimum(nmask, np.array(A < 0).sum(1)).astype(np.int)
        return metrics.keep(
            nmask, X_train, y_train, X_test, y_test, -A,
            model_generator, lambda true, pred: np.mean(pred)
        )
    fcounts = intspace(0, X.shape[1], num_fcounts)
    return "keep_negative", fcounts, score_methods(X, y, fcounts, model_generator, score_function, methods)

def batch_remove_absolute_r2(X, y, model_generator, methods, num_fcounts=11):
    return ("batch_remove_absolute_r2",) + _batch_remove_absolute(X, y, model_generator, methods, sklearn.metrics.r2_score, num_fcounts)

def _batch_remove_absolute(X, y, model_generator, methods, loss, num_fcounts):
    def score_function(fcount, X_train, X_test, y_train, y_test, attr_function):
        A_train = np.abs(attr_function(X_train))
        nmask_train = (np.ones(len(y_train)) * fcount).astype(np.int)
        #nmask_train = np.minimum(nmask_train, np.array(A_train > 0).sum(1)).astype(np.int)
        A_test = np.abs(attr_function(X_test))
        nmask_test = (np.ones(len(y_test)) * fcount).astype(np.int)
        #nmask_test = np.minimum(nmask_test, np.array(A_test > 0).sum(1)).astype(np.int)
        return metrics.batch_remove(
            nmask_train, nmask_test, X_train, y_train, X_test, y_test, A_train, A_test,
            model_generator, loss
        )
    fcounts = intspace(0, X.shape[1], num_fcounts)
    return fcounts, score_methods(X, y, fcounts, model_generator, score_function, methods)

def batch_keep_absolute_r2(X, y, model_generator, methods, num_fcounts=11):
    return ("batch_keep_absolute_r2",) + _batch_keep_absolute(X, y, model_generator, methods, sklearn.metrics.r2_score, num_fcounts)

def _batch_keep_absolute(X, y, model_generator, methods, loss, num_fcounts):
    def score_function(fcount, X_train, X_test, y_train, y_test, attr_function):
        A_train = np.abs(attr_function(X_train))
        nkeep_train = (np.ones(len(y_train)) * fcount).astype(np.int)
        #nkeep_train = np.minimum(nkeep_train, np.array(A_train > 0).sum(1)).astype(np.int)
        A_test = np.abs(attr_function(X_test))
        nkeep_test = (np.ones(len(y_test)) * fcount).astype(np.int)
        #nkeep_test = np.minimum(nkeep_test, np.array(A_test >= 0).sum(1)).astype(np.int)
        return metrics.batch_keep(
            nkeep_train, nkeep_test, X_train, y_train, X_test, y_test, A_train, A_test,
            model_generator, loss
        )
    fcounts = intspace(0, X.shape[1], num_fcounts)
    return fcounts, score_methods(X, y, fcounts, model_generator, score_function, methods)


def score_methods(X, y, fcounts, model_generator, score_function, methods):
    """ Test a set of explanation methods.
    """

    old_seed = np.random.seed()
    np.random.seed(3293)

    # average the method scores over several train/test splits
    method_reps = []
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i)

        # define the model we are going to explain
        model = model_generator()
        model.fit(X_train, y_train)

        def score(attr_function):
            scores = []
            cached_attr_function = lambda X: check_cache(attr_function, X)
            for f in fcounts:
                if f == 100000.0: # some models don't like being given all constant values
                    scores.append(0)
                else:
                    scores.append(score_function(f, X_train, X_test, y_train, y_test, cached_attr_function))
            return np.array(scores)

        # evaluate each method
        method_reps.append([[m[0], score(m[1](model, X_train))] for m in methods])
    np.random.seed(old_seed)
    return average_methods(method_reps)

def average_methods(method_reps):
    methods = copy.deepcopy(method_reps[0])
    for rep in method_reps[1:]:
        for i in range(len(rep)):
            methods[i][1] += rep[i][1]
    for i in range(len(methods)):
        methods[i][1] /= len(method_reps)

    return methods


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

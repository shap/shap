from .. import datasets
from . import scorers
from . import models
from . import methods
from .. import __version__
import numpy as np
import sklearn
import os
import pickle
import sys
import time
import subprocess
from multiprocessing import Pool
import itertools
from shap.benchmark.methods import linear_regress
from shap.benchmark.methods import tree_regress

all_scorers = [
    "runtime",
    "local_accuracy",
    "consistency_guarantees",
    "mask_keep_positive",
    "mask_keep_negative",
    "keep_positive",
    "keep_negative",
    "batch_keep_absolute_r2",
    "mask_remove_positive",
    "mask_remove_negative",
    "remove_positive",
    "remove_negative",
    "batch_remove_absolute_r2"
]

_tests = []
_tests += [["corrgroups60", "lasso", m, s] for s in all_scorers for m in linear_regress]
_tests += [["corrgroups60", "ridge", m, s] for s in all_scorers for m in linear_regress]
_tests += [["corrgroups60", "decision_tree", m, s] for s in all_scorers for m in tree_regress]
_tests += [["corrgroups60", "random_forest", m, s] for s in all_scorers for m in tree_regress]
_tests += [["corrgroups60", "gbm", m, s] for s in all_scorers for m in tree_regress]

def tests(dataset=None, model=None, method=None, scorer=None):
    for test in _tests:
        if dataset is not None and dataset != test[0]:
            continue
        if model is not None and model != test[1]:
            continue
        if method is not None and method != test[2]:
            continue
        if scorer is not None and scorer != test[3]:
            continue
        yield test

def run_test(test, use_cache=True, cache_dir="/tmp"):
    dataset_name, model_name, method_name, scorer_name = test

    # see if we have a cached version
    cache_id = "v" + "__".join([__version__, dataset_name, model_name, method_name, scorer_name])
    cache_file = os.path.join(cache_dir, cache_id + ".pickle")
    if use_cache and os.path.isfile(cache_file):
        with open(cache_file, "rb") as f:
            #print(cache_id.replace("__", " ") + " ...loaded from cache.")
            return pickle.load(f)

    # compute the scores
    print(cache_id.replace("__", " ") + " ...")
    sys.stdout.flush()
    start = time.time()
    X,y = getattr(datasets, dataset_name)()
    score = getattr(scorers, scorer_name)(
        X, y,
        getattr(models, dataset_name+"__"+model_name),
        method_name
    )
    print("...took %f seconds.\n" % (time.time() - start))

    # cache the scores
    with open(cache_file, "wb") as f:
        pickle.dump(score, f)

    return score
        

def run_tests_helper(args):
    test, cache_dir = args
    return run_test(test, cache_dir=cache_dir)

def run_tests(dataset=None, model=None, method=None, scorer=None, cache_dir="/tmp", nworkers=1):
    tests_arr = list(tests(dataset=dataset, model=model, method=method, scorer=scorer))
    if nworkers == 1:
        out = list(map(run_tests_helper, zip(tests_arr, itertools.repeat(cache_dir))))
    else:
        with Pool(nworkers) as pool:
            out = pool.map(run_tests_helper, zip(tests_arr, itertools.repeat(cache_dir)))
    return list(zip(tests_arr, out))
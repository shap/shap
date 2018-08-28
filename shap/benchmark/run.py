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


def run_test(dataset_name, model_name, methods_set_name, scorer_name, use_cache=True, cache_dir="/tmp", remote=None):
    
    # see if we have a cached version
    cache_id = "v" + "__".join([__version__, dataset_name, model_name, methods_set_name, scorer_name])
    cache_file = os.path.join(cache_dir, cache_id + ".pickle")
    if use_cache and os.path.isfile(cache_file):
        with open(cache_file, "rb") as f:
            print(cache_id.replace("__", " ") + " ...loaded from cache.")
            return pickle.load(f)
    
    if remote is not None:

        # run the benchmark on the remote machine
        start = time.time()
        subprocess.check_output([
            "ssh", "-t", remote,
            'python -c \'import shap; shap.benchmark.run_test("%s", "%s", "%s", "%s", cache_dir="%s")\'' % (
                dataset_name, model_name, methods_set_name, scorer_name, cache_dir
            )
        ])

        # copy the results back
        subprocess.check_output(["scp", remote+":"+cache_file, cache_file])

        if os.path.isfile(cache_file):
            with open(cache_file, "rb") as f:
                print(cache_id.replace("__", " ") + " ...loaded from remote after %f seconds" % (time.time() - start))
                return pickle.load(f)
        else:
            raise Exception("Remote benchmark call finished but no local file was found!")

    else:
        # compute the scores
        print(cache_id.replace("__", " ") + " ...")
        sys.stdout.flush()
        start = time.time()
        score = getattr(scorers, scorer_name)(
            *getattr(datasets, dataset_name)(),
            getattr(models, model_name),
            getattr(methods, methods_set_name)
        )
        print("...took %f seconds.\n" % (time.time() - start))

        # cache the scores
        with open(cache_file, "wb") as f:
            pickle.dump(score, f)

        return score

def run_tests_helper(args):
    test, cache_dir, remote = args
    return run_test(*test, cache_dir=cache_dir, remote=remote)

def run_tests(tests, cache_dir="/tmp", remote=None, nworkers=10):
    pool = Pool(nworkers)
    out = pool.map(run_tests_helper, zip(tests, itertools.repeat(cache_dir), itertools.repeat(remote)))
    pool.close()
    pool.join()
    return out
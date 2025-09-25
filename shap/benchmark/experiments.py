import copy
import itertools
import os
import pickle
import random
import subprocess
import sys
import time
from multiprocessing import Pool

from .. import __version__, datasets
from . import metrics, models

try:
    from queue import Queue
except ImportError:
    from Queue import Queue
from threading import Lock, Thread

regression_metrics = [
    "local_accuracy",
    "consistency_guarantees",
    "keep_positive_mask",
    "keep_positive_resample",
    # "keep_positive_impute",
    "keep_negative_mask",
    "keep_negative_resample",
    # "keep_negative_impute",
    "keep_absolute_mask__r2",
    "keep_absolute_resample__r2",
    # "keep_absolute_impute__r2",
    "remove_positive_mask",
    "remove_positive_resample",
    # "remove_positive_impute",
    "remove_negative_mask",
    "remove_negative_resample",
    # "remove_negative_impute",
    "remove_absolute_mask__r2",
    "remove_absolute_resample__r2",
    # "remove_absolute_impute__r2"
    "runtime",
]

binary_classification_metrics = [
    "local_accuracy",
    "consistency_guarantees",
    "keep_positive_mask",
    "keep_positive_resample",
    # "keep_positive_impute",
    "keep_negative_mask",
    "keep_negative_resample",
    # "keep_negative_impute",
    "keep_absolute_mask__roc_auc",
    "keep_absolute_resample__roc_auc",
    # "keep_absolute_impute__roc_auc",
    "remove_positive_mask",
    "remove_positive_resample",
    # "remove_positive_impute",
    "remove_negative_mask",
    "remove_negative_resample",
    # "remove_negative_impute",
    "remove_absolute_mask__roc_auc",
    "remove_absolute_resample__roc_auc",
    # "remove_absolute_impute__roc_auc"
    "runtime",
]

human_metrics = [
    "human_and_00",
    "human_and_01",
    "human_and_11",
    "human_or_00",
    "human_or_01",
    "human_or_11",
    "human_xor_00",
    "human_xor_01",
    "human_xor_11",
    "human_sum_00",
    "human_sum_01",
    "human_sum_11",
]

linear_regress_methods = [
    "linear_shap_corr",
    "linear_shap_ind",
    "coef",
    "random",
    "kernel_shap_1000_meanref",
    # "kernel_shap_100_meanref",
    # "sampling_shap_10000",
    "sampling_shap_1000",
    "lime_tabular_regression_1000",
    # "sampling_shap_100"
]

linear_classify_methods = [
    # NEED LIME
    "linear_shap_corr",
    "linear_shap_ind",
    "coef",
    "random",
    "kernel_shap_1000_meanref",
    # "kernel_shap_100_meanref",
    # "sampling_shap_10000",
    "sampling_shap_1000",
    # "lime_tabular_regression_1000"
    # "sampling_shap_100"
]

tree_regress_methods = [
    # NEED tree_shap_ind
    # NEED split_count?
    "tree_shap_tree_path_dependent",
    "tree_shap_independent_200",
    "saabas",
    "random",
    "tree_gain",
    "kernel_shap_1000_meanref",
    "mean_abs_tree_shap",
    # "kernel_shap_100_meanref",
    # "sampling_shap_10000",
    "sampling_shap_1000",
    "lime_tabular_regression_1000",
    "maple",
    # "sampling_shap_100"
]

rf_regress_methods = [  # methods that only support random forest models
    "tree_maple"
]

tree_classify_methods = [
    # NEED tree_shap_ind
    # NEED split_count?
    "tree_shap_tree_path_dependent",
    "tree_shap_independent_200",
    "saabas",
    "random",
    "tree_gain",
    "kernel_shap_1000_meanref",
    "mean_abs_tree_shap",
    # "kernel_shap_100_meanref",
    # "sampling_shap_10000",
    "sampling_shap_1000",
    "lime_tabular_classification_1000",
    "maple",
    # "sampling_shap_100"
]

deep_regress_methods = [
    "deep_shap",
    "expected_gradients",
    "random",
    "kernel_shap_1000_meanref",
    "sampling_shap_1000",
    # "lime_tabular_regression_1000"
]

deep_classify_methods = [
    "deep_shap",
    "expected_gradients",
    "random",
    "kernel_shap_1000_meanref",
    "sampling_shap_1000",
    # "lime_tabular_regression_1000"
]

_experiments = []
_experiments += [["corrgroups60", "lasso", m, s] for s in regression_metrics for m in linear_regress_methods]
_experiments += [["corrgroups60", "ridge", m, s] for s in regression_metrics for m in linear_regress_methods]
_experiments += [["corrgroups60", "decision_tree", m, s] for s in regression_metrics for m in tree_regress_methods]
_experiments += [
    ["corrgroups60", "random_forest", m, s]
    for s in regression_metrics
    for m in (tree_regress_methods + rf_regress_methods)
]
_experiments += [["corrgroups60", "gbm", m, s] for s in regression_metrics for m in tree_regress_methods]
_experiments += [["corrgroups60", "ffnn", m, s] for s in regression_metrics for m in deep_regress_methods]

_experiments += [["independentlinear60", "lasso", m, s] for s in regression_metrics for m in linear_regress_methods]
_experiments += [["independentlinear60", "ridge", m, s] for s in regression_metrics for m in linear_regress_methods]
_experiments += [
    ["independentlinear60", "decision_tree", m, s] for s in regression_metrics for m in tree_regress_methods
]
_experiments += [
    ["independentlinear60", "random_forest", m, s]
    for s in regression_metrics
    for m in (tree_regress_methods + rf_regress_methods)
]
_experiments += [["independentlinear60", "gbm", m, s] for s in regression_metrics for m in tree_regress_methods]
_experiments += [["independentlinear60", "ffnn", m, s] for s in regression_metrics for m in deep_regress_methods]

_experiments += [["cric", "lasso", m, s] for s in binary_classification_metrics for m in linear_classify_methods]
_experiments += [["cric", "ridge", m, s] for s in binary_classification_metrics for m in linear_classify_methods]
_experiments += [["cric", "decision_tree", m, s] for s in binary_classification_metrics for m in tree_classify_methods]
_experiments += [["cric", "random_forest", m, s] for s in binary_classification_metrics for m in tree_classify_methods]
_experiments += [["cric", "gbm", m, s] for s in binary_classification_metrics for m in tree_classify_methods]
_experiments += [["cric", "ffnn", m, s] for s in binary_classification_metrics for m in deep_classify_methods]

_experiments += [["human", "decision_tree", m, s] for s in human_metrics for m in tree_regress_methods]


def experiments(dataset=None, model=None, method=None, metric=None):
    for experiment in _experiments:
        if dataset is not None and dataset != experiment[0]:
            continue
        if model is not None and model != experiment[1]:
            continue
        if method is not None and method != experiment[2]:
            continue
        if metric is not None and metric != experiment[3]:
            continue
        yield experiment


def run_experiment(experiment, use_cache=True, cache_dir="/tmp"):
    dataset_name, model_name, method_name, metric_name = experiment

    # see if we have a cached version
    cache_id = __gen_cache_id(experiment)
    cache_file = os.path.join(cache_dir, cache_id + ".pickle")
    if use_cache and os.path.isfile(cache_file):
        with open(cache_file, "rb") as f:
            # print(cache_id.replace("__", " ") + " ...loaded from cache.")
            return pickle.load(f)

    # compute the scores
    print(cache_id.replace("__", " ", 4) + " ...")
    sys.stdout.flush()
    start = time.time()
    X, y = getattr(datasets, dataset_name)()
    score = getattr(metrics, metric_name)(X, y, getattr(models, dataset_name + "__" + model_name), method_name)
    print("...took %f seconds.\n" % (time.time() - start))

    # cache the scores
    with open(cache_file, "wb") as f:
        pickle.dump(score, f)

    return score


def run_experiments_helper(args):
    experiment, cache_dir = args
    return run_experiment(experiment, cache_dir=cache_dir)


def run_experiments(dataset=None, model=None, method=None, metric=None, cache_dir="/tmp", nworkers=1):
    experiments_arr = list(experiments(dataset=dataset, model=model, method=method, metric=metric))
    if nworkers == 1:
        out = list(map(run_experiments_helper, zip(experiments_arr, itertools.repeat(cache_dir))))
    else:
        with Pool(nworkers) as pool:
            out = pool.map(run_experiments_helper, zip(experiments_arr, itertools.repeat(cache_dir)))
    return list(zip(experiments_arr, out))


nexperiments = 0
total_sent = 0
total_done = 0
total_failed = 0
host_records = {}
worker_lock = Lock()
ssh_conn_per_min_limit = 0  # set as an argument to run_remote_experiments


def __thread_worker(q, host):
    global total_sent, total_done
    hostname, python_binary = host.split(":")
    while True:
        # make sure we are not sending too many ssh connections to the host
        # (if we send too many connections ssh thottling will lock us out)
        while True:
            all_clear = False

            worker_lock.acquire()
            try:
                if hostname not in host_records:
                    host_records[hostname] = []

                if len(host_records[hostname]) < ssh_conn_per_min_limit:
                    all_clear = True
                elif time.time() - host_records[hostname][-ssh_conn_per_min_limit] > 61:
                    all_clear = True
            finally:
                worker_lock.release()

            # if we are clear to send a new ssh connection then break
            if all_clear:
                break

            # if we are not clear then we sleep and try again
            time.sleep(5)

        experiment = q.get()

        # if we are not loading from the cache then we note that we have called the host
        cache_dir = "/tmp"
        cache_file = os.path.join(cache_dir, __gen_cache_id(experiment) + ".pickle")
        if not os.path.isfile(cache_file):
            worker_lock.acquire()
            try:
                host_records[hostname].append(time.time())
            finally:
                worker_lock.release()

        # record how many we have sent off for execution
        worker_lock.acquire()
        try:
            total_sent += 1
            __print_status()
        finally:
            worker_lock.release()

        __run_remote_experiment(experiment, hostname, cache_dir=cache_dir, python_binary=python_binary)

        # record how many are finished
        worker_lock.acquire()
        try:
            total_done += 1
            __print_status()
        finally:
            worker_lock.release()

        q.task_done()


def __print_status():
    print(
        f"Benchmark task {total_done} of {nexperiments} done ({total_failed} failed, {total_sent - total_done} running)",
        end="\r",
    )
    sys.stdout.flush()


def run_remote_experiments(experiments, thread_hosts, rate_limit=10):
    """Use ssh to run the experiments on remote machines in parallel.

    Parameters
    ----------
    experiments : iterable
        Output of shap.benchmark.experiments(...).

    thread_hosts : list of strings
        Each host has the format "host_name:path_to_python_binary" and can appear multiple times
        in the list (one for each parallel execution you want on that machine).

    rate_limit : int
        How many ssh connections we make per minute to each host (to avoid throttling issues).

    """
    global ssh_conn_per_min_limit
    ssh_conn_per_min_limit = rate_limit

    # first we kill any remaining workers from previous runs
    # note we don't check_call because pkill kills our ssh call as well
    thread_hosts = copy.copy(thread_hosts)
    random.shuffle(thread_hosts)
    for host in set(thread_hosts):
        hostname, _ = host.split(":")
        try:
            subprocess.run(["ssh", hostname, "pkill -f shap.benchmark.run_experiment"], timeout=15)
        except subprocess.TimeoutExpired:
            print("Failed to connect to", hostname, "after 15 seconds! Exiting.")
            return

    experiments = copy.copy(list(experiments))
    random.shuffle(experiments)  # this way all the hard experiments don't get put on one machine
    global nexperiments, total_sent, total_done, total_failed, host_records
    nexperiments = len(experiments)
    total_sent = 0
    total_done = 0
    total_failed = 0
    host_records = {}

    q = Queue()

    for host in thread_hosts:
        worker = Thread(target=__thread_worker, args=(q, host))
        worker.daemon = True
        worker.start()

    for experiment in experiments:
        q.put(experiment)

    q.join()


def __run_remote_experiment(experiment, remote, cache_dir="/tmp", python_binary="python"):
    global total_failed
    dataset_name, model_name, method_name, metric_name = experiment

    # see if we have a cached version
    cache_id = __gen_cache_id(experiment)
    cache_file = os.path.join(cache_dir, cache_id + ".pickle")
    if os.path.isfile(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    # this is just so we don't dump everything at once on a machine
    time.sleep(random.uniform(0, 5))

    # run the benchmark on the remote machine
    # start = time.time()
    func = f"shap.benchmark.run_experiment(['{dataset_name}', '{model_name}', '{method_name}', '{metric_name}'], cache_dir='{cache_dir}')"
    cmd = 'CUDA_VISIBLE_DEVICES="" ' + python_binary + f' -c "import shap; {func}" &> {cache_dir}/{cache_id}.output'
    try:
        subprocess.check_output(["ssh", remote, cmd])
    except subprocess.CalledProcessError as e:
        print(f"The following command failed on {remote}:", file=sys.stderr)
        print(cmd, file=sys.stderr)
        total_failed += 1
        print(e)
        return

    # copy the results back
    subprocess.check_output(["scp", remote + ":" + cache_file, cache_file])

    if os.path.isfile(cache_file):
        with open(cache_file, "rb") as f:
            # print(cache_id.replace("__", " ") + " ...loaded from remote after %f seconds" % (time.time() - start))
            return pickle.load(f)
    else:
        raise FileNotFoundError("Remote benchmark call finished but no local file was found!")


def __gen_cache_id(experiment):
    dataset_name, model_name, method_name, metric_name = experiment
    return "v" + "__".join([__version__, dataset_name, model_name, method_name, metric_name])

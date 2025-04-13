import itertools as it

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import perturbation


def update(model, attributions, X, y, masker, sort_order, perturbation_method, scores):
    metric = perturbation_method + " " + sort_order
    sp = perturbation.SequentialPerturbation(model, masker, sort_order, perturbation_method)
    xs, ys, auc = sp.model_score(attributions, X, y=y)
    scores["metrics"].append(metric)
    scores["values"][metric] = [xs, ys, auc]


def get_benchmark(model, attributions, X, y, masker, metrics):
    # convert dataframes
    if isinstance(X, (pd.Series, pd.DataFrame)):
        X = X.values
    if isinstance(masker, (pd.Series, pd.DataFrame)):
        masker = masker.values

    # record scores per metric
    scores = {"metrics": list(), "values": dict()}
    for sort_order, perturbation_method in list(it.product(metrics["sort_order"], metrics["perturbation"])):
        update(model, attributions, X, y, masker, sort_order, perturbation_method, scores)

    return scores


def get_metrics(benchmarks, selection):
    # select metrics to plot using selection function
    explainer_metrics = set()
    for explainer in benchmarks:
        scores = benchmarks[explainer]
        if len(explainer_metrics) == 0:
            explainer_metrics = set(scores["metrics"])
        else:
            explainer_metrics = selection(explainer_metrics, set(scores["metrics"]))

    return list(explainer_metrics)


def trend_plot(benchmarks):
    explainer_metrics = get_metrics(benchmarks, lambda x, y: x.union(y))

    # plot all curves if metric exists
    for metric in explainer_metrics:
        plt.clf()

        for explainer in benchmarks:
            scores = benchmarks[explainer]
            if metric in scores["values"]:
                x, y, auc = scores["values"][metric]
                plt.plot(x, y, label=f"{round(auc, 3)} - {explainer}")

        if "keep" in metric:
            xlabel = "Percent Unmasked"
        if "remove" in metric:
            xlabel = "Percent Masked"

        plt.ylabel("Model Output")
        plt.xlabel(xlabel)
        plt.title(metric)
        plt.legend()
        plt.show()


def compare_plot(benchmarks):
    explainer_metrics = get_metrics(benchmarks, lambda x, y: x.intersection(y))
    explainers = list(benchmarks.keys())
    num_explainers = len(explainers)
    num_metrics = len(explainer_metrics)

    # dummy start to evenly distribute explainers on the left
    # can later be replaced by boolean metrics
    aucs = dict()
    for i in range(num_explainers):
        explainer = explainers[i]
        aucs[explainer] = [i / (num_explainers - 1)]

    # normalize per metric
    for metric in explainer_metrics:
        max_auc, min_auc = -float("inf"), float("inf")

        for explainer in explainers:
            scores = benchmarks[explainer]
            _, _, auc = scores["values"][metric]
            min_auc = min(auc, min_auc)
            max_auc = max(auc, max_auc)

        for explainer in explainers:
            scores = benchmarks[explainer]
            _, _, auc = scores["values"][metric]
            aucs[explainer].append((auc - min_auc) / (max_auc - min_auc))

    # plot common curves
    ax = plt.gca()
    for explainer in explainers:
        plt.plot(np.linspace(0, 1, len(explainer_metrics) + 1), aucs[explainer], "--o")

    ax.tick_params(which="major", axis="both", labelsize=8)

    ax.set_yticks([i / (num_explainers - 1) for i in range(0, num_explainers)])
    ax.set_yticklabels(explainers, rotation=0)

    ax.set_xticks(np.linspace(0, 1, num_metrics + 1))
    ax.set_xticklabels([" "] + explainer_metrics, rotation=45, ha="right")

    plt.grid(which="major", axis="x", linestyle="--")
    plt.tight_layout()
    plt.ylabel("Relative Performance of Each Explanation Method")
    plt.xlabel("Evaluation Metrics")
    plt.title("Explanation Method Performance Across Metrics")
    plt.show()

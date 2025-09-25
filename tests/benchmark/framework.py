import numpy as np

import shap


def model(x):
    return np.array([np.linalg.norm(x)])


X = np.array([[3, 4], [5, 12], [7, 24]])
y = np.array([5, 13, 25])
explainer = np.array([[-1, 2], [-4, 2], [1, 2]])
masker = X


def test_update():
    """This is to test the update function within benchmark/framework"""
    sort_order = "positive"

    def score_function(true, pred):
        return np.mean(pred)

    perturbation = "keep"
    scores = {"name": "test", "metrics": list(), "values": dict()}

    shap.benchmark.update(model, X, y, explainer, masker, sort_order, score_function, perturbation, scores)

    metric = perturbation + " " + sort_order

    assert scores["metrics"][0] == metric
    assert len(scores["values"][metric]) == 3


def test_get_benchmark():
    """This is to test the get benchmark function within benchmark/framework"""
    metrics = {"sort_order": ["positive", "negative"], "perturbation": ["keep"]}
    scores = shap.benchmark.get_benchmark(model, X, y, explainer, masker, metrics)

    expected_metrics = ["keep positive", "keep negative"]

    assert set(expected_metrics) == set(scores["metrics"])
    assert len(scores["values"]) == 2


def test_get_metrics():
    """This is to test the get metrics function with respect to different selection method"""
    scores1 = {"name": "test1", "metrics": ["keep positive", "keep absolute"], "values": dict()}
    scores2 = {"name": "test2", "metrics": ["keep positive", "keep negative"], "values": dict()}
    benchmarks = {"test1": scores1, "test2": scores2}

    expected_metrics1 = set(["keep positive"])
    expected_metrics2 = set(["keep positive", "keep negative", "keep absolute"])

    assert set(shap.benchmark.get_metrics(benchmarks, lambda x, y: x.intersection(y))) == expected_metrics1
    assert set(shap.benchmark.get_metrics(benchmarks, lambda x, y: x.union(y))) == expected_metrics2

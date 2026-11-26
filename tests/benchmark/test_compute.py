import numpy as np

import shap
from shap.benchmark import BenchmarkResult, ComputeTime


def test_compute_time_call():
    n_samples = 20
    n_features = 5
    elapsed_time = 4.0

    explanation = shap.Explanation(
        values=np.zeros((n_samples, n_features)),
        compute_time=elapsed_time,
    )

    compute_metric = ComputeTime()
    res = compute_metric(explanation, name="exact_explainer")

    assert isinstance(res, BenchmarkResult)
    assert res.metric == "compute time"
    assert res.method == "exact_explainer"
    assert np.isclose(res.value, elapsed_time / n_samples)
    assert res.value_sign == -1
    assert res.full_name == "exact_explainer compute time"


def test_compute_time_varying_sizes():
    compute_metric = ComputeTime()

    for n_samples, elapsed in [(1, 0.05), (100, 2.5), (500, 10.0)]:
        exp = shap.Explanation(
            values=np.ones((n_samples, 3)),
            compute_time=elapsed,
        )
        res = compute_metric(exp, name=f"explainer_{n_samples}")
        assert np.isclose(res.value, elapsed / n_samples)
        assert res.method == f"explainer_{n_samples}"

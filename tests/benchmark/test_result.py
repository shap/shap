import numpy as np
import pytest
import sklearn.metrics

from shap.benchmark import BenchmarkResult
from shap.benchmark._result import sign_defaults


def test_benchmark_result_init_explicit():
    curve_x = np.linspace(0, 1, 10)
    curve_y = curve_x**2
    curve_y_std = np.full(10, 0.05)

    res = BenchmarkResult(
        metric="custom_metric",
        method="my_explainer",
        value=0.42,
        curve_x=curve_x,
        curve_y=curve_y,
        curve_y_std=curve_y_std,
        value_sign=1,
    )

    assert res.metric == "custom_metric"
    assert res.method == "my_explainer"
    assert res.value == 0.42
    np.testing.assert_array_equal(res.curve_x, curve_x)
    np.testing.assert_array_equal(res.curve_y, curve_y)
    np.testing.assert_array_equal(res.curve_y_std, curve_y_std)
    assert res.value_sign == 1
    assert res.full_name == "my_explainer custom_metric"


@pytest.mark.parametrize(
    "metric,expected_sign",
    [
        ("keep positive", 1),
        ("keep negative", -1),
        ("remove positive", -1),
        ("remove negative", 1),
        ("compute time", -1),
        ("keep absolute", -1),
        ("remove absolute", 1),
        ("explanation error", -1),
    ],
)
def test_benchmark_result_sign_defaults(metric, expected_sign):
    assert sign_defaults[metric] == expected_sign
    res = BenchmarkResult(metric=metric, method="explainer", value=1.0)
    assert res.value_sign == expected_sign


def test_benchmark_result_unknown_metric_sign():
    res = BenchmarkResult(metric="unregistered_metric", method="explainer", value=1.0)
    assert res.value_sign is None


def test_benchmark_result_override_sign():
    res = BenchmarkResult(metric="compute time", method="explainer", value=0.5, value_sign=1)
    assert res.value_sign == 1


def test_benchmark_result_auc_auto_computation():
    curve_x = np.array([0.0, 0.5, 1.0])
    curve_y = np.array([1.0, 3.0, 5.0])  # curve_y - curve_y[0] = [0.0, 2.0, 4.0]
    expected_auc = sklearn.metrics.auc(curve_x, curve_y - curve_y[0])

    res = BenchmarkResult(
        metric="keep positive",
        method="linear_shap",
        curve_x=curve_x,
        curve_y=curve_y,
    )

    assert res.value is not None
    assert np.isclose(res.value, expected_auc)
    assert res.full_name == "linear_shap keep positive"
    assert res.value_sign == 1

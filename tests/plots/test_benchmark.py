import matplotlib.pyplot as plt
import numpy as np
import pytest

import shap
from shap.benchmark import BenchmarkResult


@pytest.mark.mpl_image_compare(tolerance=3)
def test_benchmark_single_with_curve():
    """Test benchmark plot with a single result that has a curve."""
    curve_x = np.linspace(0, 1, 50)
    curve_y = np.sin(curve_x * np.pi) * 0.5 + 0.5
    curve_y_std = np.ones_like(curve_y) * 0.1

    result = BenchmarkResult(
        metric="remove absolute",
        method="Method A",
        curve_x=curve_x,
        curve_y=curve_y,
        curve_y_std=curve_y_std
    )

    fig = plt.figure()
    shap.plots.benchmark(result, show=False)
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare(tolerance=3)
def test_benchmark_multiple_single_metric():
    """Test benchmark plot with multiple methods for a single metric."""
    curve_x = np.linspace(0, 1, 50)

    results = []
    for i, method in enumerate(["Method A", "Method B", "Method C"]):
        curve_y = np.sin(curve_x * np.pi + i * 0.5) * 0.5 + 0.5
        curve_y_std = np.ones_like(curve_y) * 0.05

        results.append(BenchmarkResult(
            metric="remove absolute",
            method=method,
            curve_x=curve_x,
            curve_y=curve_y,
            curve_y_std=curve_y_std
        ))

    fig = plt.figure()
    shap.plots.benchmark(results, show=False)
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare(tolerance=3)
def test_benchmark_single_metric_no_curves():
    """Test benchmark plot with single metric but no curves (bar plot)."""
    results = [
        BenchmarkResult(metric="compute time", method="Method A", value=0.5),
        BenchmarkResult(metric="compute time", method="Method B", value=0.3),
        BenchmarkResult(metric="compute time", method="Method C", value=0.8),
    ]

    fig = plt.figure()
    shap.plots.benchmark(results, show=False)
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare(tolerance=3)
def test_benchmark_multiple_metrics():
    """Test benchmark plot with multiple metrics (comparison plot)."""
    results = [
        BenchmarkResult(metric="remove absolute", method="Method A", value=0.85),
        BenchmarkResult(metric="remove absolute", method="Method B", value=0.75),
        BenchmarkResult(metric="remove absolute", method="Method C", value=0.65),
        BenchmarkResult(metric="compute time", method="Method A", value=0.50),
        BenchmarkResult(metric="compute time", method="Method B", value=0.30),
        BenchmarkResult(metric="compute time", method="Method C", value=0.40),
    ]

    fig = plt.figure(figsize=(8, 6))
    shap.plots.benchmark(results, show=False)
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare(tolerance=3)
def test_benchmark_keep_positive():
    """Test benchmark plot with 'keep positive' metric."""
    curve_x = np.linspace(0, 1, 30)
    curve_y = curve_x ** 2
    curve_y_std = np.ones_like(curve_y) * 0.05

    result = BenchmarkResult(
        metric="keep positive",
        method="Method A",
        curve_x=curve_x,
        curve_y=curve_y,
        curve_y_std=curve_y_std
    )

    fig = plt.figure()
    shap.plots.benchmark(result, show=False)
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare(tolerance=3)
def test_benchmark_explanation_error():
    """Test benchmark plot with explanation error metric."""
    results = [
        BenchmarkResult(metric="explanation error", method="Method A", value=0.15),
        BenchmarkResult(metric="explanation error", method="Method B", value=0.10),
        BenchmarkResult(metric="explanation error", method="Method C", value=0.20),
    ]

    fig = plt.figure()
    shap.plots.benchmark(results, show=False)
    plt.tight_layout()
    return fig


def test_benchmark_list_input():
    """Test that benchmark function accepts list input."""
    results = [
        BenchmarkResult(metric="remove absolute", method="Method A", value=0.85),
        BenchmarkResult(metric="remove absolute", method="Method B", value=0.75),
    ]

    shap.plots.benchmark(results, show=False)
    plt.close()


def test_benchmark_value_calculation():
    """Test that benchmark result value is calculated from curve if not provided."""
    curve_x = np.linspace(0, 1, 50)
    curve_y = np.linspace(0, 1, 50)

    result = BenchmarkResult(
        metric="remove absolute",
        method="Method A",
        curve_x=curve_x,
        curve_y=curve_y,
        curve_y_std=None
    )

    # Value should be calculated as AUC
    assert result.value is not None
    plt.close()

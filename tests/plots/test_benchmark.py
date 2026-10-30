"""This file contains tests for the benchmark plot."""

import pytest

import shap


def test_benchmark_empty_list_raises():
    """Check that a ValueError is raised when the list of benchmark results is empty."""
    with pytest.raises(ValueError, match="must not be empty"):
        shap.plots.benchmark([], show=False)

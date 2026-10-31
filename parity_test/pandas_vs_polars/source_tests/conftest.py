"""Reuse fixtures required by the copied SHAP test modules."""

from pathlib import Path

from tests.conftest import compare_numpy_outputs_against_baseline as _compare_against_upstream_baseline

pytest_plugins = (
    "tests.conftest",
    "tests.explainers.conftest",
    "tests.plots.conftest",
)


def compare_numpy_outputs_against_baseline(*, func_file, baseline_dir=None, rtol=1e-4, atol=1e-6):
    """Keep copied tests pointed at the repository's established baselines."""
    if baseline_dir is None:
        baseline_dir = Path(__file__).resolve().parents[3] / "tests" / "shap_values_baselines"
    return _compare_against_upstream_baseline(
        func_file=func_file,
        baseline_dir=baseline_dir,
        rtol=rtol,
        atol=atol,
    )

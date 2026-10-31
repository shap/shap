"""Reuse fixtures required by the copied SHAP test modules."""

from pathlib import Path

import pytest

from tests.conftest import (
    compare_numpy_outputs_against_baseline as _compare_against_upstream_baseline,
)
from tests.conftest import (
    global_random_seed,  # noqa: F401 - pytest fixture re-export
)
from tests.explainers.conftest import (  # noqa: F401 - pytest fixture re-exports
    basic_translation_scenario,
    causalml_synth_data,
)
from tests.plots.conftest import explainer, reset_style_to_default  # noqa: F401 - pytest fixture re-exports


@pytest.fixture
def random_seed():
    """Use a stable seed for copied tests without adding an upstream CLI option."""
    return 0


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

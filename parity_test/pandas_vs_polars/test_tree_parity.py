import os
import subprocess
import sys
import tempfile
from importlib import import_module
from pathlib import Path

import pytest

from parity_test._utils import assert_npz_groups_equal

hypothesis = pytest.importorskip("hypothesis")
given = hypothesis.given
settings = hypothesis.settings
st = import_module("hypothesis.strategies")


@settings(max_examples=1000, deadline=None, derandomize=True)
@given(random_state=st.integers(min_value=0, max_value=2**32 - 1))
def test_lightgbm_treeexplainer_pandas_matches_polars(random_state):
    pytest.importorskip("lightgbm")
    pytest.importorskip("polars")
    pytest.importorskip("pandas")

    repo_root = Path(__file__).resolve().parents[2]
    capture_script = Path(__file__).with_name("capture.py")
    artifact_dir_env = os.getenv("PARITY_ARTIFACT_DIR")
    if artifact_dir_env:
        output = (Path(artifact_dir_env) / f"pandas_polars_{random_state}.npz").resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [sys.executable, str(capture_script), "--random-state", str(random_state), "--output", str(output)],
            cwd=tempfile.gettempdir(),
            env={**os.environ, "PYTHONPATH": str(repo_root) + os.pathsep + os.getenv("PYTHONPATH", "")},
            check=True,
        )
        assert_npz_groups_equal(output, "pandas", "polars")
    else:
        with tempfile.TemporaryDirectory(prefix="shap-pandas-polars-parity-") as temp_dir:
            output = Path(temp_dir) / "outputs.npz"
            output = output.resolve()
            subprocess.run(
                [sys.executable, str(capture_script), "--random-state", str(random_state), "--output", str(output)],
                cwd=tempfile.gettempdir(),
                env={**os.environ, "PYTHONPATH": str(repo_root) + os.pathsep + os.getenv("PYTHONPATH", "")},
                check=True,
            )
            assert_npz_groups_equal(output, "pandas", "polars")

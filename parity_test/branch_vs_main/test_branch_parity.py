from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from contextlib import nullcontext
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

from parity_test._utils import assert_npz_equal

hypothesis = pytest.importorskip("hypothesis")
given = hypothesis.given
settings = hypothesis.settings
st = import_module("hypothesis.strategies")


CURRENT_ROOT = Path(__file__).resolve().parents[2]
CAPTURE_SCRIPT = Path(__file__).with_name("capture.py")
MAIN_ROOT_ENV = "SHAP_MAIN_ROOT"
MAIN_PYTHON_ENV = "SHAP_MAIN_PYTHON"


def _capture(python: str, repo_root: Path, random_state: int, input_path: Path, output: Path) -> None:
    output = output.resolve()
    input_path = input_path.resolve()
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    pythonpath_entries = [str(repo_root), str(CURRENT_ROOT)]
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    subprocess.run(
        [
            python,
            str(CAPTURE_SCRIPT),
            "--random-state",
            str(random_state),
            "--input",
            str(input_path),
            "--output",
            str(output),
        ],
        cwd=tempfile.gettempdir(),
        env=env,
        check=True,
    )


@pytest.mark.skipif(
    not (os.getenv(MAIN_ROOT_ENV) and os.getenv(MAIN_PYTHON_ENV)),
    reason=f"Set {MAIN_ROOT_ENV} and {MAIN_PYTHON_ENV} to a main checkout and its Python executable.",
)
@settings(max_examples=1000, deadline=None, derandomize=True)
@given(random_state=st.integers(min_value=0, max_value=2**32 - 1))
def test_current_branch_matches_main(random_state: int) -> None:
    main_root = Path(os.environ[MAIN_ROOT_ENV]).resolve()
    main_python = os.environ[MAIN_PYTHON_ENV]
    candidate_python = sys.executable
    artifact_dir_env = os.getenv("PARITY_ARTIFACT_DIR")
    temp_context = tempfile.TemporaryDirectory(prefix="shap-branch-parity-") if not artifact_dir_env else nullcontext()
    with temp_context as temp_dir:
        artifact_root = Path(artifact_dir_env) if artifact_dir_env else Path(temp_dir)
        artifact_root.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(random_state)
        X = rng.normal(size=(72, 7))
        regression_y = X[:, 0] * 0.8 - X[:, 2] * 0.35 + rng.normal(scale=0.1, size=X.shape[0])
        classification_y = (X[:, 0] + X[:, 1] * 0.5 > 0).astype(np.int8)
        input_path = artifact_root / f"input_{random_state}.npz"
        np.savez_compressed(input_path, X=X, regression_y=regression_y, classification_y=classification_y)
        main_artifact = artifact_root / f"main_{random_state}.npz"
        candidate_artifact = artifact_root / f"current_{random_state}.npz"

        _capture(main_python, main_root, random_state, input_path, main_artifact)
        _capture(candidate_python, CURRENT_ROOT, random_state, input_path, candidate_artifact)
        assert_npz_equal(main_artifact, candidate_artifact)

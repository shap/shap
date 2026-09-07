r"""Smoke tests for the Pyodide/JupyterLite build.

These tests are deliberately lightweight - they verify that:
  1. shap can be imported after installation via micropip.
  2. ExactExplainer, KernelExplainer, PartitionExplainer, and
     TreeExplainer (scikit-learn RandomForest) all produce outputs of
     the correct shape.

The tests are *skipped* on all non-Emscripten platforms so they do not
interfere with the normal test suite.  On Pyodide they are run by the
build_pyodide_wheel.yml CI workflow inside a Node-backed Pyodide runtime.

To run locally inside Pyodide Node (after building the wheel):

    node -e "
      const { loadPyodide } = require('pyodide');
      (async () => {
        const py = await loadPyodide();
        await py.loadPackage('micropip');
        await py.runPythonAsync(`
          import micropip
          await micropip.install('file:///path/to/shap-*.whl')
          import subprocess, sys
          subprocess.run([sys.executable, '-m', 'pytest',
                         'tests/test_pyodide.py', '-v'], check=True)
        `);
      })();
    "
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

# Skip the entire module on non-Emscripten platforms.
pytestmark = pytest.mark.skipif(
    sys.platform != "emscripten",
    reason="Pyodide-only tests - skipped on non-Emscripten platforms.",
)


@pytest.fixture(scope="module")
def simple_tabular_data() -> tuple[np.ndarray, np.ndarray]:
    """Return a small (20, 4) feature matrix and binary labels."""
    rng = np.random.RandomState(42)
    X = rng.randn(20, 4).astype(np.float64)
    y = (X[:, 0] > 0).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Import smoke test
# ---------------------------------------------------------------------------


def test_import_shap() -> None:
    """shap must import without error on Pyodide."""
    import shap  # noqa: F401 - presence is the test

    assert hasattr(shap, "__version__"), "shap.__version__ not found"


# ---------------------------------------------------------------------------
# ExactExplainer
# ---------------------------------------------------------------------------


def test_exact_explainer(simple_tabular_data: tuple[np.ndarray, np.ndarray]) -> None:
    """ExactExplainer uses _cutils (grey-code kernel) - core CPU path."""
    from sklearn.linear_model import LogisticRegression

    import shap

    X, y = simple_tabular_data
    model = LogisticRegression(random_state=0).fit(X, y)
    masker = shap.maskers.Independent(X, max_samples=10)
    explainer = shap.ExactExplainer(model.predict_proba, masker)
    sv = explainer(X[:5])
    assert isinstance(sv, shap.Explanation)
    assert sv.values.shape[:2] == (5, 4), f"unexpected shape {sv.values.shape}"


# ---------------------------------------------------------------------------
# KernelExplainer
# ---------------------------------------------------------------------------


def test_kernel_explainer(simple_tabular_data: tuple[np.ndarray, np.ndarray]) -> None:
    """KernelExplainer uses _cutils (compute_exp_val) - core CPU path."""
    from sklearn.linear_model import LogisticRegression

    import shap

    X, y = simple_tabular_data
    model = LogisticRegression(random_state=0).fit(X, y)
    explainer = shap.KernelExplainer(model.predict_proba, X[:10])
    sv = explainer.shap_values(X[:3], nsamples=32)
    assert len(sv) == 2, f"expected 2 output classes, got {len(sv)}"
    assert sv[0].shape == (3, 4), f"unexpected shape {sv[0].shape}"


# ---------------------------------------------------------------------------
# PartitionExplainer
# ---------------------------------------------------------------------------


def test_partition_explainer(
    simple_tabular_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """PartitionExplainer uses _cutils (lower_credit, delta_masking) - CPU path."""
    from sklearn.linear_model import LogisticRegression

    import shap

    X, y = simple_tabular_data
    model = LogisticRegression(random_state=0).fit(X, y)
    masker = shap.maskers.Partition(X)
    explainer = shap.PartitionExplainer(model.predict_proba, masker)
    sv = explainer(X[:3])
    assert isinstance(sv, shap.Explanation)
    assert sv.values.shape[:2] == (3, 4), f"unexpected shape {sv.values.shape}"


# ---------------------------------------------------------------------------
# TreeExplainer (scikit-learn) - uses _cext Tree SHAP
# ---------------------------------------------------------------------------


def test_tree_explainer_sklearn(
    simple_tabular_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """TreeExplainer uses _cext (Tree SHAP C++ extension) - core CPU path."""
    from sklearn.ensemble import RandomForestClassifier

    import shap

    X, y = simple_tabular_data
    model = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
    explainer = shap.TreeExplainer(model)
    sv = explainer(X[:5])
    assert isinstance(sv, shap.Explanation)
    # RandomForest binary classification -> shape (n_samples, n_features, n_classes)
    assert sv.values.shape == (5, 4, 2), f"unexpected shape {sv.values.shape}"


# ---------------------------------------------------------------------------
# Typing stubs shipped in the wheel
# ---------------------------------------------------------------------------


def test_py_typed_present() -> None:
    """py.typed must be present in the installed shap package."""
    import importlib.resources as ir

    import shap

    pkg = ir.files(shap)
    assert (pkg / "py.typed").is_file(), "py.typed not found in installed shap package"


def test_cutils_pyi_present() -> None:
    """_cutils.pyi must be present in the installed shap package."""
    import importlib.resources as ir

    import shap

    pkg = ir.files(shap)
    assert (pkg / "_cutils.pyi").is_file(), "_cutils.pyi not found in installed shap package"

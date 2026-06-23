import json
import os
import subprocess
import sys


def _run_in_subprocess(*, eager_import: bool) -> dict[str, bool]:
    env = os.environ.copy()
    env["EAGER_IMPORT"] = "1" if eager_import else "0"

    cmd = [
        sys.executable,
        "-c",
        (
            "import json, sys; "
            "import shap; "
            "print(json.dumps({"
            "'tree_loaded': 'shap.explainers._tree' in sys.modules, "
            "'kernel_loaded': 'shap.explainers._kernel' in sys.modules, "
            "'plots_loaded': 'shap.plots' in sys.modules"
            "}))"
        ),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
    return json.loads(proc.stdout.strip())


def test_import_shap_is_lazy_by_default() -> None:
    loaded = _run_in_subprocess(eager_import=False)
    assert loaded["tree_loaded"] is False
    assert loaded["kernel_loaded"] is False


def test_eager_import_loads_public_members() -> None:
    loaded = _run_in_subprocess(eager_import=True)
    assert loaded["tree_loaded"] is True
    assert loaded["kernel_loaded"] is True


def test_public_api_still_accessible() -> None:
    import shap

    assert callable(shap.TreeExplainer)
    assert callable(shap.KernelExplainer)
    assert callable(shap.bar_plot)
    assert callable(shap.summary_plot)
    assert hasattr(shap, "datasets")
    assert hasattr(shap, "utils")

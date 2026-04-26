import json
import os
import subprocess
import sys

import pytest


def _run_python(code: str, *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    full_env = os.environ.copy()
    # Keep subprocess checks isolated from outer test runner eager-import mode.
    full_env.pop("EAGER_IMPORT", None)
    if env:
        full_env.update(env)
    return subprocess.run(
        [sys.executable, "-P", "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=full_env,
    )


@pytest.mark.parametrize(
    "module_name",
    [
        "shap.explainers._tree",
        "shap.explainers._kernel",
        "shap.plots",
        "shap.plots._bar",
    ],
)
def test_import_shap_does_not_eagerly_import_heavy_modules(module_name: str):
    proc = _run_python(
        f"import json, sys\nimport shap\nprint(json.dumps({{'loaded': '{module_name}' in sys.modules}}))\n"
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip())
    assert payload["loaded"] is False


def test_eager_import_loads_lazy_public_names():
    proc = _run_python(
        "import json, sys\n"
        "import shap\n"
        "print(json.dumps({\n"
        "  'tree_loaded': 'shap.explainers._tree' in sys.modules,\n"
        "  'plots_loaded': 'shap.plots' in sys.modules,\n"
        "  'bar_loaded': 'shap.plots._bar' in sys.modules,\n"
        "}))\n",
        env={"EAGER_IMPORT": "1"},
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip())

    assert payload["tree_loaded"] is True

    has_matplotlib = _run_python("import importlib.util; print(importlib.util.find_spec('matplotlib') is not None)")
    assert has_matplotlib.returncode == 0
    matplotlib_available = has_matplotlib.stdout.strip() == "True"

    if matplotlib_available:
        assert payload["plots_loaded"] is True
        assert payload["bar_loaded"] is True


def test_all_public_api_names_are_accessible():
    proc = _run_python(
        "import importlib.util, json\n"
        "import shap\n"
        "missing = []\n"
        "has_matplotlib = importlib.util.find_spec('matplotlib') is not None\n"
        "has_cext = importlib.util.find_spec('shap._cext') is not None\n"
        "has_cext_gpu = importlib.util.find_spec('shap._cext_gpu') is not None\n"
        "plot_names = {\n"
        "  'plots', 'bar_plot', 'summary_plot', 'decision_plot', 'multioutput_decision_plot',\n"
        "  'embedding_plot', 'force_plot', 'getjs', 'initjs', 'save_html', 'group_difference_plot',\n"
        "  'heatmap_plot', 'image_plot', 'monitoring_plot', 'partial_dependence_plot',\n"
        "  'dependence_plot', 'text_plot', 'violin_plot', 'waterfall_plot'\n"
        "}\n"
        "optional_cext_names = {'_cext', '_cext_gpu'}\n"
        "for name in shap.__all__:\n"
        "  try:\n"
        "    getattr(shap, name)\n"
        "  except Exception as exc:\n"
        "    if ((has_matplotlib or name not in plot_names) and\n"
        "        ((has_cext and has_cext_gpu) or name not in optional_cext_names)):\n"
        "      missing.append([name, type(exc).__name__, str(exc)])\n"
        "print(json.dumps(missing))\n"
    )
    assert proc.returncode == 0, proc.stderr
    missing = json.loads(proc.stdout.strip())
    assert missing == []

import json
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_python(code: str, *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    full_env = os.environ.copy()
    full_env.pop("EAGER_IMPORT", None)
    full_env["PYTHONPATH"] = os.pathsep.join(
        path for path in [str(_repo_root().parent), full_env.get("PYTHONPATH", "")] if path
    )
    if env:
        full_env.update(env)
    return subprocess.run(
        [sys.executable, "-P", "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=full_env,
    )


def test_import_shap_is_lazy_by_default() -> None:
    proc = _run_python(
        "import json, sys\n"
        "import shap\n"
        "print(json.dumps({\n"
        "  'actions_loaded': 'shap.actions' in sys.modules,\n"
        "  'explainers_loaded': 'shap.explainers' in sys.modules,\n"
        "  'maskers_loaded': 'shap.maskers' in sys.modules,\n"
        "  'models_loaded': 'shap.models' in sys.modules,\n"
        "  'plots_loaded': 'shap.plots' in sys.modules,\n"
        "  'explanation_loaded': 'shap._explanation' in sys.modules,\n"
        "  'tree_loaded': 'shap.explainers._tree' in sys.modules,\n"
        "  'kernel_loaded': 'shap.explainers._kernel' in sys.modules,\n"
        "  'deep_loaded': 'shap.explainers._deep' in sys.modules,\n"
        "  'matplotlib_loaded': 'matplotlib' in sys.modules,\n"
        "  'numba_loaded': 'numba' in sys.modules,\n"
        "  'pandas_loaded': 'pandas' in sys.modules,\n"
        "  'scipy_loaded': 'scipy' in sys.modules,\n"
        "  'sklearn_loaded': 'sklearn' in sys.modules\n"
        "}))\n"
    )
    assert proc.returncode == 0, proc.stderr
    loaded = json.loads(proc.stdout.strip())
    assert loaded == {
        "actions_loaded": False,
        "deep_loaded": False,
        "explainers_loaded": False,
        "explanation_loaded": False,
        "kernel_loaded": False,
        "maskers_loaded": False,
        "matplotlib_loaded": False,
        "models_loaded": False,
        "numba_loaded": False,
        "pandas_loaded": False,
        "plots_loaded": False,
        "scipy_loaded": False,
        "sklearn_loaded": False,
        "tree_loaded": False,
    }


def test_eager_import_loads_required_lazy_names() -> None:
    proc = _run_python(
        "import importlib.util, json, sys\n"
        "import shap\n"
        "print(json.dumps({\n"
        "  'actions_loaded': 'shap.actions' in sys.modules,\n"
        "  'explainers_loaded': 'shap.explainers' in sys.modules,\n"
        "  'maskers_loaded': 'shap.maskers' in sys.modules,\n"
        "  'models_loaded': 'shap.models' in sys.modules,\n"
        "  'plots_loaded': 'shap.plots' in sys.modules,\n"
        "  'explanation_loaded': 'shap._explanation' in sys.modules,\n"
        "  'tree_loaded': 'shap.explainers._tree' in sys.modules,\n"
        "  'kernel_loaded': 'shap.explainers._kernel' in sys.modules,\n"
        "  'deep_loaded': 'shap.explainers._deep' in sys.modules,\n"
        "  'matplotlib_loaded': 'matplotlib' in sys.modules,\n"
        "  'numba_loaded': 'numba' in sys.modules,\n"
        "  'pandas_loaded': 'pandas' in sys.modules,\n"
        "  'scipy_loaded': 'scipy' in sys.modules,\n"
        "  'sklearn_loaded': 'sklearn' in sys.modules,\n"
        "  'matplotlib_available': importlib.util.find_spec('matplotlib') is not None\n"
        "}))\n",
        env={"EAGER_IMPORT": "1"},
    )
    assert proc.returncode == 0, proc.stderr
    loaded = json.loads(proc.stdout.strip())

    assert loaded["actions_loaded"] is True
    assert loaded["deep_loaded"] is True
    assert loaded["explainers_loaded"] is True
    assert loaded["explanation_loaded"] is True
    assert loaded["kernel_loaded"] is True
    assert loaded["maskers_loaded"] is True
    assert loaded["models_loaded"] is True
    assert loaded["numba_loaded"] is True
    assert loaded["pandas_loaded"] is True
    assert loaded["scipy_loaded"] is True
    assert loaded["sklearn_loaded"] is True
    assert loaded["tree_loaded"] is True
    if loaded["matplotlib_available"]:
        assert loaded["matplotlib_loaded"] is True
        assert loaded["plots_loaded"] is True


def test_public_api_surface_is_preserved() -> None:
    import shap

    assert shap.__all__ == [
        "Cohorts",
        "Explanation",
        "other",
        "AdditiveExplainer",
        "DeepExplainer",
        "ExactExplainer",
        "Explainer",
        "GPUTreeExplainer",
        "GradientExplainer",
        "KernelExplainer",
        "LinearExplainer",
        "PartitionExplainer",
        "CoalitionExplainer",
        "PermutationExplainer",
        "SamplingExplainer",
        "TreeExplainer",
        "plots",
        "bar_plot",
        "summary_plot",
        "decision_plot",
        "multioutput_decision_plot",
        "embedding_plot",
        "force_plot",
        "getjs",
        "initjs",
        "save_html",
        "group_difference_plot",
        "heatmap_plot",
        "image_plot",
        "monitoring_plot",
        "partial_dependence_plot",
        "dependence_plot",
        "text_plot",
        "violin_plot",
        "waterfall_plot",
        "datasets",
        "links",
        "utils",
        "ActionOptimizer",
        "approximate_interactions",
        "sample",
        "kmeans",
    ]

    assert hasattr(shap, "actions")
    assert hasattr(shap, "datasets")
    assert hasattr(shap, "explainers")
    assert hasattr(shap, "links")
    assert hasattr(shap, "maskers")
    assert hasattr(shap, "models")
    assert hasattr(shap, "plots")
    assert hasattr(shap, "utils")

    assert callable(shap.ActionOptimizer)
    assert callable(shap.actions.Action)
    assert callable(shap.explainers.Tree)
    assert hasattr(shap.explainers, "other")
    assert callable(shap.maskers.Text)
    assert callable(shap.models.Model)
    assert callable(shap.plots.bar)
    assert callable(shap.utils.sample)


def test_top_level_plot_fallback_without_matplotlib() -> None:
    proc = _run_python(
        "import builtins, json\n"
        "real_import = builtins.__import__\n"
        "def fake_import(name, globals=None, locals=None, fromlist=(), level=0):\n"
        "    if name == 'matplotlib':\n"
        "        raise ImportError('blocked matplotlib')\n"
        "    return real_import(name, globals, locals, fromlist, level)\n"
        "builtins.__import__ = fake_import\n"
        "import shap\n"
        "plots = shap.plots\n"
        "plots_error = None\n"
        "bar_error = None\n"
        "try:\n"
        "    plots.bar\n"
        "except Exception as exc:\n"
        "    plots_error = [type(exc).__name__, str(exc)]\n"
        "try:\n"
        "    shap.bar_plot()\n"
        "except Exception as exc:\n"
        "    bar_error = [type(exc).__name__, str(exc)]\n"
        "print(json.dumps({'plots_type': type(plots).__name__, 'plots_error': plots_error, 'bar_error': bar_error}))\n"
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip())

    assert payload["plots_type"] == "UnsupportedModule"
    assert payload["plots_error"][0] == "ImportError"
    assert "matplotlib is not installed" in payload["plots_error"][1]
    assert payload["bar_error"][0] == "ImportError"
    assert "matplotlib is not installed" in payload["bar_error"][1]
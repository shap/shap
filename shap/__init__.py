import os
from importlib import import_module
from typing import Any, NoReturn

import lazy_loader as lazy


def _eager_import_enabled() -> bool:
    value = os.environ.get("EAGER_IMPORT", "")
    return value not in {"", "0", "false", "False"}


_eager_import_requested = _eager_import_enabled()
_previous_eager_import = os.environ.get("EAGER_IMPORT")
if _eager_import_requested:
    os.environ["EAGER_IMPORT"] = "0"

try:
    _stub_getattr, _, _ = lazy.attach_stub(__name__, __file__)
finally:
    if _eager_import_requested:
        if _previous_eager_import is None:
            os.environ.pop("EAGER_IMPORT", None)
        else:
            os.environ["EAGER_IMPORT"] = _previous_eager_import

try:
    # Version from setuptools-scm
    from ._version import version as __version__
except ImportError:
    # Expected when running locally without build
    __version__ = "0.0.0-not-built"

_no_matplotlib_warning = (
    "matplotlib is not installed so plotting is not available! Run `pip install matplotlib` to fix this."
)

_LAZY_SUBMODULES = {"actions", "datasets", "explainers", "links", "maskers", "models", "utils"}
_TOP_LEVEL_PLOT_EXPORTS = {
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
}


# plotting (only loaded if matplotlib is present)


def unsupported(*args: Any, **kwargs: Any) -> NoReturn:
    raise ImportError(_no_matplotlib_warning)


class UnsupportedModule:
    def __getattribute__(self, item: str) -> NoReturn:
        raise ImportError(_no_matplotlib_warning)


def _load_submodule(name: str) -> Any:
    value = import_module(f"{__name__}.{name}")
    globals()[name] = value
    return value


def __getattr__(name: str) -> Any:
    if name == "plots":
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            value: Any = UnsupportedModule()
            globals()[name] = value
            return value
        return _load_submodule(name)

    if name in _TOP_LEVEL_PLOT_EXPORTS:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            globals()[name] = unsupported
            return unsupported

    if name in _LAZY_SUBMODULES:
        return _load_submodule(name)

    value = _stub_getattr(name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | _LAZY_SUBMODULES | {"plots", "__version__"})


__all__ = [
    "Cohorts",
    "Explanation",
    # Explainers
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
    # Plots
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
    # Other stuff
    "datasets",
    "links",
    "utils",
    "ActionOptimizer",
    "approximate_interactions",
    "sample",
    "kmeans",
]

if _eager_import_requested:
    for _name in sorted(set(__all__) | _LAZY_SUBMODULES | {"plots"}):
        __getattr__(_name)

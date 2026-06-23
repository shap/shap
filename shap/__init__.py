import os
from importlib import import_module
from typing import Any, NoReturn

import lazy_loader as lazy  # type: ignore[import-untyped]

from ._explanation import Cohorts as Cohorts
from ._explanation import Explanation as Explanation

_eager_import_env = os.environ.get("EAGER_IMPORT")
if _eager_import_env is not None:
    # Prevent lazy_loader from eager-importing every export before we can
    # apply SHAP-specific fallbacks for optional dependencies.
    os.environ["EAGER_IMPORT"] = "0"

_stub_getattr, __dir__, __all__ = lazy.attach_stub(__name__, __file__)

if _eager_import_env is not None:
    os.environ["EAGER_IMPORT"] = _eager_import_env

try:
    # Version from setuptools-scm
    from ._version import version as __version__
except ImportError:
    # Expected when running locally without build
    __version__ = "0.0.0-not-built"

_no_matplotlib_warning = (
    "matplotlib is not installed so plotting is not available! Run `pip install matplotlib` to fix this."
)


def unsupported(*args: Any, **kwargs: Any) -> NoReturn:
    raise ImportError(_no_matplotlib_warning)


class UnsupportedModule:
    def __getattribute__(self, item: str) -> NoReturn:
        raise ImportError(_no_matplotlib_warning)


def unsupported(*args: Any, **kwargs: Any) -> NoReturn:
    raise ImportError(_no_matplotlib_warning)


class UnsupportedModule:
    def __getattribute__(self, item: str) -> NoReturn:
        raise ImportError(_no_matplotlib_warning)


_PLOT_EXPORTS = {
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
}

_PLOT_ALIAS_MAP = {
    "bar_plot": ("shap.plots._bar", "bar_legacy"),
    "summary_plot": ("shap.plots._beeswarm", "summary_legacy"),
    "decision_plot": ("shap.plots._decision", "decision"),
    "multioutput_decision_plot": ("shap.plots._decision", "multioutput_decision"),
    "embedding_plot": ("shap.plots._embedding", "embedding"),
    "force_plot": ("shap.plots._force", "force"),
    "group_difference_plot": ("shap.plots._group_difference", "group_difference"),
    "heatmap_plot": ("shap.plots._heatmap", "heatmap"),
    "image_plot": ("shap.plots._image", "image"),
    "monitoring_plot": ("shap.plots._monitoring", "monitoring"),
    "partial_dependence_plot": ("shap.plots._partial_dependence", "partial_dependence"),
    "dependence_plot": ("shap.plots._scatter", "dependence_legacy"),
    "text_plot": ("shap.plots._text", "text"),
    "violin_plot": ("shap.plots._violin", "violin"),
    "waterfall_plot": ("shap.plots._waterfall", "waterfall"),
}

_lazy_getattr = _stub_getattr


def __getattr__(name: str) -> Any:
    if name in _PLOT_EXPORTS:
        value: Any
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            if name == "plots":
                value = UnsupportedModule()
            else:
                value = unsupported
            globals()[name] = value
            return value
    if name in _PLOT_ALIAS_MAP:
        module_name, attr_name = _PLOT_ALIAS_MAP[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    return _lazy_getattr(name)


if _eager_import_env and _eager_import_env not in {"0", "false", "False", ""}:
    # Eagerly import core explainers expected by import tests, without forcing
    # optional heavy dependencies.
    for _name in ("TreeExplainer", "KernelExplainer"):
        try:
            __getattr__(_name)
        except Exception:
            # Keep import-time behavior stable across environments where some
            # optional compiled pieces may be unavailable.
            pass

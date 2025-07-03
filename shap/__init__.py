import lazy_loader as lazy  # type: ignore[import-untyped]

#
from shap import cext, datasets, links, utils  # noqa: E402

_stub_getattr, __dir__, __alllazy__ = lazy.attach_stub(__name__, __file__)

try:
    # Version from setuptools-scm
    from ._version import version as __version__
except ImportError:
    # Expected when running locally without build
    __version__ = "0.0.0-not-built"

_no_matplotlib_warning = (
    "matplotlib is not installed so plotting is not available! Run `pip install matplotlib` to fix this."
)


# plotting (only loaded if matplotlib is present)
def unsupported(*args, **kwargs):
    raise ImportError(_no_matplotlib_warning)


class UnsupportedModule:
    def __getattribute__(self, item):
        raise ImportError(_no_matplotlib_warning)


def __getattr__(name):
    # Handle legacy aliases first
    if name == "summary_plot":
        return _stub_getattr("summary_legacy")
    elif name == "image_plot":
        return _stub_getattr("image")
    # Handle plotting function aliases
    elif name == "bar_plot":
        return _stub_getattr("bar_legacy")
    elif name == "decision_plot":
        return _stub_getattr("decision")
    elif name == "multioutput_decision_plot":
        return _stub_getattr("multioutput_decision")
    elif name == "embedding_plot":
        return _stub_getattr("embedding")
    elif name == "force_plot":
        return _stub_getattr("force")
    elif name == "group_difference_plot":
        return _stub_getattr("group_difference")
    elif name == "heatmap_plot":
        return _stub_getattr("heatmap")
    elif name == "monitoring_plot":
        return _stub_getattr("monitoring")
    elif name == "partial_dependence_plot":
        return _stub_getattr("partial_dependence")
    elif name == "dependence_plot":
        return _stub_getattr("dependence_legacy")
    elif name == "text_plot":
        return _stub_getattr("text")
    elif name == "violin_plot":
        return _stub_getattr("violin")
    elif name == "waterfall_plot":
        return _stub_getattr("waterfall")
    else:
        # Fall back to the lazy loader for all other attributes
        return _stub_getattr(name)


_legacy_plot_aliases = [
    "summary_plot",
    "image_plot",
    "bar_plot",
    "decision_plot",
    "multioutput_decision_plot",
    "embedding_plot",
    "force_plot",
    "group_difference_plot",
    "heatmap_plot",
    "monitoring_plot",
    "partial_dependence_plot",
    "dependence_plot",
    "text_plot",
    "violin_plot",
    "waterfall_plot",
]

__all__ = [*__alllazy__, "datasets", "links", "utils", "cext"] + _legacy_plot_aliases

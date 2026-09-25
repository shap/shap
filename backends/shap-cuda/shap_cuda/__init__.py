from ._gpu_tree import GPUTreeExplainer

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0-not-built"

__all__ = ["GPUTreeExplainer"]

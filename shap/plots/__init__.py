import lazy_loader as lazy

try:
    matplotlib = lazy.load("matplotlib", error_on_import=True)
except ImportError:
    raise ImportError(
        "matplotlib is not installed so plotting is not available! Run `pip install matplotlib` to fix this."
    )

# Use lazy.attach_stub to enable proper type checking for plots
__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)

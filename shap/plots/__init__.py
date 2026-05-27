import os

import lazy_loader as lazy

try:
    import matplotlib  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "matplotlib is not installed so plotting is not available! Run `pip install matplotlib` to fix this."
    ) from exc

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)

if os.getenv("EAGER_IMPORT") == "1":
    for _name in __all__:
        try:
            __getattr__(_name)
        except Exception:
            pass

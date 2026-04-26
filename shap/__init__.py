import os

import lazy_loader as lazy

from ._explanation import Cohorts as Cohorts
from ._explanation import Explanation as Explanation

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)


try:
    # Version from setuptools-scm
    from ._version import version as __version__
except ImportError:
    # Expected when running locally without build
    __version__ = "0.0.0-not-built"

# SPEC 1: https://scientific-python.org/specs/spec-0001/
# The adjacent __init__.pyi stub is the single source of truth —
# lazy_loader parses it at runtime to build the lazy-load map, and
# type checkers use it for autocompletion. Only add public names there.
# Set EAGER_IMPORT=1 to force all lazy names to resolve immediately,
# which is used in CI to catch broken imports early.

if os.getenv("EAGER_IMPORT") == "1":
    for _name in __all__:
        try:
            __getattr__(_name)
        except Exception:
            pass

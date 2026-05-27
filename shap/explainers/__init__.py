import os

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)

if os.getenv("EAGER_IMPORT") == "1":
    for _name in __all__:
        try:
            __getattr__(_name)
        except Exception:
            pass

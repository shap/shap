"""Strategy-based dispatch for tree model libraries.

Each supported library (sklearn, xgboost, lightgbm, etc.) has its own
handler class that knows how to convert that library's model into the
internal TreeEnsemble representation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...utils._exceptions import InvalidModelError
from ._base import TreeModelHandler

# import handler classes
from ._catboost import CatBoostHandler
from ._gpboost import GPBoostHandler
from ._lightgbm import LightGBMHandler
from ._ngboost import NGBoostHandler
from ._other import DictHandler, ImbLearnHandler, PyODHandler
from ._pyspark import PySparkHandler
from ._sklearn import SklearnHandler
from ._xgboost import XGBoostHandler

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from .._tree import TreeEnsemble

# Order matters: dict/list must be checked first since isinstance
# won't work for them, and pyod before pyspark.
_HANDLERS: list[type[TreeModelHandler]] = [
    DictHandler,  # dict/list formats (checked first)
    SklearnHandler,  # sklearn, econml, causalml, skopt
    PyODHandler,  # pyod (before pyspark, uses safe_isinstance)
    PySparkHandler,  # pyspark (uses str(type(model)) check)
    XGBoostHandler,  # xgboost
    LightGBMHandler,  # lightgbm
    GPBoostHandler,  # gpboost
    CatBoostHandler,  # catboost
    ImbLearnHandler,  # imblearn
    NGBoostHandler,  # ngboost
]


def get_handler(model: Any) -> type[TreeModelHandler]:
    """Find the handler class that supports the given model."""
    for handler in _HANDLERS:
        if handler.can_handle(model):
            return handler
    raise InvalidModelError("Model type not yet supported by TreeExplainer: " + str(type(model)))


def _dispatch(
    model: Any,
    ensemble: TreeEnsemble,
    data: npt.NDArray[Any] | None,
    data_missing: npt.NDArray[np.bool_] | None,
) -> None:
    """Find the right handler for *model* and use it to populate *ensemble*."""
    handler = get_handler(model)
    handler.handle(model, ensemble, data, data_missing)


__all__ = [
    "TreeModelHandler",
    "_dispatch",
    "get_handler",
]

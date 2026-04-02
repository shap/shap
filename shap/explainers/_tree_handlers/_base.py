"""Abstract base class for tree model handlers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from .._tree import TreeEnsemble

# Maps library-specific objective/criterion names to our internal names.
# Uses keras-style naming conventions.
OBJECTIVE_NAME_MAP: dict[str, str] = {
    "mse": "squared_error",
    "variance": "squared_error",
    "friedman_mse": "squared_error",
    "reg:linear": "squared_error",
    "reg:squarederror": "squared_error",
    "regression": "squared_error",
    "regression_l2": "squared_error",
    "mae": "absolute_error",
    "gini": "binary_crossentropy",
    "entropy": "binary_crossentropy",
    "reg:logistic": "binary_crossentropy",
    "binary:logistic": "binary_crossentropy",
    "binary_logloss": "binary_crossentropy",
    "binary": "binary_crossentropy",
}

TREE_OUTPUT_NAME_MAP: dict[str, str] = {
    "regression": "raw_value",
    "regression_l2": "squared_error",
    "reg:linear": "raw_value",
    "reg:squarederror": "raw_value",
    "reg:logistic": "log_odds",
    "binary:logistic": "log_odds",
    "binary_logloss": "log_odds",
    "binary": "log_odds",
}


class TreeModelHandler(ABC):
    """Abstract handler that each library-specific module must subclass."""

    @staticmethod
    @abstractmethod
    def can_handle(model: Any) -> bool:
        """Return True if this handler supports the given model."""

    @staticmethod
    @abstractmethod
    def handle(
        model: Any,
        ensemble: TreeEnsemble,
        data: npt.NDArray[Any] | None,
        data_missing: npt.NDArray[np.bool_] | None,
    ) -> None:
        """Populate TreeEnsemble attributes for the given model."""

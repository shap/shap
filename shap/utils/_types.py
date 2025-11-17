from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse

if TYPE_CHECKING:
    from collections.abc import Callable

_ArrayLike: TypeAlias = np.ndarray | pd.DataFrame | pd.Series | list | scipy.sparse.spmatrix
_ArrayT = TypeVar("_ArrayT", np.ndarray, pd.DataFrame, pd.Series, scipy.sparse.spmatrix, list)


# Model protocols - duck typing for ML models
class _PredictProtocol(Protocol):
    """Protocol for models with a predict method."""

    def predict(self, X: npt.NDArray[Any] | pd.DataFrame, /) -> npt.NDArray[Any]: ...


class _PredictProbaProtocol(Protocol):
    """Protocol for models with a predict_proba method."""

    def predict_proba(self, X: npt.NDArray[Any] | pd.DataFrame, /) -> npt.NDArray[Any]: ...


class _ModelWithPredict(Protocol):
    """Protocol for models that have predict and optionally predict_proba."""

    def predict(self, X: npt.NDArray[Any] | pd.DataFrame, /) -> npt.NDArray[Any]: ...


# Callable model (a function that takes input and returns output)
_CallableModel: TypeAlias = Callable[[npt.NDArray[Any] | pd.DataFrame], npt.NDArray[Any]]

# General model type - can be a model object with predict, or a callable
# Note: Use this for input parameters. Instance attributes after model wrapping may be typed as Callable or specific types.
# Callable is listed first as most models are wrapped to be callable
_Model: TypeAlias = _CallableModel | _ModelWithPredict | Any  # Any for flexibility with various ML frameworks

# Link function type
_LinkFunction: TypeAlias = Callable[[npt.NDArray[Any]], npt.NDArray[Any]]

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
# Note: Union types cause mypy false positives. Use # type: ignore[union-attr] or # type: ignore[operator]
# where model attributes are accessed, since runtime checks ensure the model is valid.
_Model: TypeAlias = _CallableModel | _ModelWithPredict | Any  # Any for flexibility with various ML frameworks

# Link function type
_LinkFunction: TypeAlias = Callable[[npt.NDArray[Any]], npt.NDArray[Any]]


# Masker protocols - for data masking strategies
class _MaskerProtocol(Protocol):
    """Protocol for masker objects."""

    shape: tuple[int | None, int] | Callable[..., tuple[int | None, int]]
    clustering: npt.NDArray[Any] | Callable[..., Any] | None

    def __call__(self, mask: bool | npt.NDArray[Any], *args: Any) -> Any: ...


# Masker type - accepts Masker objects, data matrices, or special dicts
_MaskerLike: TypeAlias = (
    _MaskerProtocol
    | npt.NDArray[Any]
    | pd.DataFrame
    | scipy.sparse.spmatrix
    | dict[str, Any]  # For mean/cov dicts
    | None
)

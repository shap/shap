# Type stubs for shap.utils
from typing import Any, Callable

import numpy as np

# Core utility functions
def sample(
    X: Any,
    nsamples: int = 100,
    random_state: int | None = None,
) -> Any: ...
def approximate_interactions(
    f: Callable[..., Any],
    shap_values: np.ndarray,
    X: Any,
) -> np.ndarray: ...
def assert_import(module_name: str) -> None: ...
def record_import_error(
    module_name: str,
    message: str,
    error: Exception,
) -> None: ...
def safe_isinstance(obj: Any, class_names: str | list[str]) -> bool: ...
def show_progress(
    iterable: Any,
    total: int | None = None,
    title: str = "",
    silent: bool = False,
) -> Any: ...
def make_masks(clustering: Any) -> np.ndarray: ...
def partition_tree_shuffle(
    indexes: np.ndarray,
    index_mask: np.ndarray,
    partition_tree: np.ndarray,
) -> None:
    """Randomly shuffle the indexes in a way that is consistent with the given partition tree.

    Parameters
    ----------
    indexes : np.ndarray
        The output location of the indexes we want shuffled. Note that len(indexes) should equal index_mask.sum().
        This array is modified in-place.
    index_mask : np.ndarray
        A bool mask of which indexes we want to include in the shuffled list.
    partition_tree : np.ndarray
        The partition tree we should follow.
    """
    ...

# Classes and objects
class MaskedModel:
    """A wrapper for models that handles masking."""

    def __init__(
        self,
        model: Any,
        masker: Any,
        link: Any,
        linearize_link: bool = True,
    ) -> None: ...
    def __call__(self, masks: np.ndarray, **kwargs: Any) -> Any: ...

class DenseData:
    """Legacy dense data wrapper."""

    def __init__(self, data: Any, **kwargs: Any) -> None: ...

    data: Any
    transposed: bool

class SparseData:
    """Legacy sparse data wrapper."""

    def __init__(self, data: Any, **kwargs: Any) -> None: ...

    data: Any
    transposed: bool

# Clustering utilities
def hclust(
    X: np.ndarray,
    metric: str = "correlation",
    **kwargs: Any,
) -> Any: ...
def delta_minimization_order(
    X: np.ndarray,
    **kwargs: Any,
) -> np.ndarray: ...

# Exception classes
class ExplainerError(Exception): ...
class DimensionError(Exception): ...
class InvalidMaskerError(Exception): ...
class InvalidModelError(Exception): ...
class InvalidAlgorithmError(Exception): ...
class InvalidFeaturePerturbationError(Exception): ...

# Warning classes
class ExperimentalWarning(UserWarning): ...

# Legacy function
def kmeans(X: Any, k: int, **kwargs: Any) -> Any: ...

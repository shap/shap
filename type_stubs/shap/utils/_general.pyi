# Type stubs for shap.utils._general
# Generated for SHAP library to provide code completion in VS Code

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, TypeVar

import numpy as np
import pandas as pd

from ._types import _ArrayT

T = TypeVar("T")

# Global variable for tracking import errors
import_errors: dict[str, tuple[str, Exception]]

def assert_import(package_name: str) -> None:
    """Assert that a package was imported successfully, raising stored error if not.

    Parameters
    ----------
    package_name : str
        Name of the package to check for import errors.

    Raises
    ------
    Exception
        The stored exception if the package had import errors.
    """
    ...

def record_import_error(package_name: str, msg: str, e: ImportError) -> None:
    """Record an import error for later retrieval.

    Parameters
    ----------
    package_name : str
        Name of the package that failed to import.
    msg : str
        Error message to display.
    e : ImportError
        The original import error exception.
    """
    ...

def shapley_coefficients(n: int) -> np.ndarray:
    """Compute the Shapley coefficients for n features.

    Parameters
    ----------
    n : int
        Number of features.

    Returns
    -------
    np.ndarray
        Array of Shapley coefficients.
    """
    ...

def convert_name(ind: str | int, shap_values: np.ndarray, input_names: list[str]) -> str | int:
    """Convert a feature name or index to a standardized format.

    Parameters
    ----------
    ind : str or int
        Feature name or index to convert.
    shap_values : np.ndarray
        SHAP values array for context.
    input_names : list
        List of input feature names.

    Returns
    -------
    str or int
        Converted feature identifier.
    """
    ...

def potential_interactions(shap_values_column: Any, shap_values_matrix: Any) -> np.ndarray:
    """Order other features by how much interaction they seem to have with the feature at the given index.

    This just bins the SHAP values for a feature along that feature's value. For true Shapley interaction
    index values for SHAP see the interaction_contribs option implemented in XGBoost.

    Parameters
    ----------
    shap_values_column : Explanation
        SHAP values for a specific feature column.
    shap_values_matrix : Explanation
        SHAP values matrix for all features.

    Returns
    -------
    np.ndarray
        Array of feature indices ordered by interaction strength.
    """
    ...

def approximate_interactions(
    index: str | int, shap_values: np.ndarray, X: np.ndarray | pd.DataFrame, feature_names: list[str] | None = None
) -> np.ndarray:
    """Order other features by how much interaction they seem to have with the feature at the given index.

    This just bins the SHAP values for a feature along that feature's value. For true Shapley interaction
    index values for SHAP see the interaction_contribs option implemented in XGBoost.

    Parameters
    ----------
    index : str or int
        Index or name of the feature to analyze interactions for.
    shap_values : np.ndarray
        SHAP values array.
    X : np.ndarray or pd.DataFrame
        Input data.
    feature_names : list, optional
        List of feature names.

    Returns
    -------
    np.ndarray
        Array of feature indices ordered by interaction strength.
    """
    ...

def encode_array_if_needed(arr: np.ndarray, dtype: type = np.float64) -> np.ndarray:
    """Encode array values if needed to convert to numeric dtype.

    Parameters
    ----------
    arr : np.ndarray
        Array to potentially encode.
    dtype : type, optional
        Target dtype for encoding, by default np.float64.

    Returns
    -------
    np.ndarray
        Encoded array.
    """
    ...

def sample(X: _ArrayT, nsamples: int = 100, random_state: int = 0) -> _ArrayT:
    """Performs sampling without replacement of the input data ``X``.

    This is a simple wrapper over scikit-learn's ``shuffle`` function.
    It is used mainly to downsample ``X`` for use as a background
    dataset in SHAP :class:`.Explainer` and its subclasses.

    .. versionchanged :: 0.42
        The behaviour of ``sample`` was changed from sampling *with* replacement to sampling
        *without* replacement.
        Note that reproducibility might be broken when using this function pre- and post-0.42,
        even with the specification of ``random_state``.

    Parameters
    ----------
    X : array-like
        Data to sample from. Input data can be arrays, lists, dataframes
        or scipy sparse matrices with a consistent first dimension.

    nsamples : int
        Number of samples to generate from ``X``.

    random_state : int
        Determines random number generation for shuffling the data. Use this to
        ensure reproducibility across multiple function calls.
    """
    ...

def safe_isinstance(obj: Any, class_path_str: str | list[str]) -> bool:
    """Acts as a safe version of isinstance without having to explicitly
    import packages which may not exist in the users environment.

    Checks if obj is an instance of type specified by class_path_str.

    Parameters
    ----------
    obj: Any
        Some object you want to test against
    class_path_str: str or list
        A string or list of strings specifying full class paths
        Example: `sklearn.ensemble.RandomForestRegressor`

    Returns
    -------
    bool: True if isinstance is true and the package exists, False otherwise
    """
    ...

def format_value(s: Any, format_str: str) -> str:
    """Strips trailing zeros and uses a unicode minus sign.

    Parameters
    ----------
    s : Any
        Value to format.
    format_str : str
        Format string to use.

    Returns
    -------
    str
        Formatted string.
    """
    ...

def ordinal_str(n: int) -> str:
    """Converts a number to and ordinal string.

    Parameters
    ----------
    n : int
        Number to convert.

    Returns
    -------
    str
        Ordinal string (e.g., "1st", "2nd", "3rd", "4th").
    """
    ...

class OpChain:
    """A way to represent a set of dot chained operations on an object without actually running them.

    This class allows building chains of operations (method calls, attribute access, indexing)
    that can be applied to objects later. It's particularly useful in the SHAP library for
    building operation templates that can be applied to Explanation objects.

    Examples
    --------
    >>> chain = OpChain("shap.Explanation")
    >>> chain = chain.mean(axis=0).abs.sum()
    >>> # Later apply to an actual explanation object:
    >>> result = chain.apply(explanation_obj)
    """

    _ops: list[list[Any]]
    _root_name: str

    def __init__(self, root_name: str = "") -> None:
        """Initialize an OpChain with an optional root name.

        Parameters
        ----------
        root_name : str, optional
            Name to use as the root of the operation chain for display purposes.
        """
        ...

    def apply(self, obj: Any) -> Any:
        """Applies all our ops to the given object, usually an :class:`.Explanation` instance.

        Parameters
        ----------
        obj : Any
            Object to apply the operation chain to.

        Returns
        -------
        Any
            Result of applying all operations in the chain to the object.
        """
        ...

    def __call__(self, *args: Any, **kwargs: Any) -> OpChain:
        """Update the args for the previous operation.

        Parameters
        ----------
        *args : Any
            Positional arguments for the previous operation.
        **kwargs : Any
            Keyword arguments for the previous operation.

        Returns
        -------
        OpChain
            New OpChain with updated arguments for the last operation.
        """
        ...

    def __getitem__(self, item: Any) -> OpChain:
        """Add an indexing operation to the chain.

        Parameters
        ----------
        item : Any
            Index or slice to apply.

        Returns
        -------
        OpChain
            New OpChain with the indexing operation added.
        """
        ...

    def __getattr__(self, name: str) -> OpChain:
        """Add an attribute access operation to the chain.

        Parameters
        ----------
        name : str
            Name of the attribute to access.

        Returns
        -------
        OpChain
            New OpChain with the attribute access operation added.
        """
        ...

    def __repr__(self) -> str:
        """Return string representation of the operation chain.

        Returns
        -------
        str
            String showing the complete operation chain.
        """
        ...

@contextmanager
def suppress_stderr() -> Generator[None, None, None]:
    """Context manager to temporarily suppress stderr output.

    Yields
    ------
    None
        Context where stderr is suppressed.
    """
    ...

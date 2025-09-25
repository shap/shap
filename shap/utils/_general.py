from __future__ import annotations

import copy
import os
import re
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import scipy.special
import sklearn

if TYPE_CHECKING:
    from ._types import _ArrayT

import_errors: dict[str, tuple[str, Exception]] = {}


def assert_import(package_name: str) -> None:
    global import_errors
    if package_name in import_errors:
        msg, e = import_errors[package_name]
        print(msg)
        raise e


def record_import_error(package_name: str, msg: str, e: ImportError) -> None:
    global import_errors
    import_errors[package_name] = (msg, e)


def shapley_coefficients(n: int) -> np.ndarray:
    out = np.zeros(n)
    for i in range(n):
        out[i] = 1 / (n * scipy.special.comb(n - 1, i))
    return out


def convert_name(ind, shap_values, input_names):
    if isinstance(ind, str):
        nzinds = np.where(np.array(input_names) == ind)[0]
        if len(nzinds) == 0:
            # we allow rank based indexing using the format "rank(int)"
            if ind.startswith("rank("):
                return np.argsort(-np.abs(shap_values).mean(0))[int(ind[5:-1])]

            # we allow the sum of all the SHAP values to be specified with "sum()"
            # assuming here that the calling method can deal with this case
            elif ind == "sum()":
                return "sum()"
            else:
                raise ValueError("Could not find feature named: " + ind)
        else:
            return nzinds[0]
    else:
        return ind


def potential_interactions(shap_values_column, shap_values_matrix):
    """Order other features by how much interaction they seem to have with the feature at the given index.

    This just bins the SHAP values for a feature along that feature's value. For true Shapley interaction
    index values for SHAP see the interaction_contribs option implemented in XGBoost.
    """
    # ignore inds that are identical to the column
    ignore_inds = np.where((shap_values_matrix.values.T - shap_values_column.values).T.std(0) < 1e-8)

    X = shap_values_matrix.data

    if X.shape[0] > 10000:
        a = np.arange(X.shape[0])
        np.random.shuffle(a)
        inds = a[:10000]
    else:
        inds = np.arange(X.shape[0])

    x = shap_values_column.data[inds]
    srt = np.argsort(x)
    shap_ref = shap_values_column.values[inds]
    shap_ref = shap_ref[srt]
    inc = max(min(int(len(x) / 10.0), 50), 1)
    interactions = []
    for i in range(X.shape[1]):
        encoded_val_other = encode_array_if_needed(X[inds, i][srt], dtype=float)

        val_other = encoded_val_other
        v = 0.0
        if not (i in ignore_inds or np.sum(np.abs(val_other)) < 1e-8):
            for j in range(0, len(x), inc):
                if np.std(val_other[j : j + inc]) > 0 and np.std(shap_ref[j : j + inc]) > 0:
                    v += abs(np.corrcoef(shap_ref[j : j + inc], val_other[j : j + inc])[0, 1])
        val_v = v

        val_other = np.isnan(encoded_val_other)
        v = 0.0
        if not (i in ignore_inds or np.sum(np.abs(val_other)) < 1e-8):
            for j in range(0, len(x), inc):
                if np.std(val_other[j : j + inc]) > 0 and np.std(shap_ref[j : j + inc]) > 0:
                    v += abs(np.corrcoef(shap_ref[j : j + inc], val_other[j : j + inc])[0, 1])
        nan_v = v

        interactions.append(max(val_v, nan_v))

    return np.argsort(-np.abs(interactions))


def approximate_interactions(index, shap_values, X, feature_names=None):
    """Order other features by how much interaction they seem to have with the feature at the given index.

    This just bins the SHAP values for a feature along that feature's value. For true Shapley interaction
    index values for SHAP see the interaction_contribs option implemented in XGBoost.
    """
    # convert from DataFrames if we got any
    if isinstance(X, pd.DataFrame):
        if feature_names is None:
            feature_names = X.columns
        X = X.values

    index = convert_name(index, shap_values, feature_names)

    if X.shape[0] > 10000:
        a = np.arange(X.shape[0])
        np.random.shuffle(a)
        inds = a[:10000]
    else:
        inds = np.arange(X.shape[0])

    x = X[inds, index]
    srt = np.argsort(x)
    shap_ref = shap_values[inds, index]
    shap_ref = shap_ref[srt]
    inc = max(min(int(len(x) / 10.0), 50), 1)
    interactions = []
    for i in range(X.shape[1]):
        encoded_val_other = encode_array_if_needed(X[inds, i][srt], dtype=float)

        val_other = encoded_val_other
        v = 0.0
        if not (i == index or np.sum(np.abs(val_other)) < 1e-8):
            for j in range(0, len(x), inc):
                if np.std(val_other[j : j + inc]) > 0 and np.std(shap_ref[j : j + inc]) > 0:
                    v += abs(np.corrcoef(shap_ref[j : j + inc], val_other[j : j + inc])[0, 1])
        val_v = v

        val_other = np.isnan(encoded_val_other)
        v = 0.0
        if not (i == index or np.sum(np.abs(val_other)) < 1e-8):
            for j in range(0, len(x), inc):
                if np.std(val_other[j : j + inc]) > 0 and np.std(shap_ref[j : j + inc]) > 0:
                    v += abs(np.corrcoef(shap_ref[j : j + inc], val_other[j : j + inc])[0, 1])
        nan_v = v

        interactions.append(max(val_v, nan_v))

    return np.argsort(-np.abs(interactions))


def encode_array_if_needed(arr, dtype=np.float64):
    try:
        return arr.astype(dtype)
    except ValueError:
        unique_values = np.unique(arr)
        encoding_dict = {string: index for index, string in enumerate(unique_values)}
        encoded_array = np.array([encoding_dict[string] for string in arr], dtype=dtype)
        return encoded_array


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

    random_state :
        Determines random number generation for shuffling the data. Use this to
        ensure reproducibility across multiple function calls.

    """
    if hasattr(X, "shape"):
        over_count = nsamples >= X.shape[0]
    else:
        over_count = nsamples >= len(X)

    if over_count:
        return X
    return sklearn.utils.shuffle(X, n_samples=nsamples, random_state=random_state)


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
    if isinstance(class_path_str, str):
        class_path_strs = [class_path_str]
    elif isinstance(class_path_str, (list, tuple)):
        class_path_strs = class_path_str
    else:
        class_path_strs = [""]

    # try each module path in order
    for class_path_str in class_path_strs:
        if "." not in class_path_str:
            raise ValueError(
                "class_path_str must be a string or list of strings specifying a full \
                module path to a class. Eg, 'sklearn.ensemble.RandomForestRegressor'"
            )

        # Splits on last occurrence of "."
        module_name, class_name = class_path_str.rsplit(".", 1)

        # here we don't check further if the model is not imported, since we shouldn't have
        # an object of that types passed to us if the model the type is from has never been
        # imported. (and we don't want to import lots of new modules for no reason)
        if module_name not in sys.modules:
            continue

        module = sys.modules[module_name]

        # Get class
        _class = getattr(module, class_name, None)

        if _class is None:
            continue

        if isinstance(obj, _class):
            return True

    return False


def format_value(s, format_str):
    """Strips trailing zeros and uses a unicode minus sign."""
    if not issubclass(type(s), str):
        s = format_str % s
    s = re.sub(r"\.?0+$", "", s)
    if s[0] == "-":
        s = "\u2212" + s[1:]
    return s


# From: https://groups.google.com/forum/m/#!topic/openrefine/G7_PSdUeno0
def ordinal_str(n):
    """Converts a number to and ordinal string."""
    return str(n) + {1: "st", 2: "nd", 3: "rd"}.get(4 if 10 <= n % 100 < 20 else n % 10, "th")


class OpChain:
    """A way to represent a set of dot chained operations on an object without actually running them."""

    def __init__(self, root_name: str = "") -> None:
        self._ops: list[list[Any]] = []
        self._root_name = root_name

    def apply(self, obj):
        """Applies all our ops to the given object, usually an :class:`.Explanation` instance."""
        for o in self._ops:
            op, args, kwargs = o
            if args is not None:
                obj = getattr(obj, op)(*args, **kwargs)
            else:
                obj = getattr(obj, op)
        return obj

    def __call__(self, *args, **kwargs) -> OpChain:
        """Update the args for the previous operation."""
        new_self = OpChain(self._root_name)
        new_self._ops = copy.copy(self._ops)
        new_self._ops[-1][1] = args
        new_self._ops[-1][2] = kwargs
        return new_self

    def __getitem__(self, item):
        new_self = OpChain(self._root_name)
        new_self._ops = copy.copy(self._ops)
        new_self._ops.append(["__getitem__", [item], {}])
        return new_self

    def __getattr__(self, name: str) -> OpChain:
        # Don't chain special attributes
        if name.startswith("__") and name.endswith("__"):
            return None  # type: ignore
        new_self = OpChain(self._root_name)
        new_self._ops = copy.copy(self._ops)
        new_self._ops.append([name, None, None])
        return new_self

    def __repr__(self):
        out = self._root_name
        for op in self._ops:
            op_name, args, kwargs = op
            args = args or tuple()
            kwargs = kwargs or {}

            out += f".{op_name}"
            has_args = len(args) > 0
            has_kwargs = len(kwargs) > 0
            if has_args or has_kwargs:
                out += "(" + ", ".join([repr(v) for v in args] + [f"{k}={v!r}" for k, v in kwargs.items()]) + ")"
        return out


# https://thesmithfam.org/blog/2012/10/25/temporarily-suppress-console-output-in-python/
@contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

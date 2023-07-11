import copy
import os
import re
import sys
from contextlib import contextmanager

import numpy as np
import scipy.special
import sklearn

import_errors = {}

def assert_import(package_name):
    global import_errors
    if package_name in import_errors:
        msg,e = import_errors[package_name]
        print(msg)
        raise e

def record_import_error(package_name, msg, e):
    global import_errors
    import_errors[package_name] = (msg, e)


def shapley_coefficients(n):
    out = np.zeros(n)
    for i in range(n):
        out[i] = 1 / (n * scipy.special.comb(n-1,i))
    return out


def convert_name(ind, shap_values, input_names):
    if type(ind) == str:
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
    """ Order other features by how much interaction they seem to have with the feature at the given index.

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
                if np.std(val_other[j:j + inc]) > 0 and np.std(shap_ref[j:j + inc]) > 0:
                    v += abs(np.corrcoef(shap_ref[j:j + inc], val_other[j:j + inc])[0, 1])
        val_v = v

        val_other = np.isnan(encoded_val_other)
        v = 0.0
        if not (i in ignore_inds or np.sum(np.abs(val_other)) < 1e-8):
            for j in range(0, len(x), inc):
                if np.std(val_other[j:j + inc]) > 0 and np.std(shap_ref[j:j + inc]) > 0:
                    v += abs(np.corrcoef(shap_ref[j:j + inc], val_other[j:j + inc])[0, 1])
        nan_v = v

        interactions.append(max(val_v, nan_v))

    return np.argsort(-np.abs(interactions))


def approximate_interactions(index, shap_values, X, feature_names=None):
    """ Order other features by how much interaction they seem to have with the feature at the given index.

    This just bins the SHAP values for a feature along that feature's value. For true Shapley interaction
    index values for SHAP see the interaction_contribs option implemented in XGBoost.
    """

    # convert from DataFrames if we got any
    if str(type(X)).endswith("'pandas.core.frame.DataFrame'>"):
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
                if np.std(val_other[j:j + inc]) > 0 and np.std(shap_ref[j:j + inc]) > 0:
                    v += abs(np.corrcoef(shap_ref[j:j + inc], val_other[j:j + inc])[0, 1])
        val_v = v

        val_other = np.isnan(encoded_val_other)
        v = 0.0
        if not (i == index or np.sum(np.abs(val_other)) < 1e-8):
            for j in range(0, len(x), inc):
                if np.std(val_other[j:j + inc]) > 0 and np.std(shap_ref[j:j + inc]) > 0:
                    v += abs(np.corrcoef(shap_ref[j:j + inc], val_other[j:j + inc])[0, 1])
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

def sample(X, nsamples=100, random_state=0):
    if hasattr(X, "shape"):
        over_count = nsamples >= X.shape[0]
    else:
        over_count = nsamples >= len(X)

    if over_count:
        return X
    return sklearn.utils.shuffle(X, n_samples=nsamples, random_state=random_state)

def safe_isinstance(obj, class_path_str):
    """
    Acts as a safe version of isinstance without having to explicitly
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
    --------
    bool: True if isinstance is true and the package exists, False otherwise
    """
    if isinstance(class_path_str, str):
        class_path_strs = [class_path_str]
    elif isinstance(class_path_str, list) or isinstance(class_path_str, tuple):
        class_path_strs = class_path_str
    else:
        class_path_strs = ['']

    # try each module path in order
    for class_path_str in class_path_strs:
        if "." not in class_path_str:
            raise ValueError("class_path_str must be a string or list of strings specifying a full \
                module path to a class. Eg, 'sklearn.ensemble.RandomForestRegressor'")

        # Splits on last occurence of "."
        module_name, class_name = class_path_str.rsplit(".", 1)

        # here we don't check further if the model is not imported, since we shouldn't have
        # an object of that types passed to us if the model the type is from has never been
        # imported. (and we don't want to import lots of new modules for no reason)
        if module_name not in sys.modules:
            continue

        module = sys.modules[module_name]

        #Get class
        _class = getattr(module, class_name, None)

        if _class is None:
            continue

        if isinstance(obj, _class):
            return True

    return False


def format_value(s, format_str):
    """ Strips trailing zeros and uses a unicode minus sign.
    """

    if not issubclass(type(s), str):
        s = format_str % s
    s = re.sub(r'\.?0+$', '', s)
    if s[0] == "-":
        s = "\u2212" + s[1:]
    return s

# From: https://groups.google.com/forum/m/#!topic/openrefine/G7_PSdUeno0
def ordinal_str(n):
    """ Converts a number to and ordinal string.
    """
    return str(n) + {1: 'st', 2: 'nd', 3: 'rd'}.get(4 if 10 <= n % 100 < 20 else n % 10, "th")

class OpChain():
    """ A way to represent a set of dot chained operations on an object without actually running them.
    """

    def __init__(self, root_name=""):
        self._ops = []
        self._root_name = root_name

    def apply(self, obj):
        """ Applies all our ops to the given object.
        """
        for o in self._ops:
            op,args,kwargs = o
            if args is not None:
                obj = getattr(obj, op)(*args, **kwargs)
            else:
                obj = getattr(obj, op)
        return obj

    def __call__(self, *args, **kwargs):
        """ Update the args for the previous operation.
        """
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

    def __getattr__(self, name):
        # Don't chain special attributes
        if name.startswith("__") and name.endswith("__"):
            return None
        new_self = OpChain(self._root_name)
        new_self._ops = copy.copy(self._ops)
        new_self._ops.append([name, None, None])
        return new_self

    def __repr__(self):
        out = self._root_name
        for o in self._ops:
            op,args,kwargs = o
            out += "."
            out += op
            if (args is not None and len(args) > 0) or (kwargs is not None and len(kwargs) > 0):
                out += "("
                if args is not None and len(args) > 0:
                    out += ", ".join([str(v) for v in args])
                if kwargs is not None and len(kwargs) > 0:
                    out += ", " + ", ".join([str(k)+"="+str(kwargs[k]) for k in kwargs.keys()])
                out += ")"
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

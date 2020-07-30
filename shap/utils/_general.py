import re
import pandas as pd
import numpy as np
import scipy as sp
from scipy.spatial.distance import pdist
import sys
import warnings
import sklearn
import importlib



if (sys.version_info < (3, 0)):
    warnings.warn("As of version 0.29.0 shap only supports Python 3 (not 2)!")

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
        out[i] = 1 / (n * sp.special.comb(n-1,i))
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
        encoded_val_other = encode_array_if_needed(X[inds, i][srt], dtype=np.float)

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
    if nsamples >= X.shape[0]:
        return X
    else:
        return sklearn.utils.resample(X, n_samples=nsamples, random_state=random_state)

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
        
        return isinstance(obj, _class)

    return False


def format_value(s, format_str):
    """ Strips trailing zeros and uses a unicode minus sign.
    """

    if type(s) is not str:
        s = format_str % s
    s = re.sub(r'\.?0+$', '', s)
    if s[0] == "-":
        s = u"\u2212" + s[1:]
    return s


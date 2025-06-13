# Type stubs for shap.utils._types
# Generated for SHAP library to provide code completion in VS Code

from __future__ import annotations

from typing import TypeVar, Union

import numpy as np
import pandas as pd
import scipy.sparse

# Array-like type union for data that can be used in SHAP operations
# Includes numpy arrays, pandas DataFrames/Series, lists, and scipy sparse matrices
_ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series, list, scipy.sparse.spmatrix]

# Type variable for array-like objects, constrained to specific array types
# Used for functions that preserve the input type (e.g., sampling functions)
_ArrayT = TypeVar("_ArrayT", np.ndarray, pd.DataFrame, pd.Series, scipy.sparse.spmatrix, list)

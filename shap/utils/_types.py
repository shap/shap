from __future__ import annotations

from typing import TypeVar, Union

import numpy as np
import pandas as pd
import scipy.sparse

# TODO: use TypeAlias (when we drop python 3.9)
_ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series, list, scipy.sparse.spmatrix]
_ArrayT = TypeVar("_ArrayT", np.ndarray, pd.DataFrame, pd.Series, scipy.sparse.spmatrix, list)

from __future__ import annotations

from typing import TypeAlias, TypeVar

import numpy as np
import pandas as pd
import scipy.sparse

_ArrayLike: TypeAlias = np.ndarray | pd.DataFrame | pd.Series | list | scipy.sparse.spmatrix
_ArrayT = TypeVar("_ArrayT", np.ndarray, pd.DataFrame, pd.Series, scipy.sparse.spmatrix, list)

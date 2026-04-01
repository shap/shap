from __future__ import annotations

from typing import TypeVar

import numpy as np
import pandas as pd
import scipy.sparse

type _ArrayLike = np.ndarray | pd.DataFrame | pd.Series | list | scipy.sparse.spmatrix
_ArrayT = TypeVar("_ArrayT", np.ndarray, pd.DataFrame, pd.Series, scipy.sparse.spmatrix, list)

from typing import TypeAlias, TypeVar, Union

import numpy as np
import pandas as pd
import scipy.sparse

_ArrayLike: TypeAlias = Union[np.ndarray, pd.DataFrame, pd.Series, list, scipy.spare.csr_matrix]
_ArrayT = TypeVar("_ArrayT", np.ndarray, pd.DataFrame, pd.Series, scipy.sparse.csr_matrix, list)

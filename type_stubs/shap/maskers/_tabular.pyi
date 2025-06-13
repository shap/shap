# Type stubs for shap.maskers._tabular
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ._masker import Masker

class Independent(Masker):
    """A masker that treats features as independent."""

    def __init__(
        self,
        data: np.ndarray | pd.DataFrame | dict[str, Any],
        max_samples: int = 100,
        **kwargs: Any,
    ) -> None: ...
    def __call__(self, mask: np.ndarray, *args: Any) -> Any: ...

    # Instance attributes
    data: np.ndarray
    shape: tuple[int, ...]
    feature_names: list[str] | None

class Partition(Masker):
    """A masker that groups features into partitions."""

    def __init__(
        self,
        data: np.ndarray | pd.DataFrame,
        clustering: Any = None,
        max_samples: int = 100,
        **kwargs: Any,
    ) -> None: ...
    def __call__(self, mask: np.ndarray, *args: Any) -> Any: ...

    # Instance attributes
    data: np.ndarray
    shape: tuple[int, ...]
    clustering: Any
    feature_names: list[str] | None

# Legacy aliases
Impute = Independent
Tabular = Independent

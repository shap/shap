"""Tests for shap.benchmark._sequential."""

import numpy as np
import pandas as pd
import pytest

from shap.benchmark._sequential import SequentialMasker
from shap.maskers import Independent


def _model(x):
    return x.sum(axis=-1)


def test_sequential_masker_rejects_dataframe_with_clear_message():
    rs = np.random.RandomState(0)
    X = rs.random((4, 3))
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    masker = Independent(X)

    with pytest.raises(TypeError, match=r"don't iterate correctly"):
        SequentialMasker("keep", "positive", masker, _model, df)

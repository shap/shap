"""Tests for shap.benchmark._sequential."""

import inspect

import numpy as np
import pandas as pd
import pytest

from shap.benchmark._sequential import SequentialMasker, SequentialPerturbation
from shap.maskers import Independent


def _model(x):
    return x.sum(axis=-1)


def test_sequential_perturbation_indices_default_is_not_mutable():
    # Regression test for https://github.com/shap/shap/issues/4723.
    # A mutable default argument would be shared across calls and could cause
    # cross-call state bleed if the function body is ever changed to mutate it.
    default = inspect.signature(SequentialPerturbation.__call__).parameters["indices"].default
    assert default is None


def test_sequential_masker_rejects_dataframe_with_clear_message():
    rs = np.random.RandomState(0)
    X = rs.random((4, 3))
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    masker = Independent(X)

    with pytest.raises(TypeError, match=r"don't iterate correctly"):
        SequentialMasker("keep", "positive", masker, _model, df)

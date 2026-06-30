"""Tests for the Random explainer."""

from __future__ import annotations

import numpy as np
import pytest

from shap.explainers.other._random import Random
from shap.models import Model


class _DummyMaskedModel:
    def __init__(self, model, masker, link, linearize_link, *row_args):
        self.mask_shapes = [(len(row_args[0]),)] if row_args else [(0,)]

    def __len__(self):
        return 3

    def __call__(self, masks, zero_index=None, batch_size=None):
        del masks, zero_index, batch_size
        return np.array([[1.5, -0.5]])


class _MaskerWithArrayClustering:
    def __init__(self):
        self.clustering = np.array([[0, 1, 0.1, 2]])


class _MaskerWithCallableClustering:
    def __init__(self):
        self.clustering = lambda row: np.array([[0, 1, float(np.sum(row)), len(row)]])


class _MaskerWithUnsupportedClustering:
    def __init__(self):
        self.clustering = "unsupported"


def test_random_init_wraps_non_model_and_applies_call_args_defaults():
    def model_fn(x):
        return x

    masker = _MaskerWithArrayClustering()
    explainer = Random(model_fn, masker, constant=True, max_evals=123)

    assert isinstance(explainer.model, Model)
    assert explainer.constant is True
    assert explainer.constant_attributions is None
    assert explainer.__call__.__kwdefaults__["max_evals"] == 123


def test_random_explain_row_with_ndarray_clustering_and_output_names(monkeypatch):
    import shap.explainers.other._random as random_module

    monkeypatch.setattr(random_module, "MaskedModel", _DummyMaskedModel)
    monkeypatch.setattr(random_module.np.random, "randn", lambda *shape: np.ones(shape))

    model = Model(lambda x: x)
    model.output_names = ["o1", "o2"]
    explainer = Random(model, _MaskerWithArrayClustering())

    row = np.array([1.0, 2.0, 3.0])
    result = explainer.explain_row(
        row,
        max_evals=10,
        main_effects=False,
        error_bounds=False,
        batch_size=1,
        outputs=None,
        silent=True,
    )

    assert result["output_names"] == ["o1", "o2"]
    np.testing.assert_allclose(result["expected_values"], np.array([1.5, -0.5]))
    np.testing.assert_allclose(result["clustering"], np.array([[0, 1, 0.1, 2]]))
    assert result["main_effects"] is None
    assert result["error_std"] is None
    assert result["mask_shapes"] == [(3,)]
    assert result["values"].shape == (3, 2)
    np.testing.assert_allclose(result["values"], np.ones((3, 2)) * 0.001)


def test_random_explain_row_with_callable_clustering(monkeypatch):
    import shap.explainers.other._random as random_module

    monkeypatch.setattr(random_module, "MaskedModel", _DummyMaskedModel)
    explainer = Random(lambda x: x, _MaskerWithCallableClustering())

    row = np.array([1.0, 2.0, 3.0])
    result = explainer.explain_row(
        row,
        max_evals=10,
        main_effects=False,
        error_bounds=False,
        batch_size=1,
        outputs=None,
        silent=True,
    )

    np.testing.assert_allclose(result["clustering"], np.array([[0.0, 1.0, 6.0, 3.0]]))


def test_random_explain_row_raises_for_unsupported_clustering(monkeypatch):
    import shap.explainers.other._random as random_module

    monkeypatch.setattr(random_module, "MaskedModel", _DummyMaskedModel)
    explainer = Random(lambda x: x, _MaskerWithUnsupportedClustering())

    with pytest.raises(NotImplementedError, match="not yet supported by the Permutation explainer"):
        explainer.explain_row(
            np.array([1.0, 2.0, 3.0]),
            max_evals=10,
            main_effects=False,
            error_bounds=False,
            batch_size=1,
            outputs=None,
            silent=True,
        )

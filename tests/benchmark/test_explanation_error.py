import numpy as np
import pytest

from shap.benchmark._explanation_error import ExplanationError


class _DummyMasker:
    clustering = None


def test_explanation_error_honors_indices(monkeypatch):
    """Test that indices parameter correctly selects rows."""
    init_rows = []

    class DummyMaskedModel:
        def __init__(self, _model, _masker, _link, _linearize_link, *args):
            init_rows.append(args[0].copy())

        def __call__(self, masks):
            return np.zeros(len(masks), dtype=float)

    monkeypatch.setattr("shap.benchmark._explanation_error.MaskedModel", DummyMaskedModel)

    model_args = np.array([[1.0, 2.0], [10.0, 20.0]])
    attributions = np.zeros_like(model_args)

    metric = ExplanationError(_DummyMasker(), lambda x: x.sum(-1), model_args, num_permutations=1)
    metric(attributions, "dummy", indices=[1], silent=True)

    assert len(init_rows) == 1, "Should process exactly one row"
    np.testing.assert_allclose(init_rows[0], model_args[1])


def test_explanation_error_default_indices_processes_all_rows(monkeypatch):
    """Test that omitting indices processes all rows."""
    init_rows = []

    class DummyMaskedModel:
        def __init__(self, _model, _masker, _link, _linearize_link, *args):
            init_rows.append(args[0].copy())

        def __call__(self, masks):
            return np.zeros(len(masks), dtype=float)

    monkeypatch.setattr("shap.benchmark._explanation_error.MaskedModel", DummyMaskedModel)

    model_args = np.array([[1.0, 2.0], [10.0, 20.0], [100.0, 200.0]])
    attributions = np.zeros_like(model_args)

    metric = ExplanationError(_DummyMasker(), lambda x: x.sum(-1), model_args, num_permutations=1)
    metric(attributions, "dummy", silent=True)  # No indices arg

    assert len(init_rows) == 3, "Should process all three rows when indices=None"


def test_explanation_error_rejects_empty_indices():
    """Test that empty indices list raises ValueError."""
    model_args = np.array([[1.0, 2.0], [10.0, 20.0]])
    attributions = np.zeros_like(model_args)
    metric = ExplanationError(_DummyMasker(), lambda x: x.sum(-1), model_args, num_permutations=1)

    with pytest.raises(ValueError, match="at least one row index"):
        metric(attributions, "dummy", indices=[], silent=True)


def test_explanation_error_rejects_out_of_range_indices():
    """Test that out-of-range indices raise IndexError."""
    model_args = np.array([[1.0, 2.0], [10.0, 20.0]])
    attributions = np.zeros_like(model_args)
    metric = ExplanationError(_DummyMasker(), lambda x: x.sum(-1), model_args, num_permutations=1)

    with pytest.raises(IndexError, match="out-of-range"):
        metric(attributions, "dummy", indices=[0, 5], silent=True)  # 5 is out of range

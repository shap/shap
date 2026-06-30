"""This file contains tests for the Model class."""

from __future__ import annotations

import io
import pickle

import numpy as np

import shap
from shap.models import _model as model_module


def _add_two(x):
    return np.asarray(x) + 2


class _CallableWithOutputNames:
    output_names = ["first", "second"]

    def __call__(self, x):
        return np.asarray(x)


class _FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.cpu_called = False
        self.detach_called = False
        self.numpy_called = False

    def cpu(self):
        self.cpu_called = True
        return self

    def detach(self):
        self.detach_called = True
        return self

    def numpy(self):
        self.numpy_called = True
        return self.values


def test_model_init_unwraps_nested_model_and_preserves_output_names():
    """Test both constructor branches: nested Model unwrapping and output_names copying."""
    base_model = shap.models.Model(_add_two)
    wrapped_model = shap.models.Model(base_model)
    assert wrapped_model.inner_model is base_model.inner_model

    named_callable = _CallableWithOutputNames()
    model_with_names = shap.models.Model(named_callable)
    assert model_with_names.output_names == ["first", "second"]


def test_model_call_converts_non_tensor_outputs_to_numpy():
    """Test that standard callable outputs are converted to numpy arrays."""

    def f(x):
        return [x, x + 1]

    model = shap.models.Model(f)
    result = model(3)
    np.testing.assert_array_equal(result, np.array([3, 4]))


def test_model_call_handles_tensor_like_outputs(monkeypatch):
    """Test tensor path conversion via cpu().detach().numpy()."""
    fake_tensor = _FakeTensor([1, 2, 3])

    monkeypatch.setattr(model_module, "safe_isinstance", lambda *_args, **_kwargs: True)

    model = shap.models.Model(lambda _x: fake_tensor)
    result = model("ignored")

    np.testing.assert_array_equal(result, np.array([1, 2, 3]))
    assert fake_tensor.cpu_called
    assert fake_tensor.detach_called
    assert fake_tensor.numpy_called


def test_model_save_and_load_round_trip():
    """Test saving and loading a Model instance with instantiation enabled."""
    model = shap.models.Model(_add_two)

    stream = io.BytesIO()
    model.save(stream)
    stream.seek(0)

    loaded = shap.models.Model.load(stream)

    assert isinstance(loaded, shap.models.Model)
    np.testing.assert_array_equal(loaded(np.array([1, 2])), np.array([3, 4]))


def test_model_load_without_instantiation_returns_constructor_kwargs():
    """Test loading constructor kwargs directly (instantiate=False)."""
    model = shap.models.Model(_add_two)

    stream = io.BytesIO()
    model.save(stream)
    stream.seek(0)

    # The instantiate=False branch expects the outer Serializable type token
    # to already be consumed by the caller.
    loaded_type = pickle.load(stream)
    assert loaded_type is shap.models.Model

    kwargs = shap.models.Model.load(stream, instantiate=False)

    assert set(kwargs) == {"model"}
    assert callable(kwargs["model"])
    np.testing.assert_array_equal(kwargs["model"](np.array([1, 2])), np.array([3, 4]))

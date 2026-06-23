import numpy as np
import pandas as pd
import pytest

from shap.explainers.other import _lime


class _DummyExplanation:
    def __init__(self, local_exp):
        self.local_exp = local_exp


class _DummyLimeTabularExplainer:
    def __init__(self, data, mode):
        self.data = data
        self.mode = mode

    def explain_instance(self, row, model, labels, num_features):
        local_exp = {}
        for label in labels:
            local_exp[label] = [(k, float(label + k + 1)) for k in range(num_features)]
        return _DummyExplanation(local_exp)


def _install_dummy_lime(monkeypatch):
    dummy_lime = type("_DummyLime", (), {})()
    dummy_tabular = type("_DummyTabular", (), {"LimeTabularExplainer": _DummyLimeTabularExplainer})
    dummy_lime.lime_tabular = dummy_tabular
    monkeypatch.setattr(_lime, "lime", dummy_lime, raising=False)


def test_lime_tabular_invalid_mode_raises_value_error():
    with pytest.raises(ValueError, match="Invalid mode"):
        _lime.LimeTabular(lambda x: x, np.ones((2, 2)), mode="invalid")


def test_lime_tabular_classification_dataframe_init_and_attributions(monkeypatch):
    _install_dummy_lime(monkeypatch)

    def model(X):
        return X[:, 0] * 0.25

    background_df = pd.DataFrame([[1.0, 2.0], [2.0, 3.0]], columns=["a", "b"])
    explainer = _lime.LimeTabular(model, background_df, mode="classification")

    assert explainer.out_dim == 1
    assert explainer.flat_out is True
    assert isinstance(explainer.explainer.data, np.ndarray)

    wrapped_model = explainer.model
    explainer.model = model
    preds = wrapped_model(np.array([[2.0, 4.0], [4.0, 8.0]]))
    assert preds.shape == (2, 2)
    assert np.allclose(preds[:, 0] + preds[:, 1], 1.0)

    attrs = explainer.attributions(pd.DataFrame([[10.0, 20.0]], columns=["a", "b"]), num_features=2)
    assert attrs.shape == (1, 2)
    assert np.allclose(attrs[0], [1.0, 2.0])


def test_lime_tabular_regression_multiclass_output_negates_attributions(monkeypatch):
    _install_dummy_lime(monkeypatch)

    def model(X):
        return np.column_stack([X[:, 0], X[:, 1]])

    explainer = _lime.LimeTabular(model, np.array([[1.0, 2.0], [3.0, 4.0]]), mode="regression")

    assert explainer.out_dim == 2
    assert explainer.flat_out is False

    attrs = explainer.attributions(np.array([[5.0, 6.0]]), num_features=2)
    assert isinstance(attrs, list)
    assert len(attrs) == 2
    assert attrs[0].shape == (1, 2)
    assert attrs[1].shape == (1, 2)
    assert np.allclose(attrs[0][0], [-1.0, -2.0])
    assert np.allclose(attrs[1][0], [-2.0, -3.0])

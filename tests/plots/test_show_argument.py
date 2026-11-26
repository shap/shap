import warnings

import matplotlib.pyplot as plt
import pytest

import shap


def test_show_argument_deprecated_bar(explainer):
    shap_values = explainer(explainer.data)
    with pytest.deprecated_call(match=r"`show` argument to `shap\.plots\.bar`"):
        shap.plots.bar(shap_values, show=False)


def test_show_argument_deprecated_scatter(explainer):
    explanation = explainer(explainer.data)
    with pytest.deprecated_call(match=r"`show` argument to `shap\.plots\.scatter`"):
        shap.plots.scatter(explanation[:, "Age"], show=False)


def test_show_argument_deprecated_waterfall(explainer):
    explanation = explainer(explainer.data)
    with pytest.deprecated_call(match=r"`show` argument to `shap\.plots\.waterfall`"):
        shap.plots.waterfall(explanation[0], show=False)


def test_waterfall_no_implicit_show_by_default(explainer, monkeypatch):
    called = []

    def _fake_show(*args, **kwargs):
        called.append((args, kwargs))

    monkeypatch.setattr(plt, "show", _fake_show)

    explanation = explainer(explainer.data)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        shap.plots.waterfall(explanation[0])

    assert called == []
    show_warnings = [w for w in captured if "`show` argument to `shap.plots.waterfall` is deprecated" in str(w.message)]
    assert show_warnings == []

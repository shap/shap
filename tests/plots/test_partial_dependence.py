import numpy as np
import pytest
import matplotlib.pyplot as plt
import shap

def test_partial_dependence_basic():
    """Make sure a dependence plot does not crash."""
    X = np.random.randn(50, 3)
    model = lambda x: np.mean(x, axis=1)
    shap.partial_dependence_plot(0, model, X, show=False)


def test_partial_dependence_show_false(monkeypatch):
    """Make shure that show=False doesn't call plt.show()."""
    X = np.random.randn(20, 2)
    model = lambda x: np.mean(x, axis=1)

    shown = False
    def fake_show():
        nonlocal shown
        shown = True

    monkeypatch.setattr(plt, "show", fake_show)
    shap.partial_dependence_plot(0, model, X, show=False)
    assert not shown


def test_partial_dependence_subplots():
    """Test multiple subplots with different axes"""
    X = np.random.randn(30, 3)
    model = lambda x: np.mean(x, axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, ax in enumerate(axes):
        shap.partial_dependence_plot(i, model, X, ax=ax, show=False)

    assert len(plt.get_fignums()) == 1
    plt.close(fig)
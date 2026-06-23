import matplotlib.pyplot as plt
import pytest

import shap.plots._force_matplotlib as force_mpl


def _sample_plot_data(link):
    return {
        "features": {
            0: {"effect": -0.2, "value": "v0"},
            1: {"effect": 0.3, "value": "v1"},
        },
        "featureNames": ["f0", "f1"],
        "outNames": ["out"],
        "outValue": 0.15,
        "baseValue": 0.05,
        "link": link,
    }


def test_format_data_invalid_link_raises_value_error():
    with pytest.raises(ValueError, match="Unrecognized link function"):
        force_mpl.format_data(_sample_plot_data("unknown"))


def test_draw_additive_plot_logit_scale_and_show_branch(monkeypatch):
    xscale_calls = []
    show_calls = []

    monkeypatch.setattr(force_mpl, "draw_bars", lambda *args, **kwargs: ([], []))
    monkeypatch.setattr(force_mpl, "draw_labels", lambda fig, ax, *args, **kwargs: (fig, ax))
    monkeypatch.setattr(force_mpl, "draw_higher_lower_element", lambda *args, **kwargs: None)
    monkeypatch.setattr(force_mpl, "draw_base_element", lambda *args, **kwargs: None)
    monkeypatch.setattr(force_mpl, "draw_output_element", lambda *args, **kwargs: None)
    monkeypatch.setattr(force_mpl, "update_axis_limits", lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, "xscale", lambda scale: xscale_calls.append(scale))
    monkeypatch.setattr(plt, "show", lambda: show_calls.append(True))

    force_mpl.draw_additive_plot(_sample_plot_data("logit"), figsize=(4, 2), show=True)

    assert xscale_calls == ["logit"]
    assert show_calls == [True]

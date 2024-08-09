from dataclasses import asdict

import numpy as np

from shap.plots import _style


def test_default_style():
    new_style = _style._load_default_style()
    assert configs_are_equal(_style.STYLE, new_style)

    new_style.default_negative_color = "black"
    assert not configs_are_equal(_style.STYLE, new_style)


def test_style_context_manager():
    custom_style = _style.StyleConfig(text="green")
    assert _style.STYLE.text == "white"
    with _style.use_style(custom_style):
        assert _style.STYLE.text == "green"
    assert _style.STYLE.text == "white"


# Helper functions to compare equality of config dataclasses


def configs_are_equal(config1: _style.StyleConfig, config2: _style.StyleConfig):
    d1 = asdict(config1)
    d2 = asdict(config2)
    return all(_values_are_equivalent(d1[key], d2[key]) for key in d1.keys())


def _values_are_equivalent(value1, value2):
    value1_is_arraylike = isinstance(value1, (np.ndarray, list, tuple))
    value2_is_arraylike = isinstance(value2, (np.ndarray, list, tuple))
    if value1_is_arraylike and value2_is_arraylike:
        return np.allclose(value1, value2)
    elif value1_is_arraylike != value2_is_arraylike:
        return False
    return value1 == value2

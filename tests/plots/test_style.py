from dataclasses import asdict

import numpy as np
import pytest

from shap.plots import _style
from shap.utils._exceptions import InvalidOptionError

# TODO: when the API is finalised, these functions will probably be
# exposed in shap.plots, not shap.plots._style


def test_default_style():
    new_style = _style.load_default_style()
    assert configs_are_equal(_style._STYLE, new_style)

    new_style.secondary_color_negative = "black"
    assert not configs_are_equal(_style._STYLE, new_style)


def test_style_context():
    custom_style = _style.StyleConfig(text_color="green")
    assert _style._STYLE.text_color == "white"
    with _style.style_context(custom_style):
        assert _style._STYLE.text_color == "green"
    assert _style._STYLE.text_color == "white"


def test_style_overrides():
    assert _style._STYLE.text_color == "white"
    with _style.style_overrides(text_color="green"):
        assert _style._STYLE.text_color == "green"
    assert _style._STYLE.text_color == "white"


def test_style_overrides_raises_on_invalid_options():
    with pytest.raises(InvalidOptionError, match="Invalid style options"):
        with _style.style_overrides(foo="bar"):
            pass


# Helper functions to compare equality of config dataclasses


def configs_are_equal(config1: _style.StyleConfig, config2: _style.StyleConfig):
    d1 = asdict(config1)
    d2 = asdict(config2)
    assert d1.keys() == d2.keys()
    return all(_values_are_equivalent(d1[key], d2[key]) for key in d1.keys())


def _values_are_equivalent(value1, value2):
    value1_is_arraylike = isinstance(value1, (np.ndarray, list, tuple))
    value2_is_arraylike = isinstance(value2, (np.ndarray, list, tuple))
    if value1_is_arraylike and value2_is_arraylike:
        return np.allclose(value1, value2)
    elif value1_is_arraylike != value2_is_arraylike:
        return False
    return value1 == value2

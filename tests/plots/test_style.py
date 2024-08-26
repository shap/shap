from dataclasses import asdict
from typing import get_type_hints

import numpy as np
import pytest

from shap.plots import _style
from shap.plots._style import get_style
from shap.utils._exceptions import InvalidStyleOptionError

# TODO: when the API is finalised, these functions will probably be
# exposed in shap.plots, not shap.plots._style


def test_default_style():
    default_stype = _style.load_default_style()
    assert configs_are_equal(get_style(), default_stype)


def test_set_style():
    prev_style = get_style()
    _style.set_style(text_color="green")
    assert get_style().text_color == "green"
    assert not configs_are_equal(get_style(), prev_style)


def test_style_context():
    original_text_color = get_style().text_color
    with _style.style_context(text_color="green"):
        assert get_style().text_color == "green"
    assert get_style().text_color == original_text_color


def test_set_style_raises_on_invalid_options():
    with pytest.raises(InvalidStyleOptionError, match="Invalid style config option"):
        _style.set_style(foo="bar")  # type: ignore


def test_style_context_raises_on_invalid_options():
    with pytest.raises(InvalidStyleOptionError, match="Invalid style config option"):
        with _style.style_context(foo="bar"):  # type: ignore
            pass


def test_consistent_style_config_and_style_options():
    # The StyleConfig dataclass should have the same keys and Types as the StyleOptions TypedDict
    style_config_types = get_type_hints(_style.StyleConfig)
    style_options_types = get_type_hints(_style.StyleOptions)
    assert style_config_types == style_options_types


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

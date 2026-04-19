from dataclasses import asdict
from typing import get_type_hints

import numpy as np
import pytest
from matplotlib import pyplot as plt

import shap
from shap.plots import _style
from shap.plots._style import get_style
from shap.utils._exceptions import InvalidStyleOptionError

# TODO: when the API is finalised, these functions will probably be
# exposed in shap.plots, not shap.plots._style


def test_default_style():
    default_stype = _style.load_default_style()
    assert configs_are_equal(get_style(), default_stype)


def test_style_helpers_are_available_from_plots_namespace():
    assert shap.plots.get_style is _style.get_style
    assert shap.plots.set_style is _style.set_style
    assert shap.plots.style_context is _style.style_context


def test_set_style():
    prev_style = get_style()
    _style.set_style(text_color="green")
    assert get_style().text_color == "green"
    assert not configs_are_equal(get_style(), prev_style)


def test_set_style_shap_alias():
    _style.set_style(text_color="green")
    _style.set_style("shap")
    assert configs_are_equal(get_style(), _style.load_default_style())


def test_set_style_matplotlib_alias():
    with plt.rc_context({"font.size": 17, "axes.labelsize": 19, "xtick.labelsize": 7, "grid.linewidth": 2.5}):
        _style.set_style("matplotlib", text_color="green")

    current_style = get_style()
    assert current_style.font_size == 17
    assert current_style.label_size == 19
    assert current_style.tick_label_size == 7
    assert current_style.line_width == 2.5
    assert current_style.text_color == "green"


def test_set_style_matplotlib_style_sheet():
    _style.set_style("ggplot")
    assert configs_are_equal(get_style(), _style.load_matplotlib_style("ggplot"))


def test_style_context():
    original_text_color = get_style().text_color
    with _style.style_context(text_color="green"):
        assert get_style().text_color == "green"
    assert get_style().text_color == original_text_color


def test_style_context_with_matplotlib_alias():
    original_style = get_style()
    with plt.rc_context({"font.size": 17}):
        with _style.style_context("matplotlib"):
            assert get_style().font_size == 17
    assert configs_are_equal(get_style(), original_style)


def test_set_style_raises_on_invalid_options():
    with pytest.raises(InvalidStyleOptionError, match="Invalid style config option"):
        _style.set_style(foo="bar")  # type: ignore


def test_set_style_raises_on_invalid_style_alias():
    with pytest.raises(InvalidStyleOptionError, match="Invalid style config option"):
        _style.set_style("foo")  # type: ignore


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

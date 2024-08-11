"""Configuration of customisable style options for SHAP plots.

NOTE: This is experimental and subject to change!
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Union

import numpy as np

from ..utils._exceptions import InvalidOptionError
from . import colors

# Type hints, adapted from matplotlib.typing
RGBColorType = Union[tuple[float, float, float], str]
RGBAColorType = Union[
    str,  # "none" or "#RRGGBBAA"/"#RGBA" hex strings
    tuple[float, float, float, float],
    # 2 tuple (color, alpha) representations, not infinitely recursive
    # RGBColorType includes the (str, float) tuple, even for RGBA strings
    tuple[RGBColorType, float],
    # (4-tuple, float) is odd, but accepted as the outer float overriding A of 4-tuple
    tuple[tuple[float, float, float, float], float],
]
ColorType = Union[RGBColorType, RGBAColorType, np.ndarray]


@dataclass
class StyleConfig:
    """Configuration of colors across all matplotlib-based shap plots."""

    # Waterfall plot config
    primary_color_positive: ColorType = field(default_factory=lambda: colors.red_rgb)
    primary_color_negative: ColorType = field(default_factory=lambda: colors.blue_rgb)
    secondary_color_positive: ColorType = field(default_factory=lambda: colors.light_red_rgb)
    secondary_color_negative: ColorType = field(default_factory=lambda: colors.light_blue_rgb)
    hlines_color: ColorType = "#cccccc"
    vlines_color: ColorType = "#bbbbbb"
    text_color: ColorType = "white"
    tick_labels_color: ColorType = "#999999"


def load_default_style() -> StyleConfig:
    """Load the default style configuration."""
    # In future, this could allow reading from a persistent config file, like matplotlib rcParams
    return StyleConfig()


# Singleton instance that determines the current style.
# CAREFUL! To ensure the correct object is picked up, do not import this directly,
# but intead access this at runtime with get_style().
_STYLE = load_default_style()


def get_style() -> StyleConfig:
    """Return the current style configuration."""
    return _STYLE


def set_style(style: StyleConfig):
    """Set the current style configuration."""
    global _STYLE
    _STYLE = style


@contextmanager
def style_context(style: StyleConfig):
    """Context manager to temporarily change the style.

    NOTE: This is experimental and subject to change!

    Examples
    --------
    To set a new style::

        with shap.plots.style_context(new_style):
            shap.plots.waterfall(...)
    """
    global _STYLE
    old_style = _STYLE
    _STYLE = style
    yield
    _STYLE = old_style


@contextmanager
def style_overrides(**kwargs):
    """Context manager to temporarily override a subset of parameters.

    NOTE: This is experimental and subject to change!

    Examples
    --------
    To temporarily override a style option::

        with shap.plots.style_overrides(text_color="black"):
            shap.plots.waterfall(...)
    """
    global _STYLE
    old_style = _STYLE

    try:
        _STYLE = replace(old_style, **kwargs)
    except TypeError as e:
        raise InvalidOptionError("Invalid style options") from e

    yield
    _STYLE = old_style

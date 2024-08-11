"""Configuration of customisable style options for SHAP plots.

NOTE: This is experimental and subject to change!
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import TypedDict, Union, Unpack

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


@dataclass(frozen=True)
class StyleConfig:
    """All configuration options for matplotlib-based shap plots."""

    primary_color_positive: ColorType
    primary_color_negative: ColorType
    secondary_color_positive: ColorType
    secondary_color_negative: ColorType
    hlines_color: ColorType
    vlines_color: ColorType
    text_color: ColorType
    tick_labels_color: ColorType


class StyleOptions(TypedDict, total=False):
    """A TypedDict of possible updates to a style configuration"""

    # Nb. There is some duplication here with the StyleConfig dataclass, but
    # it's necessary to provide type hints. StyleConfig represents a full set of
    # options at runtime, whilst StyleOptions represents a partial set of
    # updates to the options.
    primary_color_positive: ColorType
    primary_color_negative: ColorType
    secondary_color_positive: ColorType
    secondary_color_negative: ColorType
    hlines_color: ColorType
    vlines_color: ColorType
    text_color: ColorType
    tick_labels_color: ColorType


_style_defaults = StyleConfig(
    primary_color_positive=colors.red_rgb,
    primary_color_negative=colors.blue_rgb,
    secondary_color_positive=colors.light_red_rgb,
    secondary_color_negative=colors.light_blue_rgb,
    hlines_color="#cccccc",
    vlines_color="#bbbbbb",
    text_color="white",
    tick_labels_color="#999999",
)


def load_default_style() -> StyleConfig:
    """Load the default style configuration."""
    # In future, this could allow reading from a persistent config file, like matplotlib rcParams
    return _style_defaults


# Singleton instance that determines the current style.
# CAREFUL! To ensure the correct object is picked up, do not import this directly,
# but intead access this at runtime with get_style().
_STYLE = load_default_style()


def get_style() -> StyleConfig:
    """Return all currently active global style configuration options."""
    return _STYLE


def set_style(**options: Unpack[StyleOptions]) -> None:
    """Set options in the currently active global style configuration."""
    global _STYLE
    try:
        _STYLE = replace(_STYLE, **options)
    except TypeError as e:
        raise InvalidOptionError("Invalid style options") from e


@contextmanager
def style_context(**options: Unpack[StyleOptions]):
    """Context manager to temporarily change style options.

    NOTE: This is experimental and subject to change!

    Examples
    --------
    To set a new style::

        with shap.plots.style_context(text_color="black"):
            shap.plots.waterfall(...)
    """
    global _STYLE
    old_style = _STYLE
    try:
        _STYLE = replace(_STYLE, **options)
    except TypeError as e:
        raise InvalidOptionError("Invalid style options") from e
    yield
    _STYLE = old_style

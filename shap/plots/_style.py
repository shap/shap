"""Configuration of customisable style options for SHAP plots.

NOTE: This is experimental and subject to change!
"""

from __future__ import annotations

import dataclasses
from contextlib import contextmanager
from typing import TypedDict, Union

import numpy as np
from typing_extensions import Unpack

from ..utils._exceptions import InvalidStyleOptionError
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


# TODO: Use dataclass(kw_only=True) when we drop Python 3.9
@dataclasses.dataclass(frozen=True)
class StyleConfig:
    """A complete set of configuration options for matplotlib-based shap plots."""

    primary_color_positive: ColorType
    primary_color_negative: ColorType
    secondary_color_positive: ColorType
    secondary_color_negative: ColorType
    hlines_color: ColorType
    vlines_color: ColorType
    text_color: ColorType
    tick_labels_color: ColorType

    def asdict(self):
        return dataclasses.asdict(self)


class StyleOptions(TypedDict, total=False):
    """A TypedDict of partial updates to a style configuration"""

    # Nb. There is some duplication here with the StyleConfig dataclass, but
    # it's necessary to provide helpful type hints with `typing.Unpack`.
    # See https://github.com/python/typing/issues/1495
    primary_color_positive: ColorType
    primary_color_negative: ColorType
    secondary_color_positive: ColorType
    secondary_color_negative: ColorType
    hlines_color: ColorType
    vlines_color: ColorType
    text_color: ColorType
    tick_labels_color: ColorType


_shap_defaults = StyleConfig(
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
    # In future, this could allow reading from a persistent config file, like matplotlib rcParams.
    return _shap_defaults


# Singleton instance that determines the current style.
# CAREFUL! To ensure the correct object is picked up, do not import this directly,
# but instead access this at runtime with get_style().
_STYLE = load_default_style()


def get_style() -> StyleConfig:
    """Return all currently active global style configuration options."""
    return _STYLE


def set_style(_style: StyleConfig | None = None, /, **options: Unpack[StyleOptions]) -> None:
    """Set options in the currently active global style configuration.

    Pass keyword arguments to set individual options, or pass a StyleConfig dataclass to replace all options.
    """
    if _style is not None:
        # Unpack the dataclass; any keyword arguments take precendence.
        options = _style.asdict() | options
    global _STYLE
    _STYLE = _apply_options(_STYLE, options)


@contextmanager
def style_context(**options: Unpack[StyleOptions]):
    """Context manager to temporarily change style options.

    NOTE: This is experimental and subject to change!

    Examples
    --------
    To temporarily use black text color instead of the default (white)::

        with shap.plots.style_context(text_color="black"):
            shap.plots.waterfall(...)
    """
    old_style = get_style()
    set_style(**options)
    yield
    set_style(**old_style.asdict())


def _apply_options(style: StyleConfig, changes: StyleOptions) -> StyleConfig:
    """Return a new StyleConfig with any changes applied, handling any invalid options."""
    valid_keys = set(f.name for f in dataclasses.fields(StyleConfig))
    for key in changes.keys():
        if key not in valid_keys:
            raise InvalidStyleOptionError(f"Invalid style config option: {key}")
    return dataclasses.replace(style, **changes)

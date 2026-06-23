"""Tests for shap.utils._exceptions.

The module defines a handful of narrow exception classes that shap
subpackages raise. These tests pin the public shape — class names,
inheritance relationships, message pass-through, raise/catch behaviour —
so an unintended rename or reparent is caught at CI time.

Partial coverage of shap/shap#3690 (test coverage meta-issue).
"""

import pytest

from shap.utils import _exceptions

# (class, expected base class)
_SHAP_EXCEPTIONS: list[tuple[type[BaseException], type[BaseException]]] = [
    (_exceptions.DimensionError, Exception),
    (_exceptions.InvalidAction, Exception),
    (_exceptions.ConvergenceError, Exception),
    (_exceptions.ExplainerError, Exception),
    (_exceptions.InvalidMaskerError, ValueError),
    (_exceptions.InvalidAlgorithmError, ValueError),
    (_exceptions.InvalidFeaturePerturbationError, ValueError),
    (_exceptions.InvalidModelError, ValueError),
    (_exceptions.InvalidClusteringError, ValueError),
    (_exceptions.InvalidStyleOptionError, ValueError),
]


@pytest.mark.parametrize("exc_cls,expected_base", _SHAP_EXCEPTIONS)
def test_exception_inheritance(exc_cls, expected_base):
    """Each shap exception subclasses the documented base.

    The ValueError-subclassed ones matter in particular: downstream user
    code frequently catches ``ValueError`` broadly, so dropping that base
    (e.g. repointing to ``Exception``) would silently break that code.
    """
    assert issubclass(exc_cls, expected_base)
    # ValueError also subclasses Exception, which subclasses BaseException.
    assert issubclass(exc_cls, Exception)


@pytest.mark.parametrize("exc_cls,_", _SHAP_EXCEPTIONS)
def test_exception_preserves_message(exc_cls, _):
    """str(exc) returns the message passed at construction time."""
    msg = f"sentinel message for {exc_cls.__name__}"
    with pytest.raises(exc_cls, match=msg):
        raise exc_cls(msg)


@pytest.mark.parametrize("exc_cls,expected_base", _SHAP_EXCEPTIONS)
def test_exception_caught_by_base(exc_cls, expected_base):
    """A shap exception is caught by its documented public base class.

    This is the guarantee that lets user code keep using
    ``except ValueError`` / ``except Exception`` without knowing about
    shap's internal hierarchy.
    """
    with pytest.raises(expected_base):
        raise exc_cls("should propagate to the base handler")

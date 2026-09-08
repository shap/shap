"""Custom exception classes raised by shap.

These are kept as a small, dependency-free module so any shap subpackage
can import them without pulling in heavy runtime dependencies. User code
can catch the public base-class shape — ``ValueError`` or ``Exception`` —
or catch a specific shap subclass for finer-grained handling.
"""


class DimensionError(Exception):
    """Raised when an array argument has an unsupported shape or rank.

    Typically raised by explainers and plotting functions when the shape
    of ``values``, ``data``, or a clustering matrix doesn't line up with
    what the caller expects.
    """

    pass


class InvalidAction(Exception):
    """Raised when an :class:`shap.actions.Action` configuration is invalid."""

    pass


class ConvergenceError(Exception):
    """Raised when an iterative estimator (e.g. an explainer optimisation loop) fails to converge."""

    pass


class InvalidMaskerError(ValueError):
    """Raised when a masker passed to an explainer is unsupported or mis-configured.

    Subclasses ``ValueError`` so generic ``except ValueError`` handlers
    still catch it.
    """

    pass


class ExplainerError(Exception):
    """Generic runtime error from an Explainer's internals."""

    pass


class InvalidAlgorithmError(ValueError):
    """Raised when an unknown or unsupported ``algorithm`` string is passed to :class:`shap.Explainer`.

    Subclasses ``ValueError`` so generic ``except ValueError`` handlers
    still catch it.
    """

    pass


class InvalidFeaturePerturbationError(ValueError):
    """Raised when an unknown ``feature_perturbation`` option is passed to an explainer.

    Subclasses ``ValueError`` so generic ``except ValueError`` handlers
    still catch it.
    """

    pass


class InvalidModelError(ValueError):
    """Raised when the model passed to an explainer is not of a supported type.

    Subclasses ``ValueError`` so generic ``except ValueError`` handlers
    still catch it.
    """

    pass


class InvalidClusteringError(ValueError):
    """Raised when a clustering / partition tree argument is malformed.

    For example, when a linkage matrix has the wrong shape, or when
    feature indices in a partition tree are out of range.

    Subclasses ``ValueError`` so generic ``except ValueError`` handlers
    still catch it.
    """

    pass


class InvalidStyleOptionError(ValueError):
    """Raised when a plot style option is unknown or of the wrong type.

    See :mod:`shap.plots._style`. Subclasses ``ValueError`` so generic
    ``except ValueError`` handlers still catch it.
    """

    pass

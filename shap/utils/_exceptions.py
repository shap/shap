class DimensionError(Exception):
    """Used for instances where dimensions are either
    not supported or cause errors.
    """

    pass


class InvalidAction(Exception):
    """Raised when an invalid action is provided."""

    pass


class ConvergenceError(Exception):
    """Raised when an optimization or computation fails to converge."""

    pass


class InvalidMaskerError(ValueError):
    """Raised when a masker is invalid or incompatible."""

    pass


class ExplainerError(Exception):
    """Generic errors related to Explainers"""

    pass


class InvalidAlgorithmError(ValueError):
    """Raised when an invalid algorithm is specified."""

    pass


class InvalidFeaturePerturbationError(ValueError):
    """Raised when an invalid feature perturbation method is specified."""

    pass


class InvalidModelError(ValueError):
    """Raised when a model is invalid or incompatible."""

    pass


class InvalidClusteringError(ValueError):
    """Raised when a clustering is invalid or incompatible."""

    pass


class InvalidStyleOptionError(ValueError):
    """Raised when an invalid style option is specified."""

    pass

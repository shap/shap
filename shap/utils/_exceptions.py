class DimensionError(Exception):
    """Used for instances where dimensions are either
    not supported or cause errors.
    """

    pass


class InvalidAction(Exception):
    pass


class ConvergenceError(Exception):
    pass


class InvalidMaskerError(ValueError):
    pass


class ExplainerError(Exception):
    """Generic errors related to Explainers"""

    pass


class InvalidAlgorithmError(ValueError):
    pass


class InvalidFeaturePerturbationError(ValueError):
    pass


class InvalidModelError(ValueError):
    pass


class InvalidClusteringError(ValueError):
    pass


class InvalidStyleOptionError(ValueError):
    pass

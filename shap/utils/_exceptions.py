class DimensionError(Exception):
    """
    Used for instances where dimensions are either
    not supported or cause errors.
    """

    pass

class InvalidMaskerError(Exception):
    pass

class ExplainerError(Exception):
    """
    Generic errors related to Explainers
    """
    pass
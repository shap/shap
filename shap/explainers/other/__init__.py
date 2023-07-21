import warnings

from ._coefficient import Coefficient
from ._lime import LimeTabular
from ._maple import Maple, TreeMaple
from ._random import Random
from ._treegain import TreeGain

__all__ = [
    "Coefficient",
    "LimeTabular",
    "Maple",
    "TreeMaple",
    "Random",
    "TreeGain",
]


# Deprecated class alias with incorrect spelling
def Coefficent(*args, **kwargs):  # noqa
    warnings.warn(
        "Coefficent has been renamed to Coefficient. "
        "The former is deprecated and will be removed in shap 0.45.",
        DeprecationWarning
    )
    return Coefficient(*args, **kwargs)

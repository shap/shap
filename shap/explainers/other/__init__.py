import warnings

from ._coefficient import Coefficient
from ._lime import LimeTabular
from ._maple import Maple, TreeMaple
from ._random import Random
from ._treegain import TreeGain


# Deprecated class alias with incorrect spelling
def Coefficent(*args, **kwargs):  # noqa
    warnings.warn("Coefficent has been renamed to Coefficient", DeprecationWarning)
    return Coefficient(*args, **kwargs)

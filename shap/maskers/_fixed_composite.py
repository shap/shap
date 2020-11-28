import numpy as np
from ._masker import Masker
from ..utils import shape

class FixedComposite(Masker):
    def __init__(self, masker):
        self.masker = masker

    def __call__(self, mask, *args):
        masked_X = self.masker(mask, *args)
        return (masked_X,) + args

    def shape(self, *args):
        return shape(self.masker, *args)
import numpy as np
from ._masker import Masker


class FixedComposite(Masker):
    def __init__(self, masker):
        self.masker = masker
    
    def __call__(self, mask, X):
        masked_X = self.masker(mask, X)
        return masked_X, X
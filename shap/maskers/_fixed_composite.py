import numpy as np
from ._masker import Masker


class FixedComposite(Masker):
    def __init__(self, masker):
        self.masker = masker
    
    def __call__(self, mask, x):
        masked_x = self.masker(mask, x)
        return masked_x, x
        
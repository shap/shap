import numpy as np
from ._masker import Masker

class FixedComposite(Masker):
    def __init__(self, masker):
        self.masker = masker

    def __call__(self, mask, *args):
        masked_X = self.masker(mask, *args)
        return (masked_X,) + args

    def shape(self, *args):
        masker_rows , masker_cols = None, None
        if hasattr(self.masker, "shape"):
            if callable(self.masker.shape):
                mshape = self.masker.shape(*args)
                masker_rows = mshape[0]
                masker_cols = mshape[1]
            else:
                mshape = self.masker.shape
                masker_rows = mshape[0]
                masker_cols = mshape[1]
        else:
            masker_rows = None# # just assuming...
            masker_cols = sum(np.prod(a.shape) for a in args)

        return masker_rows, masker_cols
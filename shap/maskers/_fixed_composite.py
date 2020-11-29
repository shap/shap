import numpy as np
from ._masker import Masker
from ..utils import invariants, variants, shape

class FixedComposite(Masker):
    def __init__(self, masker):
        self.masker = masker

    def __call__(self, mask, *args):
        masked_X = self.masker(mask, *args)
        wrapped_args = []
        for item in args:
            wrapped_args.append(np.array([item]))
        wrapped_args = tuple(wrapped_args)
        return [(masked_X,) + wrapped_args]

    def shape(self, *args):
        return shape(self.masker, *args)

    def invariants(self, *args):
        return invariants(self.masker, *args)

    def clustering(self, *args):
        if callable(self.masker.clustering):
            return self.masker.clustering(*args)
        else:
            return self.masker.clustering

    def mask_shapes(self, *args):
        if hasattr(self.masker, "mask_shapes") and callable(self.masker.mask_shapes):
            return self.masker.mask_shapes(*args)
        else:
            return [a.shape for a in args]

    def feature_names(self, *args):
        if callable(getattr(self.masker, "feature_names", None)):
            return self.masker.feature_names(*args)
        else:
            return None
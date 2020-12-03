import numpy as np
from ._masker import Masker
from ..utils import invariants, variants, shape, data_transform, clustering

class FixedComposite(Masker):
    def __init__(self, masker):
        """ Creates a Composite masker from an underlying masker and returns the original args along with the masked output.

        Parameters
        ----------
        masker: object
            An object of the shap.maskers.Masker base class (eg. Text/Image masker).

        Returns
        -------
        list
            A wrapped tuple consisting of the masked input using the underlying masker appended with the original args in a list.
        """
        self.masker = masker

    def __call__(self, mask, *args):
        """ Computes mask on the args using the masker data attribute and returns list having a wrapped tuple containing masked input with args.
        """
        masked_X = self.masker(mask, *args)
        wrapped_args = []
        for item in args:
            wrapped_args.append(np.array([item]))
        wrapped_args = tuple(wrapped_args)
        if not isinstance(masked_X, tuple):
            masked_X = (masked_X,)
        return masked_X + wrapped_args

    def shape(self, *args):
        return shape(self.masker, *args)

    def invariants(self, *args):
        return invariants(self.masker, *args)

    def clustering(self, *args):
        return clustering(self.masker, *args)

    def data_transform(self, s):
        return data_transform(self.masker, s)

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

    
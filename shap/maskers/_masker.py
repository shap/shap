import numpy as np

from .._serializable import Serializable


class Masker(Serializable):
    """ This is the superclass of all maskers.
    """

    def __call__(self, mask, *args):
        """ Maskers are callable objects that accept the same inputs as the model plus a binary mask.
        """

    def _standardize_mask(self, mask, *args):
        """ This allows users to pass True/False as short hand masks.
        """
        if mask is True or mask is False:
            if callable(self.shape):
                shape = self.shape(*args)
            else:
                shape = self.shape

            if mask is True:
                return np.ones(shape[1], dtype=bool)
            return np.zeros(shape[1], dtype=bool)
        return mask

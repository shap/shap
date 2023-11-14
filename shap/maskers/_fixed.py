import numpy as np

from ._masker import Masker


class Fixed(Masker):
    """ This leaves the input unchanged during masking, and is used for things like scoring labels.

    Sometimes there are inputs to the model that we do not want to explain, but rather we want to
    consider them fixed. The primary example of this is when we explain the loss of the model using
    the labels. These "true" labels are inputs to the function we are explaining, but we don't want
    to attribute credit to them, instead we want to consider them fixed and assign all the credit to
    the model's input features. This is where the Fixed masker can help, since we can apply it to the
    label inputs.
    """
    def __init__(self):
        self.shape = (None, 0)
        self.clustering = np.zeros((0, 4))

    def __call__(self, mask, x):
        return ([x],)

    def mask_shapes(self, x): # pylint: disable=no-self-use,unused-argument
        """ The shape of the masks we expect.
        """
        return [(0,)]

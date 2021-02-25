import pickle
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
        pass

    def __call__(self, x, mask):
        return x

    def save(self, out_file):
        pickle.dump(type(self), out_file)

    @classmethod
    def load(cls, in_file):
        masker_type = pickle.load(in_file)
        if not masker_type == Fixed:
            print("Warning: Saved masker type not same as the one that's attempting to be loaded. Saved masker type: ", masker_type)
        return Fixed._load(in_file)

    @classmethod
    def _load(cls, _):
        return Fixed() # note we have not parameters to load

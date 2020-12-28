import numpy as np
from ._masker import Masker

class OutputComposite(Masker):
    def __init__(self, masker, model, input_args=-1):
        """ Creates a Composite masker from an underlying masker and returns the masked input along with the model output for passed args.

        Parameters
        ----------
        masker: object
            An object of the shap.maskers.Masker base class (eg. Text/Image masker).

        model: object
            An object shap.models.Model base class used to generate output.

        input_args: int
            Number of input arguments to be masked. If input_args < number of arguments passed to call function, then we consider the 
            remaining arguments to be 'y' passed by the user (conceptually similar to teacher forcing technique).

        Returns
        -------
        tuple
            A tuple consisting of the masked input using the underlying masker appended with the model output for passed args.
        """
        self.masker = masker
        self.model = model
        self.input_args = input_args
        # define attributes to be dynamically set
        masker_attributes = ["shape", "invariants", "clustering", "data_transform", "mask_shapes", "feature_names"]
        # set attributes dynamically
        for masker_attribute in masker_attributes:
            if getattr(self.masker, masker_attribute, None) is not None:
                setattr(self, masker_attribute, getattr(self.masker, masker_attribute))

    def __call__(self, mask, *args):
        """ Computes mask on the args using the masker data attribute and returns tuple containing masked input with model output for passed args.
        """
        if self.input_args == -1:
            masked_X = self.masker(mask, *args)
            y = self.model(*args)
        else:
            masked_X = self.masker(mask, *args[:self.input_args])
            y = args[self.input_args:]
        # wrap model output
        if not isinstance(y, tuple):
            y = (y,)
        # wrap masked input
        if not isinstance(masked_X, tuple):
            masked_X = (masked_X,)
        return masked_X + y
    
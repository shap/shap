import numpy as np

from .._serializable import Deserializer, Serializer
from ._masker import Masker


class FixedComposite(Masker):
    """ A masker that outputs both the masked data and the original data as a pair.
    """

    def __init__(self, masker):
        """ Creates a Composite masker from an underlying masker and returns the original args along with the masked output.

        Parameters
        ----------
        masker: object
            An object of the shap.maskers.Masker base class (eg. Text/Image masker).

        Returns
        -------
        tuple
            A tuple consisting of the masked input using the underlying masker appended with the original args in a list.
        """
        self.masker = masker

        # copy attributes from the masker we are wrapping
        masker_attributes = ["shape", "invariants", "clustering", "data_transform", "mask_shapes", "feature_names", "text_data", "image_data"]
        for masker_attribute in masker_attributes:
            if getattr(self.masker, masker_attribute, None) is not None:
                setattr(self, masker_attribute, getattr(self.masker, masker_attribute))

    def __call__(self, mask, *args):
        """ Computes mask on the args using the masker data attribute and returns tuple containing masked input with args.
        """
        masked_X = self.masker(mask, *args)
        wrapped_args = []
        for item in args:
            wrapped_args.append(np.array([item]))
        wrapped_args = tuple(wrapped_args)
        if not isinstance(masked_X, tuple):
            masked_X = (masked_X,)
        return masked_X + wrapped_args

    def save(self, out_file):
        """ Write a FixedComposite masker to a file stream.
        """
        super().save(out_file)

        # Increment the verison number when the encoding changes!
        with Serializer(out_file, "shap.maskers.FixedComposite", version=0) as s:
            s.save("masker", self.masker)

    @classmethod
    def load(cls, in_file, instantiate=True):
        """ Load a FixedComposite masker from a file stream.
        """
        if instantiate:
            return cls._instantiated_load(in_file)

        kwargs = super().load(in_file, instantiate=False)
        with Deserializer(in_file, "shap.maskers.FixedComposite", min_version=0, max_version=0) as s:
            kwargs["masker"] = s.load("masker")
        return kwargs

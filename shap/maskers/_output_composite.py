from .._serializable import Deserializer, Serializer
from ._masker import Masker


class OutputComposite(Masker):
    """ A masker that is a combination of a masker and a model and outputs both masked args and the model's output.
    """

    def __init__(self, masker, model):
        """ Creates a masker from an underlying masker and and model.

        This masker returns the masked input along with the model output for the passed args.

        Parameters
        ----------
        masker: object
            An object of the shap.maskers.Masker base class (eg. Text/Image masker).

        model: object
            An object shap.models.Model base class used to generate output.

        Returns
        -------
        tuple
            A tuple consisting of the masked input using the underlying masker appended with the model output for passed args.
        """
        self.masker = masker
        self.model = model

        # copy attributes from the masker we are wrapping
        masker_attributes = ["shape", "invariants", "clustering", "data_transform", "mask_shapes", "feature_names", "text_data", "image_data"]
        for masker_attribute in masker_attributes:
            if getattr(self.masker, masker_attribute, None) is not None:
                setattr(self, masker_attribute, getattr(self.masker, masker_attribute))

    def __call__(self, mask, *args):
        """ Mask the args using the masker and return a tuple containing the masked input and the model output on the args.
        """
        masked_X = self.masker(mask, *args)
        y = self.model(*args)
        # wrap model output
        if not isinstance(y, tuple):
            y = (y,)
        # wrap masked input
        if not isinstance(masked_X, tuple):
            masked_X = (masked_X,)
        return masked_X + y

    def save(self, out_file):
        """ Write a OutputComposite masker to a file stream.
        """
        super().save(out_file)

        # Increment the version number when the encoding changes!
        with Serializer(out_file, "shap.maskers.OutputComposite", version=0) as s:
            s.save("masker", self.masker)
            s.save("model", self.model)

    @classmethod
    def load(cls, in_file, instantiate=True):
        """ Load a OutputComposite masker from a file stream.
        """
        if instantiate:
            return cls._instantiated_load(in_file)

        kwargs = super().load(in_file, instantiate=False)
        with Deserializer(in_file, "shap.maskers.OutputComposite", min_version=0, max_version=0) as s:
            kwargs["masker"] = s.load("masker")
            kwargs["model"] = s.load("model")
        return kwargs

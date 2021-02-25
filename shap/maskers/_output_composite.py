from ._masker import Masker
import pickle

class OutputComposite(Masker):
    def __init__(self, masker, model):
        """ Creates a Composite masker from an underlying masker and returns the masked input along with the model output for passed args.

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
        # define attributes to be dynamically set
        masker_attributes = ["shape", "invariants", "clustering", "data_transform", "mask_shapes", "feature_names"]
        # set attributes dynamically
        for masker_attribute in masker_attributes:
            if getattr(self.masker, masker_attribute, None) is not None:
                setattr(self, masker_attribute, getattr(self.masker, masker_attribute))

    def __call__(self, mask, *args):
        """ Computes mask on the args using the masker data attribute and returns tuple containing masked input with model output for passed args.
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
        super(OutputComposite, self).save(out_file)
        pickle.dump(type(self.masker), out_file)
        pickle.dump(type(self.model), out_file)
        self.masker.save(out_file)
        self.model.save(out_file)

    @classmethod
    def load(cls, in_file):
        masker_type = pickle.load(in_file)
        if not masker_type == cls:
            print("Warning: Saved masker type not same as the one that's attempting to be loaded. Saved masker type: ", masker_type)
        return OutputComposite._load(in_file)

    @classmethod
    def _load(cls, in_file):
        sub_masker_type = pickle.load(in_file)
        sub_model_type = pickle.load(in_file)
        masker = sub_masker_type.load(in_file)
        model = sub_model_type.load(in_file)
        outputcomposite_masker = OutputComposite(masker, model)
        return outputcomposite_masker
    
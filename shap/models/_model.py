import pickle
import inspect
import warnings
import cloudpickle

class Model():
    """ This is the superclass of all models.
    """

    def __init__(self, model=None):
        """ Wrap a callable model as a SHAP Model object.
        """
        if isinstance(model, Model):
            self.model = model.model
        else:
            self.model = model

    def __call__(self, *args):
        return self.model(*args)

    def save(self, out_file):
        """ Save the model to the given file stream.
        """
        pickle.dump(type(self), out_file)
        cloudpickle.dump(self.model, out_file)

    @classmethod
    def load(cls, in_file):
        """ Load a model from the given file stream.
        """
        model_type = pickle.load(in_file)
        if model_type is None:
            warnings.warn("A shap.Model was not found in saved file, please set model before using explainer.")
            return None

        if inspect.isclass(model_type) and issubclass(model_type, Model):
            return model_type._load(in_file) # pylint: disable=protected-access

        raise Exception("Invalid model type loaded from file:", model_type)

    @classmethod
    def _load(cls, in_file):
        return Model(
            model=cloudpickle.load(in_file)
        )
        
import pickle
import cloudpickle

class Model():
    def __init__(self, model=None):
        """ This superclass of all model objects.
        """
        if type(model) == Model:
            self.model = model.model
        else:
            self.model = model
    
    def __call__(self, *args):
        return self.model(*args)
    
    def save(self, out_file, *args):
        pickle.dump(type(self), out_file)
        cloudpickle.dump(self.model, out_file)
    
    @classmethod
    def load(cls, in_file):
        model_type = pickle.load(in_file)
        if model_type is None:
            print("Warning: model was not found in saved file, please set model before using explainer.")
            return None
            
        return cloudpickle.load(in_file)
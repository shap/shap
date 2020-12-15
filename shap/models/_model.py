import pickle

class Model():
    def __init__(self, model=None):
        """ This superclass of all model objects.
        """
        self.model = model
    
    def __call__(self, *args):
        return self.model(*args)
    
    def save(self, model, out_file):
        pickle.dump(model, out_file) # TODO change serialization methods based on model
    
    @classmethod
    def load(cls, in_file):
        return pickle.load(in_file)
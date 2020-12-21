import pickle

class Masker():
    def __init__(self):
        """ This superclass of all masker objects.
        """
    
    def __call__(self, mask=None, *args):
        """ Maskers are callable objects that accept the same inputs as the model plus a binary mask.
        """
        pass

    def save(self, out_file):
        """ Serializes the type of subclass of masker used, this will be used during deserialization
        """
        pickle.dump(type(self), out_file)

    @classmethod
    def load(cls, in_file):
        """ Deserializes the masker subtype, and calls respective load function
        """
        masker_type = pickle.load(in_file)
        if masker_type is None:
            print("Warning: masker was not found in saved file, please set masker before using explainer.")
            return None
        return masker_type._load(in_file)


class Masker():
    def __init__(self):
        """ This superclass of all masker objects.
        """
    
    def __call__(self, mask=None, *args):
        """ Maskers are callable objects that accept the same inputs as the model plus a binary mask.
        """
        pass
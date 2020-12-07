

class Model():
    def __init__(self, model=None):
        """ This superclass of all model objects.
        """
        self.model = model
    
    def __call__(self, *args):
        return self.model(*args)
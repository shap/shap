class Action():
    """ Abstract action class.
    """
    def __lt__(self, x):
        return False
    
    def __repr__(self):
        return f"<Action '{self.__str__()}'>"
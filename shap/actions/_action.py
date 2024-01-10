class Action:
    """ Abstract action class.
    """
    def __lt__(self, other_action):
        return self.cost < other_action.cost

    def __repr__(self):
        return f"<Action '{self.__str__()}'>"

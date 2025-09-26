class Action:
    """Abstract action class."""

    def __init__(self, cost):
        self.cost = cost
        self._group_index = 0
        self._grouped_index = 0

    def __call__(self, *args):
        raise NotImplementedError

    def __lt__(self, other_action):
        return self.cost < other_action.cost

    def __repr__(self):
        return f"<Action '{self.__str__()}'>"

class Action:
    """Abstract action class."""

    def __init__(self, cost):
        if not isinstance(cost, (int, float)):
            raise TypeError("Cost must be a number")
        if cost < 0:
            raise ValueError("Cost must be non-negative")

        self.cost = cost
        self._group_index = 0
        self._grouped_index = 0

    def __call__(self, *args):
        raise NotImplementedError

    def __lt__(self, other_action):
        if not isinstance(other_action, Action):
            return NotImplemented
        return self.cost < other_action.cost

    def __repr__(self):
        return f"<Action '{self.__str__()}'>"

import copy
import queue
import warnings

from ..utils._exceptions import ConvergenceError, InvalidAction
from ._action import Action


class ActionOptimizer:
    def __init__(self, model, actions):
        self.model = model
        warnings.warn(
            "Note that ActionOptimizer is still in an alpha state and is subjust to API changes."
        )
        # actions go into mutually exclusive groups
        self.action_groups = []
        for group in actions:

            if issubclass(type(group), Action):
                group._group_index = len(self.action_groups)
                group._grouped_index = 0
                self.action_groups.append([copy.copy(group)])
            elif issubclass(type(group), list):
                group = sorted([copy.copy(v) for v in group], key=lambda a: a.cost)
                for i, v in enumerate(group):
                    v._group_index = len(self.action_groups)
                    v._grouped_index = i
                self.action_groups.append(group)
            else:
                raise InvalidAction(
                    "A passed action was not an Action or list of actions!"
                )

    def __call__(self, *args, max_evals=10000):

        # init our queue with all the least costly actions
        q = queue.PriorityQueue()
        for i in range(len(self.action_groups)):
            group = self.action_groups[i]
            q.put((group[0].cost, [group[0]]))

        nevals = 0
        while not q.empty():

            # see if we have exceeded our runtime budget
            nevals += 1
            if nevals > max_evals:
                raise ConvergenceError(
                    f"Failed to find a solution with max_evals={max_evals}! Try reducing the number of actions or increasing max_evals."
                )

            # get the next cheapest set of actions we can do
            cost, actions = q.get()

            # apply those actions
            args_tmp = copy.deepcopy(args)
            for a in actions:
                a(*args_tmp)

            # if the model is now satisfied we are done!!
            v = self.model(*args_tmp)
            if v:
                return actions

            # if not then we add all possible follow-on actions to our queue
            else:
                for i in range(len(self.action_groups)):
                    group = self.action_groups[i]

                    # look to to see if we already have a action from this group, if so we need to
                    # move to a more expensive action in the same group
                    next_ind = 0
                    prev_in_group = -1
                    for j, a in enumerate(actions):
                        if a._group_index == i:
                            next_ind = max(next_ind, a._grouped_index + 1)
                            prev_in_group = j

                    # we are adding a new action type
                    if prev_in_group == -1:
                        new_actions = actions + [group[next_ind]]
                    # we are moving from one action to a more expensive one in the same group
                    elif next_ind < len(group):
                        new_actions = copy.copy(actions)
                        new_actions[prev_in_group] = group[next_ind]
                    # we don't have a more expensive action left in this group
                    else:
                        new_actions = None

                    # add the new option to our queue
                    if new_actions is not None:
                        q.put((sum([a.cost for a in new_actions]), new_actions))

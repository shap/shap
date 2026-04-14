"""Tests for the action optimizer."""

from __future__ import annotations

import pytest

from shap.actions._action import Action
from shap.actions._optimizer import ActionOptimizer
from shap.utils._exceptions import ConvergenceError, InvalidAction

pytestmark = pytest.mark.filterwarnings(
    "ignore:Note that ActionOptimizer is still in an alpha state and is subject to API changes."
)


class AddValueAction(Action):
    """Action that increments one key in a mapping-like sample."""

    def __init__(self, key: str, amount: int, cost: int):
        super().__init__(cost=cost)
        self.key = key
        self.amount = amount

    def __call__(self, sample):
        sample[self.key] += self.amount

    def __str__(self):
        return f"Add {self.amount} to {self.key}."


def test_finds_lowest_cost_solution_without_mutating_input():
    sample = {"a": 1, "b": 1, "c": 1}

    def passed(x):
        return sum(x.values()) >= 10

    possible_actions = [
        [AddValueAction("a", 1, cost=7), AddValueAction("a", 3, cost=10)],
        AddValueAction("b", 4, cost=6),
        [AddValueAction("c", 2, cost=3), AddValueAction("c", 5, cost=9)],
    ]
    optimizer = ActionOptimizer(passed, possible_actions)

    selected_actions = optimizer(sample)

    assert selected_actions is not None
    assert sum(action.cost for action in selected_actions) == 15
    assert sorted((action.key, action.amount) for action in selected_actions) == [("b", 4), ("c", 5)]
    assert sample == {"a": 1, "b": 1, "c": 1}


def test_replaces_with_more_expensive_action_in_same_group():
    def passed(x):
        return x["a"] >= 2

    possible_actions = [[AddValueAction("a", 1, cost=1), AddValueAction("a", 2, cost=2)]]
    optimizer = ActionOptimizer(passed, possible_actions)

    selected_actions = optimizer({"a": 0})

    assert selected_actions is not None
    assert len(selected_actions) == 1
    assert selected_actions[0].amount == 2


def test_returns_none_when_no_solution_exists():
    possible_actions = [[AddValueAction("a", 1, cost=1)]]
    optimizer = ActionOptimizer(lambda x: False, possible_actions)

    assert optimizer({"a": 0}) is None


def test_raises_convergence_error_when_eval_budget_exceeded():
    possible_actions = [
        [AddValueAction("a", 1, cost=1), AddValueAction("a", 2, cost=2)],
        AddValueAction("b", 1, cost=1),
    ]
    optimizer = ActionOptimizer(lambda x: False, possible_actions)

    with pytest.raises(ConvergenceError, match="max_evals=1"):
        optimizer({"a": 0, "b": 0}, max_evals=1)


def test_sorts_group_actions_and_assigns_internal_indices():
    group_actions = [
        AddValueAction("a", 1, cost=3),
        AddValueAction("a", 2, cost=1),
        AddValueAction("a", 3, cost=2),
    ]
    standalone_action = AddValueAction("b", 1, cost=4)
    optimizer = ActionOptimizer(lambda x: False, [group_actions, standalone_action])

    sorted_group = optimizer.action_groups[0]
    standalone_group = optimizer.action_groups[1]

    assert [action.cost for action in sorted_group] == [1, 2, 3]
    assert [(action._group_index, action._grouped_index) for action in sorted_group] == [(0, 0), (0, 1), (0, 2)]
    assert (standalone_group[0]._group_index, standalone_group[0]._grouped_index) == (1, 0)
    assert sorted_group[0] is not group_actions[1]
    assert standalone_group[0] is not standalone_action


def test_invalid_action_type_raises():
    with pytest.raises(InvalidAction):
        ActionOptimizer(lambda x: True, [object()])

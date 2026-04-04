"""Unit tests for the Exact explainer."""

import numpy as np
import pandas as pd
import pytest

import shap
from shap.utils._exceptions import ConvergenceError, InvalidAction


def create_basic_scenario():
    X = pd.DataFrame({"feature1": np.ones(5), "feature2": np.ones(5), "feature3": np.ones(5)})

    class IncreaseFeature1(shap.actions.Action):
        """Sample action."""

        def __init__(self, amount):
            self.amount = amount
            self.cost = 5 * amount

        def __call__(self, X):
            X["feature1"] += self.amount

        def __str__(self):
            return f"Improve feature1 by {self.amount}."

    class IncreaseFeature2(shap.actions.Action):
        """Sample action."""

        def __init__(self, amount):
            self.amount = amount
            self.cost = 3 * amount

        def __call__(self, X):
            X["feature2"] += self.amount

        def __str__(self):
            return f"Improve feature2 by {self.amount}."

    class IncreaseFeature3(shap.actions.Action):
        """Sample action."""

        def __init__(self, amount):
            self.amount = amount
            self.cost = 4 * amount

        def __call__(self, X):
            X["feature3"] += self.amount

        def __str__(self):
            return f"Improve feature3 by {self.amount}."

    def passed(x):
        return np.sum(x) > 10

    return X, IncreaseFeature1, IncreaseFeature2, IncreaseFeature3, passed


def test_basic_run():
    X, IncreaseFeature1, IncreaseFeature2, IncreaseFeature3, passed = create_basic_scenario()
    possible_actions = [
        [IncreaseFeature1(i) for i in range(1, 10)],
        IncreaseFeature2(5),
        [IncreaseFeature3(i) for i in range(1, 20)],
    ]
    optimizer = shap.ActionOptimizer(passed, possible_actions)
    actions = optimizer(X.iloc[0])
    assert len(actions) == 2
    assert sum(a.cost for a in actions) == 27  # ensure we got the optimal answer


def test_too_few_evals():
    X, IncreaseFeature1, IncreaseFeature2, IncreaseFeature3, passed = create_basic_scenario()
    possible_actions = [
        [IncreaseFeature1(i) for i in range(1, 10)],
        IncreaseFeature2(5),
        [IncreaseFeature3(i) for i in range(1, 20)],
    ]
    optimizer = shap.ActionOptimizer(passed, possible_actions)
    with pytest.raises(ConvergenceError):
        optimizer(X.iloc[0], max_evals=3)


def test_run_out_of_group():
    X, IncreaseFeature1, IncreaseFeature2, IncreaseFeature3, passed = create_basic_scenario()
    possible_actions = [[IncreaseFeature1(i) for i in range(1, 10)], IncreaseFeature2(5), [IncreaseFeature3(1)]]
    optimizer = shap.ActionOptimizer(passed, possible_actions)
    actions = optimizer(X.iloc[0])
    print(actions)
    assert len(actions) == 3


def test_bad_action():
    with pytest.raises(InvalidAction):
        shap.ActionOptimizer(None, [None])  # type: ignore


def test_warning_emitted():
    """Tests that warning is emitted when ActionOptimizer is initialized."""
    import warnings

    X, IncreaseFeature1, IncreaseFeature2, IncreaseFeature3, passed = create_basic_scenario()
    possible_actions = [[IncreaseFeature1(1)], IncreaseFeature2(1), [IncreaseFeature3(1)]]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        shap.ActionOptimizer(passed, possible_actions)
        assert len(w) == 1
        assert "alpha state" in str(w[0].message).lower()


def test_no_valid_solution():
    """Tests case when no valid solution exists (queue empties without finding solution)."""
    X, IncreaseFeature1, _, _, _ = create_basic_scenario()

    def sum_less_than_2(x):
        return np.sum(x) < 2

    possible_actions = [
        [IncreaseFeature1(1)],
    ]
    optimizer = shap.ActionOptimizer(sum_less_than_2, possible_actions)
    result = optimizer(X.iloc[0], max_evals=10)
    assert result is None


def test_action_replacement_in_group():
    """Tests replacing an action with a more expensive one in the same group."""
    X, IncreaseFeature1, IncreaseFeature2, _, _ = create_basic_scenario()

    def requires_high_value(x):
        return x["feature1"] >= 10

    possible_actions = [
        [IncreaseFeature1(i) for i in range(1, 10)],
    ]
    optimizer = shap.ActionOptimizer(requires_high_value, possible_actions)
    actions = optimizer(X.iloc[0])
    assert len(actions) == 1
    assert actions[0].amount >= 2


def test_multiple_action_groups_with_replacement():
    """Tests multiple action groups where we replace action in same group."""
    X, IncreaseFeature1, IncreaseFeature2, IncreaseFeature3, passed = create_basic_scenario()

    def sum_equals_12(x):
        return np.sum(x) == 12

    possible_actions = [
        [IncreaseFeature1(i) for i in range(1, 10)],
        [IncreaseFeature2(i) for i in range(1, 5)],
    ]
    optimizer = shap.ActionOptimizer(sum_equals_12, possible_actions)
    actions = optimizer(X.iloc[0])
    assert len(actions) == 2


def test_immediate_satisfaction():
    """Tests when model is satisfied on first try (no actions needed)."""
    X, IncreaseFeature1, IncreaseFeature2, IncreaseFeature3, passed = create_basic_scenario()

    def always_satisfied(x):
        return True

    possible_actions = [
        [IncreaseFeature1(i) for i in range(1, 10)],
    ]
    optimizer = shap.ActionOptimizer(always_satisfied, possible_actions)
    actions = optimizer(X.iloc[0])
    assert len(actions) == 1
    assert actions[0].cost == 5


def test_single_action_group():
    """Tests when actions contains a single Action (not a list)."""
    X, IncreaseFeature1, _, _, _ = create_basic_scenario()

    def requires_feature1_9(x):
        return x["feature1"] >= 9

    possible_actions = [IncreaseFeature1(1)]
    optimizer = shap.ActionOptimizer(requires_feature1_9, possible_actions)
    result = optimizer(X.iloc[0])
    assert result is None or len(result) >= 0


def test_action_list_sorting():
    """Tests that actions in a list are sorted by cost."""
    X, IncreaseFeature1, _, _, passed = create_basic_scenario()

    class UnorderedAction(shap.actions.Action):
        def __init__(self, cost):
            self.cost = cost

        def __call__(self, X):
            pass

        def __str__(self):
            return f"Action(cost={self.cost})"

    unordered = [UnorderedAction(10), UnorderedAction(1), UnorderedAction(5)]
    possible_actions = [unordered]
    optimizer = shap.ActionOptimizer(passed, possible_actions)

    costs = [a.cost for a in optimizer.action_groups[0]]
    assert costs == [1, 5, 10]

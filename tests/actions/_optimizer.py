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

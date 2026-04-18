"""Unit tests for SHAP actions."""

import numpy as np
import pandas as pd
import pytest

import shap


def test_create_and_run():
    X = pd.DataFrame({"feature1": np.ones(5), "feature2": np.ones(5)})

    class IncreaseFeature1(shap.actions.Action):
        """Sample action."""

        def __init__(self, amount):
            self.amount = amount
            self.cost = 5 * amount

        def __call__(self, X):
            X["feature1"] += self.amount

        def __str__(self):
            return f"Improve feature1 by {self.amount}."

    action = IncreaseFeature1(4)
    action.__repr__()
    assert not (action < action)
    row = X.iloc[0].copy()
    action(row)
    assert row["feature1"] == 5


def test_action_base_call_raises_not_implemented():
    """Test that the base Action class raises NotImplementedError when called."""
    base_action = shap.actions.Action(cost=10)
    with pytest.raises(NotImplementedError):
        base_action()


def test_action_comparisons_and_properties():
    """Test the less-than operator, string representation, and base properties."""

    class MockAction(shap.actions.Action):
        def __str__(self):
            return "Test action string"

    action_cheap = MockAction(cost=5)
    action_expensive = MockAction(cost=15)

    # Test __lt__ for different costs
    assert action_cheap < action_expensive
    assert not (action_expensive < action_cheap)

    # Test __repr__ to ensure formatting is correct
    assert repr(action_cheap) == "<Action 'Test action string'>"

    # Test default initialized properties
    assert action_cheap._group_index == 0
    assert action_cheap._grouped_index == 0

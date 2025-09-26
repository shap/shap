"""Unit tests for the Exact explainer."""

import numpy as np
import pandas as pd

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
    action(X.iloc[0])
    assert X["feature1"][0] == 5

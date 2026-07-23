from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier

from shap.datasets import adult
from shap.explainers import TreeExplainer


class TreeSuite:
    # Adapted from tests/explainers/test_tree.py
    max_samples = 100

    def setup(self):
        self.X, self.y = adult(n_points=self.max_samples)
        self.X = self.X.values

        self.regressor = RandomForestRegressor(random_state=0)
        self.regressor.fit(self.X, self.y)

        self.classifier = RandomForestClassifier(random_state=0)
        self.classifier.fit(self.X, self.y)

        self.xgboost = XGBClassifier(tree_method="exact", base_score=0.5)
        self.xgboost.fit(self.X, self.y)

    def time_single_output(self):
        ex = TreeExplainer(self.regressor)
        _ = ex(self.X)

    def time_multi_output(self):
        ex = TreeExplainer(self.classifier)
        _ = ex(self.X)

    def time_interactions(self):
        ex = TreeExplainer(self.regressor)
        _ = ex(self.X, interactions=True)

    def time_xgboost(self):
        ex = TreeExplainer(self.xgboost)
        _ = ex(self.X)

    def time_xgboost_interactions(self):
        ex = TreeExplainer(self.xgboost)
        _ = ex(self.X, interactions=True)

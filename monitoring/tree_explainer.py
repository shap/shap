"""CPU path-dependent values and interactions, without downloads or GPU dependencies."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from shap import TreeExplainer


class TreeQuadratureSuite:
    params = ([4, 8, 12], [1, 3])
    param_names = ["max_depth", "outputs"]

    def setup(self, max_depth, outputs):
        rng = np.random.default_rng(0)
        rows = rng.normal(size=(500, 16))
        target = rows[:, 0] * rows[:, 1] + rows[:, 2] ** 2
        if outputs > 1:
            target = np.column_stack([target * (i + 1) for i in range(outputs)])
        model = RandomForestRegressor(n_estimators=20, max_depth=max_depth, random_state=0, n_jobs=1)
        model.fit(rows, target)
        self.explainer = TreeExplainer(model, feature_perturbation="tree_path_dependent")
        self.rows = rows[:50]

    def time_values(self, max_depth, outputs):
        self.explainer.shap_values(self.rows)

    def time_interactions(self, max_depth, outputs):
        self.explainer.shap_interaction_values(self.rows)

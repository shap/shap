from xgboost import XGBClassifier

from shap.datasets import adult
from shap.explainers import PartitionExplainer


class PartitionSuite:
    # Adapted from tests/explainers/test_partition.py
    # TODO: should we add translation tests here too?
    # This would introduce a dependency on torch and transformers.
    max_samples = 100

    def setup(self):
        self.model = XGBClassifier(tree_method="exact", base_score=0.5)

        # get a dataset on income prediction
        self.X, self.y = adult()
        if self.max_samples is not None:
            self.X = self.X.iloc[: self.max_samples]
            self.y = self.y[: self.max_samples]
        self.X = self.X.values

        # fit the model on the data
        self.model.fit(self.X, self.y)

    def time_single_output(self):
        ex = PartitionExplainer(self.model.predict, self.X)
        _ = ex(self.X)

    def time_multi_output(self):
        ex = PartitionExplainer(self.model.predict_proba, self.X)
        _ = ex(self.X)

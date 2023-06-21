import numpy as np

from .._explainer import Explainer


class TreeGain(Explainer):
    """ Simply returns the global gain/gini feature importances for tree models.

    This is only for benchmark comparisons and is not meant to approximate SHAP values.
    """
    def __init__(self, model):
        if str(type(model)).endswith("sklearn.tree.tree.DecisionTreeRegressor'>"):
            pass
        elif str(type(model)).endswith("sklearn.tree.tree.DecisionTreeClassifier'>"):
            pass
        elif str(type(model)).endswith("sklearn.ensemble.forest.RandomForestRegressor'>"):
            pass
        elif str(type(model)).endswith("sklearn.ensemble.forest.RandomForestClassifier'>"):
            pass
        elif str(type(model)).endswith("xgboost.sklearn.XGBRegressor'>"):
            pass
        elif str(type(model)).endswith("xgboost.sklearn.XGBClassifier'>"):
            pass
        else:
            raise NotImplementedError("The passed model is not yet supported by TreeGainExplainer: " + str(type(model)))
        assert hasattr(model, "feature_importances_"), "The passed model does not have a feature_importances_ attribute!"
        self.model = model

    def attributions(self, X):
        return np.tile(self.model.feature_importances_, (X.shape[0], 1))

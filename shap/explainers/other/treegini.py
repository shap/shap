from ..explainer import Explainer
import numpy as np

class TreeGiniExplainer(Explainer):
    """ Simply returns the global gini feature importances for tree models.

    This is only for benchmark comparisons and is not meant to approximate SHAP values.
    """
    def __init__(self, model):
        if str(type(model)).endswith("sklearn.tree.tree.DecisionTreeRegressor'>"):
            pass
        elif str(type(model)).endswith("sklearn.ensemble.forest.RandomForestRegressor'>"):
            pass
        else:
            raise Exception("The passed model is not yet supported by TreeGiniExplainer: " + str(type(model)))
        assert hasattr(model, "feature_importances_"), "The passed model does not have a feature_importances_ attribute!"
        self.model = model

    def attributions(self, X):
        return np.tile(self.model.feature_importances_, (X.shape[0], 1))

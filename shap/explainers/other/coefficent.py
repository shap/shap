from ..explainer import Explainer

class CoefficentExplainer(Explainer):
    """ Simply returns the model coefficents as the feature attributions.

    This is only for benchmark comparisons and does not approximate SHAP values in a
    meaningful way.
    """
    def __init__(self, model):
        assert hasattr(self.model, "coef_"), "The passed model does not have a coef_ attribute!"
        self.model = model

    def attributions(self, X):
        return self.model.coef_ * np.ones(X.shape)

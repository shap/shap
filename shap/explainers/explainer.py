
class Explainer(object):
    """ This is the superclass of all explainers.
    """

    def shap_values(self, X):
        raise Exception("SHAP values not implemented for this explainer!")

    def attributions(self, X):
        return self.shap_values(X)

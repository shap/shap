
class Explainer(object):
    """ This is the superclass of all explainers.
    """

    def attributions(self, X):
        return self.shap_values(X)

from .deep_pytorch import PyTorchDeepExplainer
from .deep_tf import TFDeepExplainer
from shap.explainers.explainer import Explainer


class DeepExplainer(Explainer):
    """ Meant to approximate SHAP values for deep learning models.

    This is an enhanced version of the DeepLIFT algorithm (Deep SHAP) where, similar to Kernel SHAP, we
    approximate the conditional expectations of SHAP values using a selection of background samples.
    Lundberg and Lee, NIPS 2017 showed that the per node attribution rules in DeepLIFT (Shrikumar,
    Greenside, and Kundaje, arXiv 2017) can be chosen to approximate Shapley values. By integrating
    over many backgound samples DeepExplainer estimates approximate SHAP values such that they sum
    up to the difference between the expected model output on the passed background samples and the
    current model output (f(x) - E[f(x)]).
    """

    def __init__(self, model, data, session=None, learning_phase_flags=None):
        # first, we need to find the framework
        if type(model) is tuple:
            a, b = model
            try:
                a.named_parameters()
                framework = 'pytorch'
            except:
                framework = 'tensorflow'
        else:
            try:
                model.named_parameters()
                framework = 'pytorch'
            except:
                framework = 'tensorflow'

        if framework == 'tensorflow':
            self.explainer = TFDeepExplainer(model, data, session, learning_phase_flags)
        elif framework == 'pytorch':
            self.explainer = PyTorchDeepExplainer(model, data)

    def shap_values(self, X, ranked_outputs=None, output_rank_order='max'):
        """ Return approximate SHAP values for the model applied to the data given by X.

        Parameters
        ----------
        X : list, numpy.array, or pandas.DataFrame
            A tensor (or list of tensors) of samples (where X.shape[0] == # samples) on which to
            explain the model's output.

        ranked_outputs : None or int
            If ranked_outputs is None then we explain all the outputs in a multi-output model. If
            ranked_outputs is a positive integer then we only explain that many of the top model
            outputs (where "top" is determined by output_rank_order). Note that this causes a pair
            of values to be returned (shap_values, indexes), where shap_values is a list of numpy
            arrays for each of the output ranks, and indexes is a matrix that indicates for each sample
            which output indexes were choses as "top".

        output_rank_order : "max", "min", or "max_abs"
            How to order the model outputs when using ranked_outputs, either by maximum, minimum, or
            maximum absolute value.

        Returns
        -------
        For a models with a single output this returns a tensor of SHAP values with the same shape
        as X. For a model with multiple outputs this returns a list of SHAP value tensors, each of
        which are the same shape as X. If ranked_outputs is None then this list of tensors matches
        the number of model outputs. If ranked_outputs is a positive integer a pair is returned
        (shap_values, indexes), where shap_values is a list of tensors with a length of
        ranked_outputs, and indexes is a matrix that indicates for each sample which output indexes
        were chosen as "top".
        """
        return self.explainer.shap_values(X, ranked_outputs, output_rank_order)

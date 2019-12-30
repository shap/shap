import numpy as np
import scipy as sp
import warnings
from .explainer import Explainer
from ..common import safe_isinstance

class AdditiveExplainer(Explainer):
    """ Computes SHAP values for generalized additive models.

    This assumes that the model given only has first order effects. Extending this to
    2nd and third order effects is future work (if you apply this to those models right now
    you will get incorrect answers that fail additivity).

    Parameters
    ----------
    model : function or ExplainableBoostingRegressor
        User supplied additive model either as either a function or a model object.

    data : numpy.array, pandas.DataFrame
        The background dataset to use for computing conditional expectations.
    feature_perturbation : "interventional"
        Only the standard interventional SHAP values are supported by AdditiveExplainer right now.
    """

    def __init__(self, model, data, feature_perturbation="interventional"):
        if feature_perturbation != "interventional":
            raise Exception("Unsupported type of feature_perturbation provided: " + feature_perturbation)

        if safe_isinstance(model, "interpret.glassbox.ebm.ebm.ExplainableBoostingRegressor"):
            self.f = model.predict
        elif callable(model):
            self.f = model
        else:
            raise ValueError("The passed model must be a recognized object or a function!")
        
        # convert dataframes
        if safe_isinstance(data, "pandas.core.series.Series"):
            data = data.values
        elif safe_isinstance(data, "pandas.core.frame.DataFrame"):
            data = data.values
        self.data = data
        
        # compute the expected value of the model output
        self.expected_value = self.f(data).mean()
        
        # pre-compute per-feature offsets
        tmp = np.zeros(data.shape)
        self._zero_offest = self.f(tmp).mean()
        self._feature_offset = np.zeros(data.shape[1])
        for i in range(data.shape[1]):
            tmp[:,i] = data[:,i]
            self._feature_offset[i] = self.f(tmp).mean() - self._zero_offest
            tmp[:,i] = 0


    def shap_values(self, X):
        """ Estimate the SHAP values for a set of samples.

        Parameters
        ----------
        X : numpy.array, pandas.DataFrame or scipy.csr_matrix
            A matrix of samples (# samples x # features) on which to explain the model's output.

        Returns
        -------
        For models with a single output this returns a matrix of SHAP values
        (# samples x # features). Each row sums to the difference between the model output for that
        sample and the expected value of the model output (which is stored as expected_value
        attribute of the explainer).
        """

        # convert dataframes
        if str(type(X)).endswith("pandas.core.series.Series'>"):
            X = X.values
        elif str(type(X)).endswith("'pandas.core.frame.DataFrame'>"):
            X = X.values

        #assert str(type(X)).endswith("'numpy.ndarray'>"), "Unknown instance type: " + str(type(X))
        assert len(X.shape) == 1 or len(X.shape) == 2, "Instance must have 1 or 2 dimensions!"

        # convert dataframes
        if safe_isinstance(X, "pandas.core.series.Series"):
            X = X.values
        elif safe_isinstance(X, "pandas.core.frame.DataFrame"):
            X = X.values
            
            
        phi = np.zeros(X.shape)
        tmp = np.zeros(X.shape)
        for i in range(X.shape[1]):
            tmp[:,i] = X[:,i]
            phi[:,i] = self.f(tmp) - self._zero_offest - self._feature_offset[i]
            tmp[:,i] = 0
            
        return phi
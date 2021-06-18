import numpy as np
import scipy as sp
import warnings
from ._explainer import Explainer
from ..utils import safe_isinstance, MaskedModel
from .. import maskers


class Additive(Explainer):
    """ Computes SHAP values for generalized additive models.

    This assumes that the model only has first order effects. Extending this to
    2nd and third order effects is future work (if you apply this to those models right now
    you will get incorrect answers that fail additivity).
    """

    def __init__(self, model, masker, link=None, feature_names=None):
        """ Build an Additive explainer for the given model using the given masker object.

        Parameters
        ----------
        model : function
            A callable python object that executes the model given a set of input data samples.

        masker : function or numpy.array or pandas.DataFrame
            A callable python object used to "mask" out hidden features of the form `masker(mask, *fargs)`.
            It takes a single a binary mask and an input sample and returns a matrix of masked samples. These
            masked samples are evaluated using the model function and the outputs are then averaged.
            As a shortcut for the standard masking used by SHAP you can pass a background data matrix
            instead of a function and that matrix will be used for masking. To use a clustering
            game structure you can pass a shap.maskers.Tabular(data, hclustering=\"correlation\") object, but
            note that this structure information has no effect on the explanations of additive models.
        """
        super(Additive, self).__init__(model, masker, feature_names=feature_names)

        

        if safe_isinstance(model, "interpret.glassbox.ExplainableBoostingClassifier"):
            self.model = model.decision_function

            if self.masker is None:
                self._expected_value = model.intercept_
                # num_features = len(model.additive_terms_)

                # fm = MaskedModel(self.model, self.masker, self.link, np.zeros(num_features))
                # masks = np.ones((1, num_features), dtype=bool)
                # outputs = fm(masks)
                # self.model(np.zeros(num_features))
                # self._zero_offset = self.model(np.zeros(num_features))#model.intercept_#outputs[0]
                # self._input_offsets = np.zeros(num_features) #* self._zero_offset
                raise Exception("Masker not given and we don't yet support pulling the distribution centering directly from the EBM model!")
                return

        # here we need to compute the offsets ourselves because we can't pull them directly from a model we know about
        assert safe_isinstance(self.masker, "shap.maskers.Independent"), "The Additive explainer only supports the Tabular masker at the moment!"

        # pre-compute per-feature offsets
        fm = MaskedModel(self.model, self.masker, self.link, np.zeros(self.masker.shape[1]))
        masks = np.ones((self.masker.shape[1]+1, self.masker.shape[1]), dtype=bool)
        for i in range(1, self.masker.shape[1]+1):
            masks[i,i-1] = False
        outputs = fm(masks)
        self._zero_offset = outputs[0]
        self._input_offsets = np.zeros(masker.shape[1])
        for i in range(1, self.masker.shape[1]+1):
            self._input_offsets[i-1] = outputs[i] - self._zero_offset

        self._expected_value = self._input_offsets.sum() + self._zero_offset

    def __call__(self, *args, max_evals=None, silent=False):
        """ Explains the output of model(*args), where args represents one or more parallel iteratable args.
        """

        # we entirely rely on the general call implementation, we override just to remove **kwargs
        # from the function signature
        return super(Additive, self).__call__(*args, max_evals=max_evals, silent=silent)

    @staticmethod
    def supports_model_with_masker(model, masker):
        """ Determines if this explainer can handle the given model.

        This is an abstract static method meant to be implemented by each subclass.
        """
        if safe_isinstance(model, "interpret.glassbox.ExplainableBoostingClassifier"):
            if model.interactions is not 0:
                raise Exception("Need to add support for interaction effects!")
            return True
            
        return False

    def explain_row(self, *row_args, max_evals, main_effects, error_bounds, batch_size, outputs, silent):
        """ Explains a single row and returns the tuple (row_values, row_expected_values, row_mask_shapes).
        """

        x = row_args[0]
        inputs = np.zeros((len(x), len(x)))
        for i in range(len(x)):
            inputs[i,i] = x[i]
        
        phi = self.model(inputs) - self._zero_offset - self._input_offsets

        return {
            "values": phi,
            "expected_values": self._expected_value,
            "mask_shapes": [a.shape for a in row_args],
            "main_effects": phi,
            "clustering": getattr(self.masker, "clustering", None)
        }

# class AdditiveExplainer(Explainer):
#     """ Computes SHAP values for generalized additive models.

#     This assumes that the model only has first order effects. Extending this to
#     2nd and third order effects is future work (if you apply this to those models right now
#     you will get incorrect answers that fail additivity).

#     Parameters
#     ----------
#     model : function or ExplainableBoostingRegressor
#         User supplied additive model either as either a function or a model object.

#     data : numpy.array, pandas.DataFrame
#         The background dataset to use for computing conditional expectations.
#     feature_perturbation : "interventional"
#         Only the standard interventional SHAP values are supported by AdditiveExplainer right now.
#     """

#     def __init__(self, model, data, feature_perturbation="interventional"):
#         if feature_perturbation != "interventional":
#             raise Exception("Unsupported type of feature_perturbation provided: " + feature_perturbation)

#         if safe_isinstance(model, "interpret.glassbox.ebm.ebm.ExplainableBoostingRegressor"):
#             self.f = model.predict
#         elif callable(model):
#             self.f = model
#         else:
#             raise ValueError("The passed model must be a recognized object or a function!")
        
#         # convert dataframes
#         if safe_isinstance(data, "pandas.core.series.Series"):
#             data = data.values
#         elif safe_isinstance(data, "pandas.core.frame.DataFrame"):
#             data = data.values
#         self.data = data
        
#         # compute the expected value of the model output
#         self.expected_value = self.f(data).mean()
        
#         # pre-compute per-feature offsets
#         tmp = np.zeros(data.shape)
#         self._zero_offset = self.f(tmp).mean()
#         self._feature_offset = np.zeros(data.shape[1])
#         for i in range(data.shape[1]):
#             tmp[:,i] = data[:,i]
#             self._feature_offset[i] = self.f(tmp).mean() - self._zero_offset
#             tmp[:,i] = 0


#     def shap_values(self, X):
#         """ Estimate the SHAP values for a set of samples.

#         Parameters
#         ----------
#         X : numpy.array, pandas.DataFrame or scipy.csr_matrix
#             A matrix of samples (# samples x # features) on which to explain the model's output.

#         Returns
#         -------
#         For models with a single output this returns a matrix of SHAP values
#         (# samples x # features). Each row sums to the difference between the model output for that
#         sample and the expected value of the model output (which is stored as expected_value
#         attribute of the explainer).
#         """

#         # convert dataframes
#         if str(type(X)).endswith("pandas.core.series.Series'>"):
#             X = X.values
#         elif str(type(X)).endswith("'pandas.core.frame.DataFrame'>"):
#             X = X.values

#         #assert str(type(X)).endswith("'numpy.ndarray'>"), "Unknown instance type: " + str(type(X))
#         assert len(X.shape) == 1 or len(X.shape) == 2, "Instance must have 1 or 2 dimensions!"

#         # convert dataframes
#         if safe_isinstance(X, "pandas.core.series.Series"):
#             X = X.values
#         elif safe_isinstance(X, "pandas.core.frame.DataFrame"):
#             X = X.values
            
            
#         phi = np.zeros(X.shape)
#         tmp = np.zeros(X.shape)
#         for i in range(X.shape[1]):
#             tmp[:,i] = X[:,i]
#             phi[:,i] = self.f(tmp) - self._zero_offset - self._feature_offset[i]
#             tmp[:,i] = 0
            
#         return phi
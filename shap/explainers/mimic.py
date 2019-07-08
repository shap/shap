import numpy as np
import multiprocessing
import sys
from .explainer import Explainer

try:
    import xgboost
except ImportError:
    pass
except:
    print("xgboost is installed...but failed to load!")
    pass

class MimicExplainer(Explainer):
    """Fits a mimic model to the original model and then explains predictions using the mimic model.

    Tree SHAP allows for very fast SHAP value explainations of flexible gradient boosted decision
    tree (GBDT) models. Since GBDT models are so flexible we can train them to mimic any black-box
    model and then using Tree SHAP we can explain them. This won't work well for images, but for
    any type of problem that GBDTs do reasonable well on, they should also be able to learn how to
    explain black-box models on the data. This mimic explainer also allows you to use a linear model,
    but keep in mind that will not do as well at explaining typical non-linear black-box models. In
    the future we could include other mimic model types given enough demand/help. Finally, we would
    like to note that this explainer is vaugely inspired by https://arxiv.org/abs/1802.07814 where
    they learn an explainer that can be applied to any input.


    Parameters
    ----------
    model : function or iml.Model
        User supplied function that takes a matrix of samples (# samples x # features) and
        computes a the output of the model for those samples. The output can be a vector
        (# samples) or a matrix (# samples x # model outputs).

    data : numpy.array or pandas.DataFrame or iml.DenseData
        The background dataset to use for integrating out features. To determine the impact
        of a feature, that feature is set to "missing" and the change in the model output
        is observed. Since most models aren't designed to handle arbitrary missing data at test
        time, we simulate "missing" by replacing the feature with the values it takes in the
        background dataset. So if the background dataset is a simple sample of all zeros, then
        we would approximate a feature being missing by setting it to zero. For small problems
        this background datset can be the whole training set, but for larger problems consider
        using a single reference value or using the kmeans function to summarize the dataset.
    """

    def __init__(self, model, data, mimic_model="xgboost", mimic_model_params={}):
        self.mimic_model_type = mimic_model
        self.mimic_model_params = mimic_model_params

        # convert incoming inputs to standardized iml objects
        self.link = convert_to_link(link)
        self.model = convert_to_model(model)
        self.keep_index = kwargs.pop("keep_index", False)
        self.data = convert_to_data(data, keep_index=self.keep_index)
        match_model_to_data(self.model, self.data)

        if kwargs:
            raise TypeError("Unused keyword arguments: " + ", ".join(kwargs.keys()))

        self.model_out = self.model.f(data.data)

        # enforce our current input type limitations
        assert isinstance(self.data, DenseData), "Shap explainer only supports the DenseData input currently."
        assert not self.data.transposed, "Shap explainer does not support transposed DenseData currently."

        # warn users about large background data sets
        if len(self.data.weights) < 100:
            log.warning("Using only " + str(len(self.data.weights)) + " training data samples could cause " +
                        "the mimic model poorly to fit the real model. Consider using more training samples " +
                        "or if you don't have more samples, using shap.inflate(data, N) to generate more.")

        self._train_mimic_model()

    def _train_mimic_model(self):
        if self.mimic_model_type == "xgboost":
            self.mimic_model = xgboost.train(self.mimic_model_params, xgboost.DMatrix(data.data))

    def shap_values(self, X, **kwargs):
        """ Estimate the SHAP values for a set of samples.

        Parameters
        ----------
        X : numpy.array or pandas.DataFrame
            A matrix of samples (# samples x # features) on which to explain the model's output.

        Returns
        -------
        For a models with a single output this returns a matrix of SHAP values
        (# samples x # features + 1). The last column is the base value of the model, which is
        the expected value of the model applied to the background dataset. This causes each row to
        sum to the model output for that sample. For models with vector outputs this returns a list
        of such matrices, one for each output.
        """

        if kwargs:
            raise TypeError("Unused keyword arguments: " + ", ".join(kwargs.keys()))

        phi = None
        if self.mimic_model_type == "xgboost":
            if not str(type(X)).endswith("xgboost.core.DMatrix'>"):
                X = xgboost.DMatrix(X)
            phi = self.trees.predict(X, pred_contribs=True)

        if phi is not None:
            if len(phi.shape) == 3:
                return [phi[:, i, :] for i in range(phi.shape[1])]
            else:
                return phi

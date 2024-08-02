import warnings

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from tqdm.auto import tqdm

from .. import links, maskers
from ..utils._exceptions import (
    DimensionError,
    InvalidFeaturePerturbationError,
    InvalidModelError,
)
from ._explainer import Explainer


class LinearExplainer(Explainer):
    """Computes SHAP values for a linear model, optionally accounting for inter-feature correlations.

    This computes the SHAP values for a linear model and can account for the
    correlations among the input features. Assuming features are independent
    leads to interventional SHAP values which for a linear model are ``coef[i] *
    (x[i] - X.mean(0)[i])`` for the ith feature. If instead we account for
    correlations, then we prevent any problems arising from collinearity and
    share credit among correlated features. Accounting for correlations can be
    computationally challenging, but ``LinearExplainer`` uses sampling to
    estimate a transform that can then be applied to explain any prediction of
    the model.

    Parameters
    ----------
    model : (coef, intercept) or sklearn.linear_model.*
        User supplied linear model either as either a parameter pair or sklearn object.

    masker : function, numpy.array, pandas.DataFrame, tuple of (mean, cov), shap.maskers.Masker
        A callable Python object used to "mask" out hidden features of the form
        ``masker(binary_mask, x)``. It takes a single input sample and a binary
        mask and returns a matrix of masked samples. These masked samples are
        evaluated using the model function and the outputs are then averaged.

        As a shortcut for the standard masking using by SHAP you can pass a
        background data matrix instead of a function and that matrix will be
        used for masking.

        You can also provide a tuple of ``(mean, covariance)``, or pass in a
        masker meant for tabular data (i.e., :class:`.maskers.Independent`,
        :class:`.maskers.Impute`, or :class:`.maskers.Partition`) directly.

    data : (mean, cov), numpy.array, pandas.DataFrame, iml.DenseData or scipy.csr_matrix
        The background dataset to use for computing conditional expectations.
        Note that only the mean and covariance of the dataset are used. This
        means passing a raw data matrix is just a convenient alternative to
        passing the mean and covariance directly.

    nsamples : int
        Number of samples to use when estimating the transformation matrix used
        to account for feature correlations.

    feature_perturbation : "interventional" (default) or "correlation_dependent"
        There are two ways we might want to compute SHAP values, either the full
        conditional SHAP values or the interventional SHAP values.

        For interventional SHAP values we break any dependence structure between
        features in the model and so uncover how the model would behave if we
        intervened and changed some of the inputs. For the full conditional SHAP
        values we respect the correlations among the input features, so if the
        model depends on one input but that input is correlated with another
        input, then both get some credit for the model's behavior. The
        interventional option stays "true to the model" meaning it will only
        give credit to features that are actually used by the model, while the
        correlation option stays "true to the data" in the sense that it only
        considers how the model would behave when respecting the correlations in
        the input data. For sparse case only interventional option is supported.

        Note that the ``feature_perturbation`` option is deprecated and will be
        removed in a future release. It is recommended to use the appropriate
        tabular ``masker`` instead.

    Examples
    --------
    See `Linear explainer examples <https://shap.readthedocs.io/en/latest/api_examples/explainers/LinearExplainer.html>`_

    """

    def __init__(self, model, masker, link=links.identity, nsamples=1000, feature_perturbation=None, **kwargs):
        if "feature_dependence" in kwargs:
            emsg = "The option feature_dependence has been renamed to feature_perturbation!"
            raise ValueError(emsg)

        if feature_perturbation is not None:  # pragma: no cover
            wmsg = (
                "The feature_perturbation option is now deprecated in favor of using the appropriate "
                "masker (maskers.Independent, maskers.Partition or maskers.Impute)."
            )
            warnings.warn(wmsg, FutureWarning)
        else:
            feature_perturbation = "interventional"
        if feature_perturbation not in ("interventional", "correlation_dependent"):
            emsg = "feature_perturbation must be one of 'interventional' or 'correlation_dependent'"
            raise InvalidFeaturePerturbationError(emsg)
        self.feature_perturbation = feature_perturbation

        # wrap the incoming masker object as a shap.Masker object before calling
        # parent class constructor, which does the same but without respecting
        # the user-provided feature_perturbation choice
        if isinstance(masker, pd.DataFrame) or ((isinstance(masker, np.ndarray) or issparse(masker)) and len(masker.shape) == 2):
            if self.feature_perturbation == "correlation_dependent":
                masker = maskers.Impute(masker)
            else:
                masker = maskers.Independent(masker)
        elif issubclass(type(masker), tuple) and len(masker) == 2:
            if self.feature_perturbation == "correlation_dependent":
                masker = maskers.Impute({"mean": masker[0], "cov": masker[1]}, method="linear")
            else:
                masker = maskers.Independent({"mean": masker[0], "cov": masker[1]})

        super().__init__(model, masker, link=link, **kwargs)

        self.nsamples = nsamples


        # extract what we need from the given model object
        self.coef, self.intercept = LinearExplainer._parse_model(model)

        # extract the data
        if issubclass(type(self.masker), (maskers.Independent, maskers.Partition)):
            self.feature_perturbation = "interventional"
        elif issubclass(type(self.masker), maskers.Impute):
            self.feature_perturbation = "correlation_dependent"
        else:
            raise NotImplementedError("The Linear explainer only supports the Independent, Partition, and Impute maskers right now!")
        data = getattr(self.masker, "data", None)

        # convert DataFrame's to numpy arrays
        if isinstance(data, pd.DataFrame):
            data = data.values

        # get the mean and covariance of the model
        if getattr(self.masker, "mean", None) is not None:
            self.mean = self.masker.mean
            self.cov = self.masker.cov
        elif isinstance(data, dict) and len(data) == 2:
            self.mean = data["mean"]
            if isinstance(self.mean, pd.Series):
                self.mean = self.mean.values

            self.cov = data["cov"]
            if isinstance(self.cov, pd.DataFrame):
                self.cov = self.cov.values
        elif isinstance(data, tuple) and len(data) == 2:
            self.mean = data[0]
            if isinstance(self.mean, pd.Series):
                self.mean = self.mean.values

            self.cov = data[1]
            if isinstance(self.cov, pd.DataFrame):
                self.cov = self.cov.values
        elif data is None:
            raise ValueError("A background data distribution must be provided!")
        else:
            if issparse(data):
                self.mean = np.array(np.mean(data, 0))[0]
                if self.feature_perturbation != "interventional":
                    raise NotImplementedError("Only feature_perturbation = 'interventional' is supported for sparse data")
            else:
                self.mean = np.array(np.mean(data, 0)).flatten() # assumes it is an array
                if self.feature_perturbation == "correlation_dependent":
                    self.cov = np.cov(data, rowvar=False)
        #print(self.coef, self.mean.flatten(), self.intercept)
        # Note: mean can be numpy.matrixlib.defmatrix.matrix or numpy.matrix type depending on numpy version
        if issparse(self.mean) or str(type(self.mean)).endswith("matrix'>"):
            # accept both sparse and dense coef
            # if not issparse(self.coef):
            #     self.coef = np.asmatrix(self.coef)
            self.expected_value = np.dot(self.coef, self.mean) + self.intercept

            # unwrap the matrix form
            if len(self.expected_value) == 1:
                self.expected_value = self.expected_value[0,0]
            else:
                self.expected_value = np.array(self.expected_value)[0]
        else:
            self.expected_value = np.dot(self.coef, self.mean) + self.intercept

        self.M = len(self.mean)

        # if needed, estimate the transform matrices
        if self.feature_perturbation == "correlation_dependent":
            self.valid_inds = np.where(np.diag(self.cov) > 1e-8)[0]
            self.mean = self.mean[self.valid_inds]
            self.cov = self.cov[:,self.valid_inds][self.valid_inds,:]
            self.coef = self.coef[self.valid_inds]

            # group perfectly redundant variables together
            self.avg_proj,sum_proj = duplicate_components(self.cov)
            self.cov = np.matmul(np.matmul(self.avg_proj, self.cov), self.avg_proj.T)
            self.mean = np.matmul(self.avg_proj, self.mean)
            self.coef = np.matmul(sum_proj, self.coef)

            # if we still have some multi-collinearity present then we just add regularization...
            e,_ = np.linalg.eig(self.cov)
            if e.min() < 1e-7:
                self.cov = self.cov + np.eye(self.cov.shape[0]) * 1e-6

            mean_transform, x_transform = self._estimate_transforms(nsamples)
            self.mean_transformed = np.matmul(mean_transform, self.mean)
            self.x_transform = x_transform
        elif self.feature_perturbation == "interventional":
            if nsamples != 1000:
                warnings.warn("Setting nsamples has no effect when feature_perturbation = 'interventional'!")
        else:
            raise InvalidFeaturePerturbationError("Unknown type of feature_perturbation provided: " + self.feature_perturbation)

    def _estimate_transforms(self, nsamples):
        """Uses block matrix inversion identities to quickly estimate transforms.

        After a bit of matrix math we can isolate a transform matrix (# features x # features)
        that is independent of any sample we are explaining. It is the result of averaging over
        all feature permutations, but we just use a fixed number of samples to estimate the value.

        TODO: Do a brute force enumeration when # feature subsets is less than nsamples. This could
              happen through a recursive method that uses the same block matrix inversion as below.
        """
        M = len(self.coef)

        mean_transform = np.zeros((M,M))
        x_transform = np.zeros((M,M))
        inds = np.arange(M, dtype=int)
        for _ in tqdm(range(nsamples), "Estimating transforms"):
            np.random.shuffle(inds)
            cov_inv_SiSi = np.zeros((0,0))
            cov_Si = np.zeros((M,0))
            for j in range(M):
                i = inds[j]

                # use the last Si as the new S
                cov_S = cov_Si
                cov_inv_SS = cov_inv_SiSi

                # get the new cov_Si
                cov_Si = self.cov[:,inds[:j+1]]

                # compute the new cov_inv_SiSi from cov_inv_SS
                d = cov_Si[i,:-1].T
                t = np.matmul(cov_inv_SS, d)
                Z = self.cov[i, i]
                u = Z - np.matmul(t.T, d)
                cov_inv_SiSi = np.zeros((j+1, j+1))
                if j > 0:
                    cov_inv_SiSi[:-1, :-1] = cov_inv_SS + np.outer(t, t) / u
                    cov_inv_SiSi[:-1, -1] = cov_inv_SiSi[-1,:-1] = -t / u
                cov_inv_SiSi[-1, -1] = 1 / u

                # + coef @ (Q(bar(Sui)) - Q(bar(S)))
                mean_transform[i, i] += self.coef[i]

                # + coef @ R(Sui)
                coef_R_Si = np.matmul(self.coef[inds[j+1:]], np.matmul(cov_Si, cov_inv_SiSi)[inds[j+1:]])
                mean_transform[i, inds[:j+1]] += coef_R_Si

                # - coef @ R(S)
                coef_R_S = np.matmul(self.coef[inds[j:]], np.matmul(cov_S, cov_inv_SS)[inds[j:]])
                mean_transform[i, inds[:j]] -= coef_R_S

                # - coef @ (Q(Sui) - Q(S))
                x_transform[i, i] += self.coef[i]

                # + coef @ R(Sui)
                x_transform[i, inds[:j+1]] += coef_R_Si

                # - coef @ R(S)
                x_transform[i, inds[:j]] -= coef_R_S

        mean_transform /= nsamples
        x_transform /= nsamples
        return mean_transform, x_transform

    @staticmethod
    def _parse_model(model):
        """Attempt to pull out the coefficients and intercept from the given model object."""
        # raw coefficients
        if type(model) == tuple and len(model) == 2:
            coef = model[0]
            intercept = model[1]

        # sklearn style model
        elif hasattr(model, "coef_") and hasattr(model, "intercept_"):
            # work around for multi-class with a single class
            if len(model.coef_.shape) > 1 and model.coef_.shape[0] == 1:
                coef = model.coef_[0]
                try:
                    intercept = model.intercept_[0]
                except TypeError:
                    intercept = model.intercept_
            else:
                coef = model.coef_
                intercept = model.intercept_
        else:
            raise InvalidModelError("An unknown model type was passed: " + str(type(model)))

        return coef,intercept

    @staticmethod
    def supports_model_with_masker(model, masker):
        """Determines if we can parse the given model."""
        if not isinstance(masker, (maskers.Independent, maskers.Partition, maskers.Impute)):
            return False

        try:
            LinearExplainer._parse_model(model)
        except Exception:
            return False
        return True

    def explain_row(self, *row_args, max_evals, main_effects, error_bounds, batch_size, outputs, silent):
        """Explains a single row and returns the tuple (row_values, row_expected_values, row_mask_shapes)."""
        assert len(row_args) == 1, "Only single-argument functions are supported by the Linear explainer!"

        X = row_args[0]
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        # convert dataframes
        if isinstance(X, (pd.Series, pd.DataFrame)):
            X = X.values

        if len(X.shape) not in (1, 2):
            raise DimensionError(f"Instance must have 1 or 2 dimensions! Not: {len(X.shape)}")

        if self.feature_perturbation == "correlation_dependent":
            if issparse(X):
                raise InvalidFeaturePerturbationError("Only feature_perturbation = 'interventional' is supported for sparse data")
            phi = np.matmul(np.matmul(X[:,self.valid_inds], self.avg_proj.T), self.x_transform.T) - self.mean_transformed
            phi = np.matmul(phi, self.avg_proj)

            full_phi = np.zeros((phi.shape[0], self.M))
            full_phi[:,self.valid_inds] = phi
            phi = full_phi

        elif self.feature_perturbation == "interventional":
            if issparse(X):
                phi = np.array(np.multiply(X - self.mean, self.coef))

                # if len(self.coef.shape) == 1:
                #     return np.array(np.multiply(X - self.mean, self.coef))
                # else:
                #     return [np.array(np.multiply(X - self.mean, self.coef[i])) for i in range(self.coef.shape[0])]
            else:
                phi = np.array(X - self.mean) * self.coef
                # if len(self.coef.shape) == 1:
                #     phi = np.array(X - self.mean) * self.coef
                #     return np.array(X - self.mean) * self.coef
                # else:
                #     return [np.array(X - self.mean) * self.coef[i] for i in range(self.coef.shape[0])]

        return {
            "values": phi.T,
            "expected_values": self.expected_value,
            "mask_shapes": (X.shape[1:],),
            "main_effects": phi.T,
            "clustering": None
        }


    def shap_values(self, X):
        """Estimate the SHAP values for a set of samples.

        Parameters
        ----------
        X : numpy.array, pandas.DataFrame or scipy.csr_matrix
            A matrix of samples (# samples x # features) on which to explain the model's output.

        Returns
        -------
        np.array
            Estimated SHAP values, usually of shape ``(# samples x # features)``.

            Each row sums to the difference between the model output for that
            sample and the expected value of the model output (which is stored
            as the ``expected_value`` attribute of the explainer).

            The shape of the returned array depends on the number of model outputs:

            * one output: array of shape ``(#num_samples, *X.shape[1:])``.
            * multiple outputs: array of shape ``(#num_samples, *X.shape[1:],
              #num_outputs)``.

            .. versionchanged:: 0.45.0
                Return type for models with multiple outputs changed from list to np.ndarray.

        """
        # convert dataframes
        if isinstance(X, (pd.Series, pd.DataFrame)):
            X = X.values

        # assert isinstance(X, np.ndarray), "Unknown instance type: " + str(type(X))
        if len(X.shape) not in (1, 2):
            raise DimensionError(f"Instance must have 1 or 2 dimensions! Not: {len(X.shape)}")

        if self.feature_perturbation == "correlation_dependent":
            if issparse(X):
                raise InvalidFeaturePerturbationError("Only feature_perturbation = 'interventional' is supported for sparse data")
            phi = np.matmul(np.matmul(X[:,self.valid_inds], self.avg_proj.T), self.x_transform.T) - self.mean_transformed
            phi = np.matmul(phi, self.avg_proj)

            full_phi = np.zeros((phi.shape[0], self.M))
            full_phi[:,self.valid_inds] = phi

            return full_phi

        elif self.feature_perturbation == "interventional":
            if issparse(X):
                if len(self.coef.shape) == 1:
                    return np.array(np.multiply(X - self.mean, self.coef))
                else:
                    return np.stack([np.array(np.multiply(X - self.mean, self.coef[i])) for i in range(self.coef.shape[0])], axis=-1)
            else:
                if len(self.coef.shape) == 1:
                    return np.array(X - self.mean) * self.coef
                else:
                    return np.stack([np.array(X - self.mean) * self.coef[i] for i in range(self.coef.shape[0])], axis=-1)

def duplicate_components(C):
    D = np.diag(1/np.sqrt(np.diag(C)))
    C = np.matmul(np.matmul(D, C), D)
    components = -np.ones(C.shape[0], dtype=int)
    count = -1
    for i in range(C.shape[0]):
        found_group = False
        for j in range(C.shape[0]):
            if components[j] < 0 and np.abs(2*C[i,j] - C[i,i] - C[j,j]) < 1e-8:
                if not found_group:
                    count += 1
                    found_group = True
                components[j] = count

    proj = np.zeros((len(np.unique(components)), C.shape[0]))
    proj[0, 0] = 1
    for i in range(1,C.shape[0]):
        proj[components[i], i] = 1
    return (proj.T / proj.sum(1)).T, proj

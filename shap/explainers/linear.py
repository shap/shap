import numpy as np
import scipy as sp
import warnings
from tqdm.autonotebook import tqdm
from .explainer import Explainer

class LinearExplainer(Explainer):
    """ Computes SHAP values for a linear model, optionally accounting for inter-feature correlations.

    This computes the SHAP values for a linear model and can account for the correlations among
    the input features. Assuming features are independent leads to interventional SHAP values which
    for a linear model are coef[i] * (x[i] - X.mean(0)[i]) for the ith feature. If instead we account
    for correlations then we prevent any problems arising from colinearity and share credit among
    correlated features. Accounting for correlations can be computationally challenging, but
    LinearExplainer uses sampling to estimate a transform that can then be applied to explain
    any prediction of the model.

    Parameters
    ----------
    model : (coef, intercept) or sklearn.linear_model.*
        User supplied linear model either as either a parameter pair or sklearn object.

    data : (mean, cov), numpy.array, pandas.DataFrame, iml.DenseData or scipy.csr_matrix
        The background dataset to use for computing conditional expectations. Note that only the
        mean and covariance of the dataset are used. This means passing a raw data matrix is just
        a convienent alternative to passing the mean and covariance directly.
    nsamples : int
        Number of samples to use when estimating the transformation matrix used to account for
        feature correlations.
    feature_dependence : "independent" (default) or "correlation"
        There are two ways we might want to compute SHAP values, either the full conditional SHAP
        values or the independent SHAP values. For independent SHAP values we break any
        dependence structure between features in the model and so uncover how the model would behave if we
        intervened and changed some of the inputs. For the full conditional SHAP values we respect
        the correlations among the input features, so if the model depends on one input but that
        input is correlated with another input, then both get some credit for the model's behavior. The
        independent option stays "true to the model" meaning it will only give credit to features that are
        actually used by the model, while the correlation option stays "true to the data" in the sense that
        it only considers how the model would behave when respecting the correlations in the input data.
        For sparse case only independent option is supported.
    """

    def __init__(self, model, data, nsamples=1000, feature_dependence=None):
        self.nsamples = nsamples
        if feature_dependence == "interventional":
            warnings.warn('The option feature_dependence="interventional" is has been renamed to feature_dependence="independent"!')
            feature_dependence = "independent"
        elif feature_dependence is None:
            warnings.warn('The default value for feature_dependence has been changed to "independent"!')
            feature_dependence = "independent"
        self.feature_dependence = feature_dependence

        # raw coefficents
        if type(model) == tuple and len(model) == 2:
            self.coef = model[0]
            self.intercept = model[1]

        # sklearn style model
        elif hasattr(model, "coef_") and hasattr(model, "intercept_"):
            self.coef = model.coef_
            self.intercept = model.intercept_
        else:
            raise Exception("An unknown model type was passed: " + str(type(model)))

        # convert DataFrame's to numpy arrays
        if str(type(data)).endswith("'pandas.core.frame.DataFrame'>"):
            data = data.values

        # get the mean and covariance of the model
        if type(data) == tuple and len(data) == 2:
            self.mean = data[0]
            self.cov = data[1]
        elif data is None:
            raise Exception("A background data distribution must be provided!")
        else:
            if sp.sparse.issparse(data):
                self.mean = np.array(np.mean(data, 0))[0]
                if feature_dependence != "independent":
                    raise Exception("Only feature_dependence = 'independent' is supported for sparse data")
            else:
                self.mean = np.array(np.mean(data, 0)).flatten() # assumes it is an array
                if feature_dependence == "correlation":
                    self.cov = np.cov(data, rowvar=False)
        # Note: mean can be numpy.matrixlib.defmatrix.matrix or numpy.matrix type depending on numpy version
        if sp.sparse.issparse(self.mean) or str(type(self.mean)).endswith("matrix'>"):
            # accept both sparse and dense coef
            # if not sp.sparse.issparse(self.coef):
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
        if feature_dependence == "correlation":
            self.valid_inds = np.where(np.diag(self.cov) > 1e-8)[0]
            self.mean = self.mean[self.valid_inds]
            self.cov = self.cov[:,self.valid_inds][self.valid_inds,:]
            self.coef = self.coef[self.valid_inds]

            # group perfectly redundant variables together
            self.avg_proj,sum_proj = duplicate_components(self.cov)
            self.cov = np.matmul(np.matmul(self.avg_proj, self.cov), self.avg_proj.T)
            self.mean = np.matmul(self.avg_proj, self.mean)
            self.coef = np.matmul(sum_proj, self.coef)

            # if we still have some multi-colinearity present then we just add regularization...
            e,_ = np.linalg.eig(self.cov)
            if e.min() < 1e-7:
                self.cov = self.cov + np.eye(self.cov.shape[0]) * 1e-6

            mean_transform, x_transform = self._estimate_transforms(nsamples)
            self.mean_transformed = np.matmul(mean_transform, self.mean)
            self.x_transform = x_transform
        elif feature_dependence == "independent":
            if nsamples != 1000:
                warnings.warn("Setting nsamples has no effect when feature_dependence = 'independent'!")
        else:
            raise Exception("Unknown type of feature_dependence provided: " + feature_dependence)

    def _estimate_transforms(self, nsamples):
        """ Uses block matrix inversion identities to quickly estimate transforms.

        After a bit of matrix math we can isolate a transform matrix (# features x # features)
        that is independent of any sample we are explaining. It is the result of averaging over
        all feature permutations, but we just use a fixed number of samples to estimate the value.

        TODO: Do a brute force enumeration when # feature subsets is less than nsamples. This could
              happen through a recursive method that uses the same block matrix inversion as below.
        """
        M = len(self.coef)

        mean_transform = np.zeros((M,M))
        x_transform = np.zeros((M,M))
        inds = np.arange(M, dtype=np.int)
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

        if self.feature_dependence == "correlation":
            if sp.sparse.issparse(X):
                raise Exception("Only feature_dependence = 'independent' is supported for sparse data")
            phi = np.matmul(np.matmul(X[:,self.valid_inds], self.avg_proj.T), self.x_transform.T) - self.mean_transformed
            phi = np.matmul(phi, self.avg_proj)

            full_phi = np.zeros(((phi.shape[0], self.M)))
            full_phi[:,self.valid_inds] = phi

            return full_phi

        elif self.feature_dependence == "independent":
            if sp.sparse.issparse(X):
                if len(self.coef.shape) == 1:
                    return np.array(np.multiply(X - self.mean, self.coef))
                else:
                    return [np.array(np.multiply(X - self.mean, self.coef[i])) for i in range(self.coef.shape[0])]
            else:
                if len(self.coef.shape) == 1:
                    return np.array(X - self.mean) * self.coef
                else:
                    return [np.array(X - self.mean) * self.coef[i] for i in range(self.coef.shape[0])]

def duplicate_components(C):
    D = np.diag(1/np.sqrt(np.diag(C)))
    C = np.matmul(np.matmul(D, C), D)
    components = -np.ones(C.shape[0], dtype=np.int)
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

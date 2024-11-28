import logging

import numpy as np
import pandas as pd
from numba import njit

from .. import utils
from .._serializable import Deserializer, Serializer
from ..utils import MaskedModel
from ..utils._exceptions import DimensionError, InvalidClusteringError
from ._masker import Masker

log = logging.getLogger("shap")


class Tabular(Masker):
    """A common base class for tabular data maskers, such as Independent and Partition."""

    def __init__(self, data, max_samples=100, clustering=None):
        """This masks out tabular features by integrating over the given background dataset.

        Parameters
        ----------
        data : np.array, pandas.DataFrame
            The background dataset that is used for masking.

        max_samples : int
            The maximum number of samples to use from the passed background data. If data has more
            than max_samples then shap.utils.sample is used to subsample the dataset. The number of
            samples coming out of the masker (to be integrated over) matches the number of samples in
            the background dataset. This means larger background dataset cause longer runtimes. Normally
            about 1, 10, 100, or 1000 background samples are reasonable choices.

        clustering : string or None (default) or numpy.ndarray
            The distance metric to use for creating the clustering of the features. The
            distance function can be any valid scipy.spatial.distance.pdist's metric argument.
            However we suggest using 'correlation' in most cases. The full list of options is
            `braycurtis`, `canberra`, `chebyshev`, `cityblock`, `correlation`, `cosine`, `dice`,
            `euclidean`, `hamming`, `jaccard`, `jensenshannon`, `kulsinski`, `mahalanobis`,
            `matching`, `minkowski`, `rogerstanimoto`, `russellrao`, `seuclidean`,
            `sokalmichener`, `sokalsneath`, `sqeuclidean`, `yule`. These are all
            the options from scipy.spatial.distance.pdist's metric argument.

        """
        self.output_dataframe = False
        if isinstance(data, pd.DataFrame):
            self.feature_names = data.columns
            data = data.values
            self.output_dataframe = True

        if isinstance(data, dict) and "mean" in data:
            self.mean = data.get("mean", None)
            self.cov = data.get("cov", None)
            data = np.expand_dims(data["mean"], 0)

        if hasattr(data, "shape") and data.shape[0] > max_samples:
            data = utils.sample(data, max_samples)

        self.data = data
        self.clustering = clustering
        self.max_samples = max_samples

        # # warn users about large background data sets
        # if self.data.shape[0] > 100:
        #     log.warning("Using " + str(self.data.shape[0]) + " background data samples could cause slower " +
        #                 "run times. Consider shap.utils.sample(data, K) to summarize the background using only K samples.")

        # compute the clustering of the data
        if clustering is not None:
            if isinstance(clustering, str):
                self.clustering = utils.hclust(data, metric=clustering)
            elif isinstance(clustering, np.ndarray):
                self.clustering = clustering
            else:
                raise InvalidClusteringError(
                    "Unknown clustering given! Make sure you pass a distance metric as a string, or a clustering as a numpy.ndarray."
                )
        else:
            self.clustering = None

        # self._last_mask = np.zeros(self.data.shape[1], dtype=bool)
        self._masked_data = data.copy()
        self._last_mask = np.zeros(data.shape[1], dtype=bool)
        self.shape = self.data.shape
        self.supports_delta_masking = True
        # self._last_x = None
        # self._data_variance = np.ones(self.data.shape, dtype=bool)

        # this is property that allows callers to check what rows actually changed since last time.
        # self.changed_rows = np.ones(self.data.shape[0], dtype=bool)

    def __call__(self, mask, x):
        mask = self._standardize_mask(mask, x)

        # make sure we are given a single sample
        if len(x.shape) != 1 or x.shape[0] != self.data.shape[1]:
            raise DimensionError("The input passed for tabular masking does not match the background data shape!")

        # if mask is an array of integers then we are doing delta masking
        if np.issubdtype(mask.dtype, np.integer):
            variants = ~self.invariants(x)
            curr_delta_inds = np.zeros(len(mask), dtype=int)
            num_masks = (mask >= 0).sum()
            varying_rows_out = np.zeros((num_masks, self.shape[0]), dtype=bool)
            masked_inputs_out = np.zeros((num_masks * self.shape[0], self.shape[1]))
            self._last_mask[:] = False
            self._masked_data[:] = self.data
            _delta_masking(
                mask,
                x,
                curr_delta_inds,
                varying_rows_out,
                self._masked_data,
                self._last_mask,
                self.data,
                variants,
                masked_inputs_out,
                MaskedModel.delta_mask_noop_value,
            )
            if self.output_dataframe:
                return (pd.DataFrame(masked_inputs_out, columns=self.feature_names),), varying_rows_out

            return (masked_inputs_out,), varying_rows_out

        # otherwise we update the whole set of masked data for a single sample
        self._masked_data[:] = x * mask + self.data * np.invert(mask)
        self._last_mask[:] = mask

        if self.output_dataframe:
            return pd.DataFrame(self._masked_data, columns=self.feature_names)

        return (self._masked_data,)

    # def reset_delta_masking(self):
    #     """ This resets the masker back to all zeros when delta masking.

    #     Note that the presence of this function also denotes that we support delta masking.
    #     """
    #     self._masked_data[:] = self.data
    #     self._last_mask[:] = False

    def invariants(self, x):
        """This returns a mask of which features change when we mask them.

        This optional masking method allows explainers to avoid re-evaluating the model when
        the features that would have been masked are all invariant.
        """
        # make sure we got valid data
        if x.shape != self.data.shape[1:]:
            raise DimensionError(
                "The passed data does not match the background shape expected by the masker! The data of shape "
                + str(x.shape)
                + " was passed while the masker expected data of shape "
                + str(self.data.shape[1:])
                + "."
            )

        return np.isclose(x, self.data)

    def save(self, out_file):
        """Write a Tabular masker to a file stream."""
        super().save(out_file)

        # Increment the version number when the encoding changes!
        with Serializer(out_file, "shap.maskers.Tabular", version=0) as s:
            # save the data in the format it was given to us
            if self.output_dataframe:
                s.save("data", pd.DataFrame(self.data, columns=self.feature_names))
            elif getattr(self, "mean", None) is not None:
                s.save("data", (self.mean, self.cov))
            else:
                s.save("data", self.data)

            s.save("max_samples", self.max_samples)
            s.save("clustering", self.clustering)

    @classmethod
    def load(cls, in_file, instantiate=True):
        """Load a Tabular masker from a file stream."""
        if instantiate:
            return cls._instantiated_load(in_file)

        kwargs = super().load(in_file, instantiate=False)
        with Deserializer(in_file, "shap.maskers.Tabular", min_version=0, max_version=0) as s:
            kwargs["data"] = s.load("data")
            kwargs["max_samples"] = s.load("max_samples")
            kwargs["clustering"] = s.load("clustering")
        return kwargs


@njit
def _single_delta_mask(dind, masked_inputs, last_mask, data, x, noop_code):
    if dind == noop_code:
        pass
    elif last_mask[dind]:
        masked_inputs[:, dind] = data[:, dind]
        last_mask[dind] = False
    else:
        masked_inputs[:, dind] = x[dind]
        last_mask[dind] = True


@njit
def _delta_masking(
    masks,
    x,
    curr_delta_inds,
    varying_rows_out,
    masked_inputs_tmp,
    last_mask,
    data,
    variants,
    masked_inputs_out,
    noop_code,
):
    """Implements the special (high speed) delta masking API that only flips the positions we need to.

    Note that we attempt to avoid doing any allocation inside this function for speed reasons.
    """
    dpos = 0
    i = -1
    masks_pos = 0
    output_pos = 0
    N = masked_inputs_tmp.shape[0]
    while masks_pos < len(masks):
        i += 1

        # update the tmp masked inputs array
        dpos = 0
        curr_delta_inds[0] = masks[masks_pos]
        while curr_delta_inds[dpos] < 0:  # negative values mean keep going
            curr_delta_inds[dpos] = -curr_delta_inds[dpos] - 1  # -value + 1 is the original index that needs flipped
            _single_delta_mask(curr_delta_inds[dpos], masked_inputs_tmp, last_mask, data, x, noop_code)
            dpos += 1
            curr_delta_inds[dpos] = masks[masks_pos + dpos]
        _single_delta_mask(curr_delta_inds[dpos], masked_inputs_tmp, last_mask, data, x, noop_code)

        # copy the tmp masked inputs array to the output
        masked_inputs_out[output_pos : output_pos + N] = masked_inputs_tmp
        masks_pos += dpos + 1

        # mark which rows have been updated, so we can only evaluate the model on the rows we need to
        if i == 0:
            varying_rows_out[i, :] = True

        else:
            # only one column was changed
            if dpos == 0:
                varying_rows_out[i, :] = variants[:, curr_delta_inds[dpos]]

            # more than one column was changed
            else:
                varying_rows_out[i, :] = np.sum(variants[:, curr_delta_inds[: dpos + 1]], axis=1) > 0

        output_pos += N


class Independent(Tabular):
    """This masks out tabular features by integrating over the given background dataset."""

    def __init__(self, data, max_samples=100):
        """Build a Independent masker with the given background data.

        Parameters
        ----------
        data : numpy.ndarray, pandas.DataFrame
            The background dataset that is used for masking.

        max_samples : int
            The maximum number of samples to use from the passed background data. If data has more
            than max_samples then shap.utils.sample is used to subsample the dataset. The number of
            samples coming out of the masker (to be integrated over) matches the number of samples in
            the background dataset. This means larger background dataset cause longer runtimes. Normally
            about 1, 10, 100, or 1000 background samples are reasonable choices.

        """
        super().__init__(data, max_samples=max_samples, clustering=None)


class Partition(Tabular):
    """This masks out tabular features by integrating over the given background dataset.

    Unlike Independent, Partition respects a hierarchical structure of the data.
    """

    def __init__(self, data, max_samples=100, clustering="correlation"):
        """Build a Partition masker with the given background data and clustering.

        Parameters
        ----------
        data : numpy.ndarray, pandas.DataFrame
            The background dataset that is used for masking.

        max_samples : int
            The maximum number of samples to use from the passed background data. If data has more
            than max_samples then shap.utils.sample is used to subsample the dataset. The number of
            samples coming out of the masker (to be integrated over) matches the number of samples in
            the background dataset. This means larger background dataset cause longer runtimes. Normally
            about 1, 10, 100, or 1000 background samples are reasonable choices.

        clustering : string or numpy.ndarray
            If a string, then this is the distance metric to use for creating the clustering of
            the features. The distance function can be any valid scipy.spatial.distance.pdist's metric
            argument. However we suggest using 'correlation' in most cases. The full list of options is
            `braycurtis`, `canberra`, `chebyshev`, `cityblock`, `correlation`, `cosine`, `dice`,
            `euclidean`, `hamming`, `jaccard`, `jensenshannon`, `kulsinski`, `mahalanobis`,
            `matching`, `minkowski`, `rogerstanimoto`, `russellrao`, `seuclidean`,
            `sokalmichener`, `sokalsneath`, `sqeuclidean`, `yule`. These are all
            the options from scipy.spatial.distance.pdist's metric argument.
            If an array, then this is assumed to be the clustering of the features.

        """
        super().__init__(data, max_samples=max_samples, clustering=clustering)


class Impute(Masker):  # we should inherit from Tabular once we add support for arbitrary masking
    """This imputes the values of missing features using the values of the observed features.

    Unlike Independent, Gaussian imputes missing values based on correlations with observed data points.
    """

    def __init__(self, data, method="linear"):
        """Build a Partition masker with the given background data and clustering.

        Parameters
        ----------
        data : numpy.ndarray, pandas.DataFrame or {"mean: numpy.ndarray, "cov": numpy.ndarray} dictionary
            The background dataset that is used for masking.

        """
        if data is dict and "mean" in data:
            self.mean = data.get("mean", None)
            self.cov = data.get("cov", None)
            data = np.expand_dims(data["mean"], 0)

        self.data = data
        self.method = method


class Causal(Tabular):
    """This masks out tabular features by integrating over the given background dataset.

    The Causal Masker samples from the interventional distribution, which allows for computing causal Shapley values. As proposed in [1]
    """

    def __init__(self, data, ordering=None, confounding=None, max_samples=100):
        """Build a Causal Masker with the given background data and causal ordering.

        Parameters
        ----------
        data : np.array, pandas.DataFrame
            The background dataset that is used for masking.

        ordering : numpy.ndarray, pandas.DataFrame
            The (partial) causal ordering of the features in the form of a partial chain graph.
            Example: [[1, 2], [3], [4, 5, 6]]

        confounding : list, numpy.array
            A 1-dimensional boolean array that specifies which causal groups contain confounding factors.
            Example: [True, False, False]

        max_samples : int
            The maximum number of samples to use from the passed background data. If data has more
            than max_samples then shap.utils.sample is used to subsample the dataset. The number of
            samples coming out of the masker (to be integrated over) matches the number of samples in
            the background dataset. This means larger background dataset cause longer runtimes. Normally
            about 1, 10, 100, or 1000 background samples are reasonable choices.
        """
        super().__init__(data, max_samples=max_samples, clustering=None)

        self.n_features = data.shape[0]
        self.n_samples = max_samples

        if ordering is None:
            self.ordering = list(range(self.n_features))
            log.warning(
                "No causal ordering provided. Consider using the Independent Masker if your goal is to compute marginal Shapley values."
            )
        else:
            # TODO: Enable features names

            # if ordering.shape == None:
            #     raise DimensionError('Ordering has to be a 2d list.')

            n_features_in_ordering = sum(len(group) for group in ordering)
            if n_features_in_ordering != self.n_features:
                raise DimensionError(
                    f"{n_features_in_ordering} features were provided in the ordering, while {self.n_features} were expected."
                )

            # if pass:
            #     Exception("Feature {} occurs multiple times in the provided ordering.")

            if max(*ordering) > self.n_features or min(*ordering) < 0:
                raise Exception()

            self.ordering = ordering

        if confounding is None:
            self.confounding = self.n_features * [False]
            log.warning("No confounding provided. Assuming there are no confounders present.")
        else:
            confounding = confounding.to_numpy()
            if confounding.shape != ordering.shape[0]:
                raise DimensionError(
                    f"Provided confounding shape is {confounding.shape}, which does not match the number of causal groups {len(self.ordering)}. Please specify confounding for each group."
                )

            if confounding.dtype != bool:
                raise TypeError("Confounding has to be a boolean array.")

            self.confounding = confounding

        self.mean = data.mean().values
        self.covariance_matrix = data.cov().to_numpy()

        # Ensure that the covariance matrix is positive-definite, as required by numpy multivariate_normal
        if self._is_positive_definitive(self.covariance_matrix.values):
            raise Exception("Covariance matrix is not positive-definite, not yet implemented")

    def __call__(self, mask, x):
        # Standardize
        mask = self._standardize_mask(mask, x)
        x = x.to_numpy()

        # make sure we are given a single sample
        if len(x.shape) != 1 or x.shape[0] != self.data.shape[1]:
            raise DimensionError("The input passed for tabular masking does not match the background data shape!")

        assert mask.shape == x.shape[0], "mask must have the same shape as features"
        assert mask.dtype == bool, "mask must be of Boolean dtype"

        # TODO What do delta masks look like, can we support, or at least map them here?

        # Identify indices of variables not intervened upon (dependent variables)
        out_of_coalition_indices = np.setdiff1d(np.arange(self.n_features), mask)

        # Initialize array to store sampled data and fix values for intervened variables (do-operator)
        samples = np.empty([self.n_samples, self.n_features])
        samples[:, mask] = np.repeat(x[mask], self.n_samples, axis=0)

        n_causal_groups = len(self.ordering)
        for group_idx in range(n_causal_groups):
            # Identify features to sample in the current causal component
            to_be_sampled = np.intersect1d(self.ordering[group_idx], out_of_coalition_indices)
            if len(to_be_sampled) == 0:
                continue

            # Condition upon all ancestors in causal ordering
            to_be_conditioned = np.concatenate(self.ordering[:group_idx])
            if not self.confounding[group_idx]:
                # also condition on the in-coalition features from the current causal group
                in_coalition = np.intersect1d(self.ordering[group_idx], mask)  # does this work?
                to_be_conditioned = np.union1d(to_be_conditioned, in_coalition).astype(int)

            # Retrieve samples for the features not intervened upon (apply DO-operator)
            new_samples = self._sample_gaussian(samples, to_be_conditioned, to_be_sampled)
            samples[:, to_be_sampled] = new_samples

        self._masked_data = samples
        self._last_mask[:] = mask

        if self.output_dataframe:
            return pd.DataFrame(self._masked_data, columns=self.feature_names)

        return (self._masked_data,)

    def _sample_gaussian(self, samples, to_be_conditioned, to_be_sampled):
        # Case 1: No conditioning required (sample marginal distribution)
        if len(to_be_conditioned) == 0:
            new_samples = np.random.multivariate_normal(
                mean=self.mean[to_be_sampled],
                cov=self.covariance_matrix[np.ix_(to_be_sampled, to_be_sampled)],
                size=self.n_samples,
            )

        # Case 2: Conditional sampling (Gaussian conditional distribution)
        else:
            # Compute covariance components for conditional Gaussian sampling
            c = self.covariance_matrix[np.ix_(to_be_sampled, to_be_conditioned)]
            d = self.covariance_matrix[np.ix_(to_be_conditioned, to_be_conditioned)]
            cd_inv = c.dot(np.linalg.inv(d))
            conditional_covariance = self.covariance_matrix[np.ix_(to_be_sampled, to_be_sampled)] - cd_inv.dot(c.T)

            # Ensure covariance matrix symmetry, as required by numpy multivariate_normal
            if not self._is_symmetric(conditional_covariance):
                conditional_covariance = self._symmetric_part(conditional_covariance)

            # Compute conditional mean
            to_sample_means = np.repeat(np.array([self.mean[to_be_sampled]]), self.n_samples, axis=0)
            to_condition_means = np.repeat(np.array([self.mean[to_be_conditioned]]), self.n_samples, axis=0)
            conditional_mean = to_sample_means + cd_inv.dot((samples[:, to_be_conditioned] - to_condition_means).T).T

            # Sample
            new_samples = np.random.multivariate_normal(
                mean=np.zeros(len(to_be_sampled)), cov=conditional_covariance, size=self.n_samples
            )
            new_samples += conditional_mean

        return new_samples

    @staticmethod
    def _is_positive_definitive(matrix):
        # Checks if covariance matrix is positive-definite, which is required for numpy multivariate_normal
        eigen_values = np.linalg.eigvals(matrix)
        return np.any(eigen_values > 1e-06)

    @staticmethod
    def _is_symmetric(matrix, rtol=1e-05, atol=1e-08):
        # Checks if covariance matrix is positive-definite, which is required for numpy multivariate_normal
        return np.allclose(matrix, matrix.T, rtol=rtol, atol=atol)

    @staticmethod
    def _symmetric_part(matrix):
        # Retrieve the symmetric part from a covariance matrix
        return (matrix + matrix.T) / 2

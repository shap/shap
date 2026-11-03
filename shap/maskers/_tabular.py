import logging

import numpy as np
import pandas as pd

from .. import utils

# nanobind's stub generator skips underscore-prefixed names, so _cutils.pyi has
# no declaration for this one even though the binding exists.
from .._cutils import _delta_masking  # type: ignore[attr-defined]
from .._serializable import Deserializer, Serializer
from ..utils import MaskedModel
from ..utils._exceptions import DimensionError, InvalidClusteringError
from ._masker import Masker

log = logging.getLogger("shap")


class Tabular(Masker):
    """Common base class for tabular maskers. Private class, do not use directly.

    Parent of Independent and Partition maskers.

    This masks out tabular features by integrating over the given background dataset.

    Parameters
    ----------
    data : 2D np.ndarray, or pandas.DataFrame
        The background dataset that is used for masking.

    max_samples : int, optional, default=100
        The maximum number of samples to use from the passed background data. If data has more
        than max_samples then shap.utils.sample is used to subsample the dataset. The number of
        samples coming out of the masker (to be integrated over) matches the number of samples in
        the background dataset. This means larger background dataset cause longer runtimes. Normally
        about 1, 10, 100, or 1000 background samples are reasonable choices.

    clustering : string or None (default) or numpy.ndarray, optional
        The distance metric to use for creating the clustering of the features. The
        distance function can be any valid scipy.spatial.distance.pdist's metric argument.
        However we suggest using 'correlation' in most cases. The full list of options is
        `braycurtis`, `canberra`, `chebyshev`, `cityblock`, `correlation`, `cosine`, `dice`,
        `euclidean`, `hamming`, `jaccard`, `jensenshannon`, `kulsinski`, `mahalanobis`,
        `matching`, `minkowski`, `rogerstanimoto`, `russellrao`, `seuclidean`,
        `sokalmichener`, `sokalsneath`, `sqeuclidean`, `yule`. These are all
        the options from scipy.spatial.distance.pdist's metric argument.
    """

    def __init__(self, data, max_samples=100, clustering=None):
        self.max_samples = max_samples

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
            log.warning(
                "Background dataset has %d samples but max_samples=%d. "
                "Subsampling to %d samples for SHAP value computation. "
                "To use all samples, set max_samples=%d when initializing the masker.",
                data.shape[0],
                max_samples,
                max_samples,
                data.shape[0],
            )
            data = utils.sample(data, max_samples)

        self.data = data

        # compute the clustering of the data
        if isinstance(clustering, str):
            self.clustering = utils.hclust(data, metric=clustering)
        elif isinstance(clustering, np.ndarray) or clustering is None:
            self.clustering = clustering
        else:
            raise InvalidClusteringError(
                "Unknown clustering given! Make sure you pass a distance metric as a "
                "string, or a clustering as a numpy.ndarray."
            )

        # this attribute does nothing, but present for backwards compatibility
        self.partition = None

        self._masked_data = data.copy()
        self._last_mask = np.zeros(data.shape[1], dtype=bool)
        self.shape = self.data.shape
        self.supports_delta_masking = True

    def __call__(self, mask, x):
        """Masker calls accept the same inputs as the model plus a binary mask.

        Parameters
        ----------
        mask : bool or np.ndarray
            A binary mask indicating which features to keep (True) and
            which to mask (False). The length of the mask should match the number of
            features in the input data, usually `x` in `MyMasker__call__`.
            Masked features will be replaced with a background value upon `__call__`.
            A common such background value is the mean of the feature across
            the training data.

        x : 2D np.ndarray
            Data to be masked, present as explicit kwarg in most child classes.
            Usually contains a 2D dataframe-like value representing the input data,
            called `x`, usually `np.ndarray`.
            The columns of `x` correspond to the indices in `mask`.
        """
        mask = self._standardize_mask(mask, x)

        # make sure we are given a single sample
        if len(x.shape) != 1 or x.shape[0] != self.data.shape[1]:
            raise DimensionError(
                "The input passed for tabular masking does not match the "
                f"background data shape! Input shape: {str(x.shape)} ; expected "
                f"data shape {str(self.data.shape[1:])}."
            )

        # if mask is an array of integers then we are doing delta masking
        if np.issubdtype(mask.dtype, np.integer):
            variants = ~self.invariants(x)
            mask = np.asarray(mask, dtype=np.int64)
            x = np.asarray(x, dtype=self.data.dtype)
            curr_delta_inds = np.zeros(len(mask), dtype=np.int64)
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
                df = pd.DataFrame(masked_inputs_out, columns=self.feature_names)
                return (df,), varying_rows_out

            return (masked_inputs_out,), varying_rows_out

        # otherwise we update the whole set of masked data for a single sample
        self._masked_data[:] = x * mask + self.data * np.invert(mask)
        self._last_mask[:] = mask

        if self.output_dataframe:
            return pd.DataFrame(self._masked_data, columns=self.feature_names)

        return (self._masked_data,)

    def invariants(self, x):
        """This returns a mask of which features change when we mask them.

        This optional masking method allows explainers to avoid re-evaluating the model when
        the features that would have been masked are all invariant.
        """
        # make sure we got valid data
        if x.shape != self.data.shape[1:]:
            raise DimensionError(
                "The passed data does not match the background shape expected "
                f"by the masker! The data of shape {str(x.shape)}"
                " was passed while the masker expected "
                f"data of shape {str(self.data.shape[1:])}."
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
            s.save("partition", None)

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
            kwargs["partition"] = s.load("partition")
        return kwargs


class Independent(Tabular):
    """Build a Independent masker with the given background data.

    This masks out tabular features by integrating over the given background dataset.

    Parameters
    ----------
    data : 2D numpy.ndarray or pandas.DataFrame
        The background dataset that is used for masking.

    max_samples : int, optional, default=100
        The maximum number of samples to use from the passed background data. If data has more
        than max_samples then shap.utils.sample is used to subsample the dataset. The number of
        samples coming out of the masker (to be integrated over) matches the number of samples in
        the background dataset. This means larger background dataset cause longer runtimes. Normally
        about 1, 10, 100, or 1000 background samples are reasonable choices.
    """

    def __init__(self, data, max_samples=100):
        super().__init__(data, max_samples=max_samples)


class Partition(Tabular):
    """Build a Partition masker with the given background data and clustering.

    This masks out tabular features by integrating over the given background dataset.

    Unlike Independent, Partition respects a hierarchical structure of the data.

    Parameters
    ----------
    data : 2D numpy.ndarray or pandas.DataFrame
        The background dataset that is used for masking.

    max_samples : int, optional, default=100
        The maximum number of samples to use from the passed background data. If data has more
        than max_samples then shap.utils.sample is used to subsample the dataset. The number of
        samples coming out of the masker (to be integrated over) matches the number of samples in
        the background dataset. This means larger background dataset cause longer runtimes. Normally
        about 1, 10, 100, or 1000 background samples are reasonable choices.

    clustering : string or numpy.ndarray or None, optional, default="correlation"
        If None, then no clustering is used and the masker behaves like Independent.
        If a string, then this is the distance metric to use for creating
        the clustering of
        the features. The distance function can be any valid
        `scipy.spatial.distance.pdist`'s metric
        argument. However we suggest using 'correlation' in most cases. The full list of options is
        `braycurtis`, `canberra`, `chebyshev`, `cityblock`, `correlation`, `cosine`, `dice`,
        `euclidean`, `hamming`, `jaccard`, `jensenshannon`, `kulsinski`, `mahalanobis`,
        `matching`, `minkowski`, `rogerstanimoto`, `russellrao`, `seuclidean`,
        `sokalmichener`, `sokalsneath`, `sqeuclidean`, `yule`. These are all
        the options from scipy.spatial.distance.pdist's metric argument.
        If an array, then this is assumed to be the clustering of the features.
    """

    def __init__(self, data, max_samples=100, clustering="correlation"):

        self.clustering = clustering
        super().__init__(data, max_samples=max_samples, clustering=clustering)


# we should inherit from Tabular once we add support for arbitrary masking
class Impute(Masker):
    """Build a Partition masker with the given background data and clustering.

    This imputes the values of missing features using the values of the observed features.

    Unlike Independent, Gaussian imputes missing values based on correlations with observed data points.

    Parameters
    ----------
    data : 2D numpy.ndarray, pandas.DataFrame or {"mean: numpy.ndarray, "cov": numpy.ndarray} dictionary
        The background dataset that is used for masking.
    """

    def __init__(self, data, method="linear"):
        if isinstance(data, dict) and "mean" in data:
            self.mean = data.get("mean", None)
            self.cov = data.get("cov", None)
            data = np.expand_dims(data["mean"], 0)

        self.data = data
        self.method = method

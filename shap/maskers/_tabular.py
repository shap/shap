import pandas as pd
import numpy as np
import scipy as sp
import scipy.cluster
from .. import utils
from ..utils import safe_isinstance, MaskedModel
from ._masker import Masker
from numba import jit
import logging

log = logging.getLogger('shap')


class Tabular(Masker):
    """ A common base class for TabularIndependent and TabularPartitions.
    """

    def __init__(self, data, sample=None, clustering=None):
        """ This masks out tabular features by integrating over the given background dataset. 
        
        Parameters
        ----------
        data : np.array, pandas.DataFrame
            The background dataset that is used for masking. The number of samples coming out of
            the masker (to be integrated over) matches the number of samples in this background
            dataset. This means larger background dataset cause longer runtimes. Normally about
            1, 10, 100, or 1000 background samples are reasonable choices.

        clustering : string or None (default) or numpy.ndarray
            The distance metric to use for creating the clustering of the features. The
            distance function can be any valid scipy.spatial.distance.pdist's metric argument.
            However we suggest using 'correlation' in most cases. The full list of options is
            ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’,
            ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulsinski’, ‘mahalanobis’,
            ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’,
            ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’. These are all
            the options from scipy.spatial.distance.pdist's metric argument.
        """

        self.output_dataframe = False
        if safe_isinstance(data, "pandas.core.frame.DataFrame"):
            self.input_names = data.columns
            data = data.values
            self.output_dataframe = True

        if sample is not None:
            data = utils.sample(data, sample)
            
        self.data = data
        self.clustering = clustering

        # warn users about large background data sets
        if self.data.shape[0] > 100:
            log.warning("Using " + str(self.data.shape[0]) + " background data samples could cause slower " +
                        "run times. Consider shap.utils.sample(data, K) to summarize the background as K samples.")

        # compute the clustering of the data
        if clustering is not None:
            if type(clustering) is str:
                self.clustering = utils.hclust(data, metric=clustering)
            elif safe_isinstance(clustering, "numpy.ndarray"):
                self.clustering = clustering
            else:
                raise Exception("Unknown clustering given! Make sure you pass a distance metric as a string or a clustering as an numpy.ndarray.")
        else:
            self.clustering = None

        # self._last_mask = np.zeros(self.data.shape[1], dtype=np.bool)
        self._masked_data = data.copy()
        self._last_mask = np.zeros(data.shape[1], dtype=np.bool)
        self.shape = self.data.shape
        self.supports_delta_masking = True
        # self._last_x = None
        # self._data_variance = np.ones(self.data.shape, dtype=np.bool)

        # this is property that allows callers to check what rows actually changed since last time.
        # self.changed_rows = np.ones(self.data.shape[0], dtype=np.bool)
    
    def __call__(self, mask, x):

        # make sure we are given a single sample
        if len(x.shape) != 1 or x.shape[0] != self.data.shape[1]:
            raise Exception("The input passed to maskers.Tabular for masking does not match the background data shape!")
        
        # if mask is an array of integers then we are doing delta masking
        if np.issubdtype(mask.dtype, np.integer):
            
            variants = ~self.invariants(x)
            variants_column_sums = variants.sum(0)
            batch_positions = np.zeros(len(mask)+1, dtype=np.int)
            curr_delta_inds = np.zeros(len(mask), dtype=np.int)
            num_masks = (mask >= 0).sum()
            varying_rows_out = np.zeros((num_masks, self.shape[0]), dtype=np.bool)
            masked_inputs_out = np.zeros((num_masks * self.shape[0], self.shape[1]))
            self._last_mask[:] = False
            self._masked_data[:] = self.data
            _delta_masking(
                mask, x, batch_positions, curr_delta_inds,
                varying_rows_out, self._masked_data, self._last_mask, self. data, variants,
                variants_column_sums, masked_inputs_out, MaskedModel.delta_mask_noop_value
            )
            if self.output_dataframe:
                return (pd.DataFrame(masked_inputs_out, columns=self.input_names),), varying_rows_out
            else:
                return (masked_inputs_out,), varying_rows_out
        
        # otherwise we update the whole set of masked data for a single sample
        else:
            self._masked_data[:] = x * mask + self.data * np.invert(mask)
            self._last_mask[:] = mask

        if self.output_dataframe:
            return pd.DataFrame(self._masked_data, columns=self.input_names)
        else:
            return self._masked_data

    # def reset_delta_masking(self):
    #     """ This resets the masker back to all zeros when delta masking.

    #     Note that the presence of this function also denotes that we support delta masking.
    #     """
    #     self._masked_data[:] = self.data
    #     self._last_mask[:] = False
    

    def invariants(self, x):

        # make sure we got valid data
        if x.shape != self.data.shape[1:]:
            raise Exception(
                "The passed data does not match the background shape expected by the masker! The data of shape " + \
                str(x.shape) + " was passed while the masker expected data of shape " + str(self.data.shape[1:]) + "."
            )

        return np.isclose(x, self.data)

@jit
def _single_delta_mask(dind, masked_inputs, last_mask, data, x, noop_code):
    if dind == noop_code:
        pass
    elif last_mask[dind]:
        masked_inputs[:,dind] = data[:,dind]
        last_mask[dind] = False
    else:
        masked_inputs[:,dind] = x[dind]
        last_mask[dind] = True

@jit
def _delta_masking(masks, x, batch_positions, curr_delta_inds, varying_rows_out,
                   masked_inputs_tmp, last_mask, data, variants, variants_column_sums, masked_inputs_out, noop_code):
    """ Implements the special (high speed) delta masking API that only flips the positions we need to.

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
        while curr_delta_inds[dpos] < 0: # negative values mean keep going
            curr_delta_inds[dpos] = -curr_delta_inds[dpos] - 1 # -value + 1 is the original index that needs flipped
            _single_delta_mask(curr_delta_inds[dpos], masked_inputs_tmp, last_mask, data, x, noop_code)
            dpos += 1
            curr_delta_inds[dpos] = masks[masks_pos + dpos]
        _single_delta_mask(curr_delta_inds[dpos], masked_inputs_tmp, last_mask, data, x, noop_code)
        
        # copy the tmp masked inputs array to the output
        masked_inputs_out[output_pos:output_pos+N] = masked_inputs_tmp
        masks_pos += dpos + 1

        # mark which rows have been updated, so we can only evaluate the model on the rows we need to
        if i == 0: 
            varying_rows_out[i,:] = True
            
        else: 
            # only one column was changed
            if dpos == 0: 
                varying_rows_out[i,:] = variants[:,curr_delta_inds[dpos]]
                
            # more than one column was changed
            else: 
                varying_rows_out[i,:] = np.sum(variants[:,curr_delta_inds[:dpos+1]], axis=1) > 0
        
        output_pos += N



class TabularIndependent(Tabular):
    """ This masks out tabular features by integrating over the given background dataset. 
    """

    def __init__(self, data, sample=None):
        """ Build a TabularIndependent masker with the given background data.

        Parameters
        ----------
        data : numpy.ndarray, pandas.DataFrame
            The background dataset that is used for masking. The number of samples coming out of
            the masker (to be integrated over) matches the number of samples in this background
            dataset. This means larger background dataset cause longer runtimes. Normally about
            1, 10, 100, or 1000 background samples are reasonable choices.

        sample : None or int
            If not None then we randomly subsample the passed data to this number of samples. This
            is important since we often need to evaluate the model for each sample every time we mask
            the input in a different way. Common values here are 10, or 100 (or just passing a single
            sample as a background reference).
        """
        super(TabularIndependent, self).__init__(data, sample=sample, clustering=None)


class TabularPartitions(Tabular):
    """ This masks out tabular features by integrating over the given background dataset.

    Unlike TabularIndependent, TabularPartitions respects a hierarchial structure o
    """

    def __init__(self, data, sample=None, clustering="correlation"):
        """ Build a TabularPartitions masker with the given background data and clustering.

        Parameters
        ----------
        data : numpy.ndarray, pandas.DataFrame
            The background dataset that is used for masking. The number of samples coming out of
            the masker (to be integrated over) matches the number of samples in this background
            dataset. This means larger background dataset cause longer runtimes. Normally about
            1, 10, 100, or 1000 background samples are reasonable choices.

        clustering : string or numpy.ndarray
            If a string, then this is the distance metric to use for creating the clustering of
            the features. The distance function can be any valid scipy.spatial.distance.pdist's metric
            argument. However we suggest using 'correlation' in most cases. The full list of options is
            ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’,
            ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulsinski’, ‘mahalanobis’,
            ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’,
            ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’. These are all
            the options from scipy.spatial.distance.pdist's metric argument.
            If an array, then this is assumed to be the clustering of 
        """
        super(TabularPartitions, self).__init__(data, sample=sample, clustering=clustering)


# class ConditionedTabular(ConditionedMasker):
#     def __init__(self, masker, x, collapse_invariances=True):
#         """ This represents a Tabular masker that has been conditioned on a specific input sample. 
        
#         Parameters
#         ----------
#         masker : shap.maskers.Tabular
#             This is the unconditioned masker object that we will then condition on a single sample.
#             By building a new object conditioned on a single sample we can optimize the masking
#             process to that sample. This is important since explainers often evaluate many masking
#             patterns on the same input sample to explain the model's performance for that input.

#         x : np.ndarray
#             A single row of the data that would be passed to the model.

#         collapse_invariances : bool
#             If true then we drop masking positions where masking makes no difference.
#         """

#         assert collapse_invariances, "right now we always do this"

#         self.masker = masker
#         self.clustering = self.masker.clustering # we use a global shared partition tree
#         self.x = x
#         self._data_variance = ~np.isclose(x, self.masker.data)
#         self.changed_rows[:] = True
#         self.last_mask = 

#         # this is property that allows callers to check what rows actually changed since last time.
#         self.changed_rows = np.ones(self.data.shape[0], dtype=np.bool)

#         self.varying_positions = np.where(np.any(self._data_variance, axis=0))[0]

#         self.output_dataframe = False
#         if safe_isinstance(data, "pandas.core.frame.DataFrame"):
#             self.input_names = data.columns
#             data = data.values
#             self.output_dataframe = True
            
#         self.data = data
#         self.clustering = clustering

#         # warn users about large background data sets
#         if self.data.shape[0] > 100:
#             log.warning("Using " + str(self.data.shape[0]) + " background data samples could cause slower " +
#                         "run times. Consider shap.sample(data, K) to summarize the background as K samples.")

#         # compute the clustering of the data
#         if clustering is not None:
#             bg_no_nan = data.copy()
#             for i in range(bg_no_nan.shape[1]):
#                 np.nan_to_num(bg_no_nan[:,i], nan=np.nanmean(bg_no_nan[:,i]), copy=False)
#             D = sp.spatial.distance.pdist(bg_no_nan.T + np.random.randn(*bg_no_nan.T.shape)*1e-8, metric=clustering)
#             self.clustering = sp.cluster.hierarchy.complete(D)
#         else:
#             self.clustering = None

#         self._last_mask = np.zeros(self.data.shape[1], dtype=np.bool)
#         self._masked_data = data.copy()
#         self._last_x = None
#         self._data_variance = np.ones(self.data.shape, dtype=np.bool)

#         # this is property that allows callers to check what rows actually changed since last time.
#         self.changed_rows = np.ones(self.data.shape[0], dtype=np.bool)
    
#     def __call__(self, mask=None, invariances=False):

#         # if mask is not given then we mask all features
#         if mask is None:
#             mask = np.zeros(self.masker.data.shape[1], dtype=np.bool)
        
#         if self._last_mask

#         # if we are given a different x to mask than last time, then we update our cache
#         if self._last_x is None or not np.allclose(x, self._last_x):
#             self._last_x = x
#             self._data_variance[:] = ~np.isclose(x, self.data)
#             self.changed_rows[:] = True

#         # otherwise we note which rows changed
#         else:
#             np.any(self._data_variance * (self._last_mask ^ mask), axis=1, out=self.changed_rows)
        
#         # update our masked data values
#         self._masked_data[:] = x * mask + self.data * np.invert(mask)
#         self._last_mask[:] = mask

#         if self.output_dataframe:
#             return pd.DataFrame(self._masked_data, columns=self.input_names)
#         else:
#             return self._masked_data

#     def __len__(self):
#         return self.data.shape[1]
# import pandas as pd
# import numpy as np
# import scipy as sp
from ._masker import Masker

class Composite(Masker):
    """ This merges several maskers for different inputs together into a single composite masker.

    This is not yet implemented.
    """

    # def __init__(self, *maskers, clustering=None):
    #     """ This merges several maskers for different inputs together into a single composite masker.

    #     Parameters
    #     ----------
    #     background_data : np.array, pandas.DataFrame
    #         The background dataset that is used for masking. The number of samples coming out of
    #         the masker (to be integrated over) matches the number of samples in this background
    #         dataset. This means larger background dataset cause longer runtimes. Normally about
    #         1, 10, 100, or 1000 background samples are reasonable choices.

    #     clustering : "correlation", string or None (default)
    #         The distance metric to use for creating the partition_tree of the features. The
    #         distance function can be any valid scipy.spatial.distance.pdist's metric argument.
    #         However we suggest using 'correlation' in most cases. The full list of options is
    #         ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’,
    #         ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulsinski’, ‘mahalanobis’,
    #         ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’,
    #         ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’. These are all
    #         the options from scipy.spatial.distance.pdist's metric argument.
    #     """

    #     self.maskers = maskers

    #     # self.output_dataframe = False
    #     # if safe_isinstance(background_data, "pandas.core.frame.DataFrame"):
    #     #     self.input_names = background_data.columns
    #     #     background_data = background_data.values
    #     #     self.output_dataframe = True

    #     self.background_data = background_data
    #     self.clustering = clustering

    #     # compute the clustering of the data
    #     if clustering is not None:
    #         bg_no_nan = background_data.copy()
    #         for i in range(bg_no_nan.shape[1]):
    #             np.nan_to_num(bg_no_nan[:,i], nan=np.nanmean(bg_no_nan[:,i]), copy=False)
    #         D = sp.spatial.distance.pdist(bg_no_nan.T + np.random.randn(*bg_no_nan.T.shape)*1e-8, metric=clustering)
    #         self.partition_tree = sp.cluster.hierarchy.complete(D)
    #     else:
    #         self.partition_tree = None

    # def __call__(self, x, mask=None):

    #     # if mask is not given then we mask all features
    #     if mask is None:
    #         mask = np.zeros(np.prod(x.shape), dtype=bool)

    #     out = x * mask + self.background_data * np.invert(mask)

    #     if self.output_dataframe:
    #         return pd.DataFrame(out, columns=self.input_names)
    #     else:
    #         return out

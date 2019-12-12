import pandas as pd
import scipy as sp
import numpy as np
import warnings

class PartitionExplainer():
    
    def __init__(self, model, masker, clustering):
        """ Uses the Partition SHAP method to explain the output of any function.

        Partition SHAP computes Shapley values recursively through a hierarchy of features, this
        hierarchy defines feature coalitions and results in the Owen values from game theory. The
        PartitionExplainer has two particularly nice properties: 1) PartitionExplainer is
        model-agnostic but only has quadradic exact runtime when using a balanced partition tree
        (in term of the number of input features). This is in contrast to the exponential exact
        runtime of KernalExplainer. 2) PartitionExplainer always assigns to groups of correlated
        features the credit that that set of features would have had when treated as a group. This
        means if the hierarchal clustering given to PartitionExplainer groups correlated features
        together, then feature correlations are "accounted for"...meaning that the total credit assigned
        to a group of tightly dependent features does net depend on how they behave if their correlation
        structure was broken during the explanation's perterbation process. Note that for linear models
        with independent features the Owen values that PartitionExplainer returns are the same as the
        input-level Shapley values.


        Parameters
        ----------
        model : function
            User supplied function that takes a matrix of samples (# samples x # features) and
            computes a the output of the model for those samples.

        masker : function or numpy.array or pandas.DataFrame
            The function used to "mask" out hidden features of the form `masker(x, mask)`. It takes a
            single input sample and a binary mask and returns a matrix of masked samples. These
            masked samples will then be evaluated using the model function and the outputs averaged.
            As a shortcut for the standard masking using by SHAP you can pass a background data matrix
            instead of a function and that matrix will be used for masking.

        clustering : numpy.array
            A hierarchal clustering of the input features represted by a matrix that follows the format
            used by scipy.cluster.hierarchy (see the notebooks/partition_explainer directory an example). 
        """

        warnings.warn("PartitionExplainer is still in an alpha state, so use with caution...")
        
        # If the user just gave a dataset as the masker
        # then we make a masker that perturbs features independently
        if type(masker) == np.ndarray:
            self.masker_data = masker
            self.masker = lambda x, mask: x * mask + self.masker_data * np.invert(mask)
        
        self.model = model
        self.expected_value = None
        self.clustering = clustering
        
        self.mask_matrix = make_masks(self.clustering)

        self.merge_clusters = -np.ones((2 * clustering.shape[0] + 1, 2), dtype=np.int64)
        self.merge_clusters[clustering.shape[0] + 1:] = clustering[:,:2]

        self.values = np.zeros(self.merge_clusters.shape[0])
        self.counts = np.zeros(self.merge_clusters.shape[0], dtype=np.int64)
    
    def shap_values(self, x, tol=0):
        out = np.zeros(x.shape)
        for i in range(x.shape[0]):
            out[i] = self.explain(x[i], tol)
        return out
    
    def explain(self, x, tol):
        
        if self.expected_value is None:
            self.expected_value = self.model(self.masker(x, np.zeros(x.shape, dtype=np.bool))).mean(0)
        
        self.values[:] = 0
        self.counts[:] = 0
        owen(
            self.model, x, self.masker, np.zeros(self.mask_matrix.shape[1], dtype=np.bool), 
            self.expected_value, self.model(x.reshape(1,len(x)))[0], len(self.values)-1,
            self.values, self.counts, self.merge_clusters, self.mask_matrix,
            tol=tol
        )
        self.values[:-1] /= self.counts[:-1] + 1e-8
        
        return self.values[:len(x)]  


def rec_fill_masks(mask_matrix, cluster_matrix, ind=None):
    if ind is None:
        ind = cluster_matrix.shape[0] - 1

    lind = int(cluster_matrix[ind,0]) #- mask_matrix.shape[1]
    rind = int(cluster_matrix[ind,1]) #- mask_matrix.shape[1]
    
    ind += mask_matrix.shape[1]
    
    if lind < mask_matrix.shape[1]:
        mask_matrix[ind, lind] = 1
    else:
        rec_fill_masks(mask_matrix, cluster_matrix, lind - mask_matrix.shape[1])
        mask_matrix[ind, :] += mask_matrix[lind, :]
        
    if rind < mask_matrix.shape[1]:
        mask_matrix[ind, rind] = 1
    else:
        rec_fill_masks(mask_matrix, cluster_matrix, rind - mask_matrix.shape[1])
        mask_matrix[ind, :] += mask_matrix[rind, :]


def make_masks(cluster_matrix):
    mask_matrix = np.zeros((2 * cluster_matrix.shape[0] + 1, cluster_matrix.shape[0] + 1), dtype=np.bool)
    for i in range(cluster_matrix.shape[0] + 1):
        mask_matrix[i,i] = 1
    rec_fill_masks(mask_matrix, cluster_matrix)
    return mask_matrix

def owen(f, x, r, m00, f00, f11, ind, values, counts, merge_clusters, mask_matrix, tol=-1):
    """ Compute a nested set of recursive Owen values.
    """
    
    # get the left are right children of this cluster
    lind = merge_clusters[ind, 0]
    rind = merge_clusters[ind, 1]
    
    # check if we are a leaf node
    if lind < 0: return
    
    # build the masks
    m10 = m00 + mask_matrix[lind, :]
    m01 = m00 + mask_matrix[rind, :]

    # evaluate the model on the two new masked inputs
    f10 = f(r(x, m10)).mean(0)
    f01 = f(r(x, m01)).mean(0)

    # update the left node
    values[lind] += (f10 - f00) + (f11 - f01)
    counts[lind] += 2

    # recurse on the left node
    if np.abs((f10 - f00) - (f11 - f01)) > tol: # don't do two recursions if there is no interaction
        owen(f, x, r, m01, f01, f11, lind, values, counts, merge_clusters, mask_matrix, tol)
    owen(f, x, r, m00, f00, f10, lind, values, counts, merge_clusters, mask_matrix, tol)

    # update the right node
    values[rind] += (f01 - f00) + (f11 - f10)
    counts[rind] += 2
    
    # recurse on the right node
    if np.abs((f01 - f00) - (f11 - f10)) > tol: # don't do two recursions if there is no interaction
        owen(f, x, r, m10, f10, f11, rind, values, counts, merge_clusters, mask_matrix, tol)
    owen(f, x, r, m00, f00, f01, rind, values, counts, merge_clusters, mask_matrix, tol)
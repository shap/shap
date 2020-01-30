import pandas as pd
import scipy as sp
import numpy as np
import warnings
import time
from tqdm.auto import tqdm
from ..common import safe_isinstance


class PartitionExplainer():
    
    def __init__(self, model, masker, partition_tree):
        """ Uses the Partition SHAP method to explain the output of any function.

        Partition SHAP computes Shapley values recursively through a hierarchy of features, this
        hierarchy defines feature coalitions and results in the Owen values from game theory. The
        PartitionExplainer has two particularly nice properties: 1) PartitionExplainer is
        model-agnostic but only has quadradic exact runtime when using a balanced partition tree
        (in term of the number of input features). This is in contrast to the exponential exact
        runtime of KernalExplainer. 2) PartitionExplainer always assigns to groups of correlated
        features the credit that that set of features would have had when treated as a group. This
        means if the hierarchical clustering given to PartitionExplainer groups correlated features
        together, then feature correlations are "accounted for"...meaning that the total credit assigned
        to a group of tightly dependent features does net depend on how they behave if their correlation
        structure was broken during the explanation's perterbation process. Note that for linear models
        the Owen values that PartitionExplainer returns are the same as the standard non-hierarchical
        Shapley values.


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

        partition_tree : function or numpy.array
            A hierarchical clustering of the input features represted by a matrix that follows the format
            used by scipy.cluster.hierarchy (see the notebooks/partition_explainer directory an example).
            If this is a function then the function produces a clustering matrix when given a single input
            example. 
        """

        warnings.warn("PartitionExplainer is still in an alpha state, so use with caution...")
        
        # convert dataframes
        if safe_isinstance(masker, "pandas.core.series.Series"):
            masker = masker.values
        elif safe_isinstance(masker, "pandas.core.frame.DataFrame"):
            masker = masker.values

        # If the user just gave a dataset as the masker
        # then we make a masker that perturbs features independently
        if type(masker) == np.ndarray:
            self.masker_data = masker
            self.masker = lambda x, mask: x * mask + self.masker_data * np.invert(mask)
        else:
            self.masker = masker

        self.model = model
        self.expected_value = None
        self.partition_tree = partition_tree

        # if we don't have a dynamic clustering algorithm then we can precompute
        # a lot of information
        if not callable(self.partition_tree):
            self.create_cluster_matrices(self.partition_tree)

    def create_cluster_matrices(self, partition_tree):
        self.mask_matrix = make_masks(partition_tree)
        self.merge_clusters = -np.ones((2 * partition_tree.shape[0] + 1, 3), dtype=np.int64)
        self.merge_clusters[partition_tree.shape[0] + 1:,:2] = partition_tree[:,:2]
        for i in range(self.merge_clusters.shape[0]):
            if self.merge_clusters[i,0] < 0:
                self.merge_clusters[i,2] = 1
            else:
                self.merge_clusters[i,2] = self.merge_clusters[self.merge_clusters[i,0],2] + self.merge_clusters[self.merge_clusters[i,1],2]

        self.values = np.zeros(self.merge_clusters.shape[0])
        self.counts = np.zeros(self.merge_clusters.shape[0], dtype=np.int64)
        self.dvalues = np.zeros(self.merge_clusters.shape[0])
        self.dcounts = np.zeros(self.merge_clusters.shape[0], dtype=np.int64)
    
    def hierarchical_shap_values(self, x, tol=0, max_evals=None, mode="single_context", silent=False):
        
        # convert dataframes
        if safe_isinstance(x, "pandas.core.series.Series"):
            x = x.values
        elif safe_isinstance(x, "pandas.core.frame.DataFrame"):
            x = x.values

        # handle list inputs that are not multiple inputs by auto-wrapping them as np.arrays
        if type(x) is list and not hasattr(x[0], "__len__"):
            x = np.array(x)

        
        single_instance = False
        if len(x.shape) == 1:
            single_instance = True
            x = x.reshape(1,-1)
        
        orig_shape = x.shape[1:]
        x = x.reshape(x.shape[0], -1)
        self.model_orig = self.model
        self.model = lambda x: self.model_orig(x.reshape(x.shape[0], *orig_shape))

        out = np.zeros((x.shape[0], x.shape[1]*2 - 1))
        for i in range(x.shape[0]):
            out[i] = self.explain(x[i], tol, "hierarchical", max_evals, mode, silent)
        
        self.model = self.model_orig

        if single_instance:
            return out[0]
        else:
            return out
        
    def shap_values(self, x, tol=0, max_evals=None, mode="single_context", silent=False):
        
        # convert dataframes
        if safe_isinstance(x, "pandas.core.series.Series"):
            x = x.values
        elif safe_isinstance(x, "pandas.core.frame.DataFrame"):
            x = x.values
        
        # handle list inputs that are not multiple inputs by auto-wrapping them as np.arrays
        if type(x) is list and not hasattr(x[0], "__len__"):
            x = np.array(x)
        elif type(x) is np.array and type(x[0]) is list:
            x = np.array([np.array(v) for v in x])
        
        single_instance = False
        if not hasattr(x[0], "__len__"):
            single_instance = True
            x = x.reshape(1,-1)
        
        out = []
        pbar = None
        start_time = time.time()
        for i in range(x.shape[0]):
            row_out = self.explain(x[i], tol, "marginal", max_evals, mode, silent)
            out.append(row_out)
            if pbar is None and time.time() - start_time > 5:
                pbar = tqdm(total=x.shape[0], disable=silent, leave=False)
                pbar.update(i)
            if pbar is not None:
                pbar.update(1)
        if pbar is not None:
            pbar.close()
        out = np.array(out)
        
        if single_instance:
            return out[0]
        else:
            return out

    def explain(self, x, tol, output_type, max_evals, mode, silent):
        if max_evals is None:
            max_evals = 10000000
        
        if self.expected_value is None:
            self.eval_time = time.time()
            self.expected_value = self.model(self.masker(x, np.zeros(x.shape, dtype=np.bool))).mean(0)
            self.eval_time = time.time() - self.eval_time
        
        if callable(self.partition_tree):
            self.create_cluster_matrices(self.partition_tree(x))
        
        self.values[:] = 0
        self.counts[:] = 0
        self.dvalues[:] = 0
        self.dcounts[:] = 0
        if mode == "single_context":
            single_context_owen(
                self.model, x, self.masker, np.zeros(self.mask_matrix.shape[1], dtype=np.bool), 
                self.expected_value, self.model(x.reshape(1,len(x)))[0], len(self.values)-1,
                self.dvalues, self.merge_clusters, self.mask_matrix, max_evals, silent, self.eval_time
            )

            # drop the interaction terms down onto self.values
            self.values[:] = self.dvalues
            M = len(x)
            if output_type == "marginal":
                def lower_credit(i, value=0):
                    if i < M:
                        self.values[i] += value
                        return
                    li = self.merge_clusters[i,0]
                    ri = self.merge_clusters[i,1]
                    group_size = self.merge_clusters[i,2]
                    lsize = self.merge_clusters[li,2]
                    rsize = self.merge_clusters[ri,2]
                    assert lsize+rsize == group_size, "Ah"
                    self.values[i] += value
                    lower_credit(li, self.values[i] * lsize / group_size)
                    lower_credit(ri, self.values[i] * rsize / group_size)
                lower_credit(len(self.dvalues) - 1)
        else:
            owen(
                self.model, x, self.masker, np.zeros(self.mask_matrix.shape[1], dtype=np.bool), 
                self.expected_value, self.model(x.reshape(1,len(x)))[0], len(self.values)-1,
                self.values, self.counts, self.merge_clusters, self.mask_matrix,
                tol=tol
            )
            self.values[:-1] /= self.counts[:-1] + 1e-8
            self.dvalues[:-1] /= self.dcounts[:-1] + 1e-8
        
        if output_type == "hierarchical":
            return self.dvalues
        else:
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

def owen(f, x, r, m00, f00, f11, ind, values, counts, merge_clusters, mask_matrix, tol=-1, multiplier=1):
    """ Compute a nested set of recursive Owen values.
    """
    
    # get the left and right children of this cluster
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

    dividend = f11 - f10 - f01 + f00

    # recurse on the left node
    if abs(dividend) > tol: # don't do two recursions if there is no meaningful interaction
        owen(f, x, r, m01, f01, f11, lind, values, counts, merge_clusters, mask_matrix, tol, multiplier)
        values[lind] += (f11 - f01) * multiplier
        counts[lind] += multiplier
    
        owen(f, x, r, m00, f00, f10, lind, values, counts, merge_clusters, mask_matrix, tol, multiplier)
        values[lind] += (f10 - f00) * multiplier
        counts[lind] += multiplier
    else:
        owen(f, x, r, m00, f00, f10, lind, values, counts, merge_clusters, mask_matrix, tol, multiplier * 2)
        values[lind] += (f10 - f00) * multiplier * 2
        counts[lind] += multiplier * 2
    
    # recurse on the right node
    if abs(dividend) > tol: # don't do two recursions if there is no meaningful interaction
        owen(f, x, r, m00, f00, f01, rind, values, counts, merge_clusters, mask_matrix, tol, multiplier)
        values[rind] += (f01 - f00) * multiplier
        counts[rind] += multiplier

        owen(f, x, r, m10, f10, f11, rind, values, counts, merge_clusters, mask_matrix, tol, multiplier)
        values[rind] += (f11 - f10) * multiplier
        counts[rind] += multiplier
    else:
        owen(f, x, r, m10, f10, f11, rind, values, counts, merge_clusters, mask_matrix, tol, multiplier * 2)
        values[rind] += (f11 - f10) * multiplier * 2
        counts[rind] += multiplier * 2


import queue
def single_context_owen(f, x, r, m00, f00, f11, ind, dvalues, merge_clusters, mask_matrix, max_evals, silent, eval_time):
    """ Compute a nested set of recursive Owen values.
    """
    
    q = queue.PriorityQueue()
    q.put((0, (m00, f00, f11, ind)))
    eval_count = 0
    total_evals = min(max_evals, (len(x)-1)*2)
    pbar = tqdm(total=total_evals, disable=silent or eval_time * total_evals < 5, leave=False)
    while not q.empty():
        
        # if we passed our execution limit then leave everything else on the internal nodes
        if eval_count > max_evals:
            while not q.empty():
                m00, f00, f11, ind = q.get()[1]
                dvalues[ind] = f11 - f00
            break
                
        # get our next set of arguments
        m00, f00, f11, ind = q.get()[1]
    
        # get the left are right children of this cluster
        lind = merge_clusters[ind, 0]
        rind = merge_clusters[ind, 1]

        # check if we are a leaf node or terminated our decent early and dumping credit at an internal node
        if lind < 0:
            dvalues[ind] = f11 - f00
            continue

        # build the masks
        m10 = m00 + mask_matrix[lind, :]
        m01 = m00 + mask_matrix[rind, :]

        # evaluate the model on the two new masked inputs
        f10 = f(r(x, m10)).mean(0)
        f01 = f(r(x, m01)).mean(0)
        eval_count += 2
        pbar.update(2)

        # update our dividends
        dvalues[ind] += f11 - f01 - f10 + f00

        # recurse on the left node
        args = (m00, f00, f10, lind)
        q.put((-abs(f10 - f00), args)) #/ np.sqrt(m10.sum()

        # recurse on the right node
        args = (m00, f00, f01, rind)
        q.put((-abs(f01 - f00), args))
    pbar.close()
    







openers = {
    "(": ")"
}
closers = {
    ")": "("
}
enders = [".", ","]
connectors = ["but", "and", "or"]

class Token():
    def __init__(self, value):
        self.s = value
        if value in openers or value in closers:
            self.balanced = False
        else:
            self.balanced = True
            
    def __str__(self):
        return self.s
    
    def __repr__(self):
        if not self.balanced:
            return self.s + "!"
        return self.s
    
class TokenGroup():
    def __init__(self, group, index=None):
        self.g = group
        self.index = index
    
    def __repr__(self):
        return self.g.__repr__()
    
    def __getitem__(self, index):
        return self.g[index]
    
    def __add__(self, o):
        return TokenGroup(self.g + o.g)
    
    def __len__(self):
        return len(self.g)

def merge_score(group1, group2):
    score = 0
    
    # merge broken-up parts of words first
    if group2[0].s.startswith("##"):
        score += 20
    
    start_ctrl = group1[0].s[0] == "[" and group1[0].s[-1] == "]"
    end_ctrl = group2[-1].s[0] == "[" and group2[-1].s[-1] == "]"
    if (start_ctrl and not end_ctrl) or (end_ctrl and not start_ctrl):
        score -= 1000
    if group2[0].s in openers and not group2[0].balanced:
        score -= 100
    if group1[-1].s in closers and not group1[-1].balanced:
        score -= 100
    
    # attach surrounding an openers and closers a bit later
    if group1[0].s in openers and not group2[-1] in closers:
        score -= 2
    
    # reach across connectors later
    if group1[-1].s in connectors or group2[0].s in connectors:
        score -= 2
        
    # reach across commas later
    if group1[-1].s == ",":
        score -= 10
    if group2[0].s == ",":
        if len(group2) > 1: # reach across
            score -= 10
        else:
            score -= 1
        
    # reach across peroids later
    if group1[-1].s == ".":
        score -= 20
    if group2[0].s == ".":
        if len(group2) > 1: # reach across
            score -= 20
        else:
            score -= 1
    
    score -= len(group1) + len(group2)
    
    return score
    
def merge_closest_groups(groups):
    scores = [merge_score(groups[i], groups[i+1]) for i in range(len(groups)-1)]
    #print(scores)
    ind = np.argmax(scores)
    groups[ind] = groups[ind] + groups[ind+1]
    #print(groups[ind][0].s in openers, groups[ind][0])
    if groups[ind][0].s in openers and groups[ind+1][-1].s == openers[groups[ind][0].s]:
        groups[ind][0].balanced = True
        groups[ind+1][-1].balanced = True
        
    
    groups.pop(ind+1)    
    
def partition_tree(decoded_tokens):
    token_groups = [TokenGroup([Token(t)], i) for i,t in enumerate(decoded_tokens)]
#     print(token_groups)
    M = len(decoded_tokens)
    new_index = M
    clustm = np.zeros((M-1, 2), dtype=np.int)
    for i in range(len(token_groups)-1):
        scores = [merge_score(token_groups[i], token_groups[i+1]) for i in range(len(token_groups)-1)]
#         print(scores)
        ind = np.argmax(scores)

        clustm[new_index-M,0] = token_groups[ind].index
        clustm[new_index-M,1] = token_groups[ind+1].index

        token_groups[ind] = token_groups[ind] + token_groups[ind+1]
        token_groups[ind].index = new_index

        # track balancing of openers/closers
        if token_groups[ind][0].s in openers and token_groups[ind+1][-1].s == openers[token_groups[ind][0].s]:
            token_groups[ind][0].balanced = True
            token_groups[ind+1][-1].balanced = True

        token_groups.pop(ind+1)
        new_index += 1
#         print(token_groups)
    return clustm

class TokenMasker():
    """ This masks out tokens according to the given tokenizer.

    The masked variables are 
    """
    def __init__(self, tokenizer, mask_token="auto"):
        self.mask_history = {}
        self.tokenizer = tokenizer
        null_tokens = tokenizer.encode("")
        
        if mask_token == "auto":
            if hasattr(self.tokenizer, "mask_token_id"):
                self.mask_token = self.tokenizer.mask_token_id
            else:
                self.mask_token = None
        assert len(null_tokens) % 2 == 0, "An odd number of boundary tokens are added to the null string!"
        self.keep_prefix = len(null_tokens) // 2
        self.keep_suffix = len(null_tokens) // 2
    
    def __call__(self, x, mask):
        mask = mask.copy()
        mask[:self.keep_prefix] = True
        mask[-self.keep_suffix:] = True
        if self.mask_token is None:
            out = x[mask]
        else:
#             out = []
#             need_mask_token = True
#             for i in range(len(mask)):
#                 if mask[i]:
#                     out.append(x[i])
#                     need_mask_token = True
#                 elif need_mask_token:
#                     out.append(self.mask_token)
#                     need_mask_token = False
#             out = np.array(out)
            out = np.array([x[i] if mask[i] else self.mask_token for i in range(len(mask))])
        #print(out)
        return out.reshape(1,-1)

    def partition_tree(self, x):
        decoded_x = [self.tokenizer.decode([v]) for v in x]
        return partition_tree(decoded_x)
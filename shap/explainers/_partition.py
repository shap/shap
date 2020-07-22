import pandas as pd
import scipy as sp
import numpy as np
import warnings
import time
from tqdm.auto import tqdm
import queue
from ..utils import assert_import, record_import_error, safe_isinstance, make_masks
from .._explanation import Explanation
from .. import maskers
from ._explainer import Explainer

# .shape[0] messes up pylint a lot here
# pylint: disable=unsubscriptable-object


class Partition(Explainer):
    
    def __init__(self, model, masker, partition_tree=None, output_names=None):
        """ Uses the Partition SHAP method to explain the output of any function.

        Partition SHAP computes Shapley values recursively through a hierarchy of features, this
        hierarchy defines feature coalitions and results in the Owen values from game theory. The
        PartitionExplainer has two particularly nice properties: 1) PartitionExplainer is
        model-agnostic but when using a balanced partition tree only has quadradic exact runtime 
        (in term of the number of input features). This is in contrast to the exponential exact
        runtime of KernalExplainer or SamplingExplainer. 2) PartitionExplainer always assigns to groups of
        correlated features the credit that set of features would have had if treated as a group. This
        means if the hierarchical clustering given to PartitionExplainer groups correlated features
        together, then feature correlations are "accounted for" ... in the sense that the total credit assigned
        to a group of tightly dependent features does net depend on how they behave if their correlation
        structure was broken during the explanation's perterbation process. Note that for linear models
        the Owen values that PartitionExplainer returns are the same as the standard non-hierarchical
        Shapley values.


        Parameters
        ----------
        model : function
            User supplied function that takes a matrix of samples (# samples x # features) and
            computes the output of the model for those samples.

        masker : function or numpy.array or pandas.DataFrame or tokenizer
            The function used to "mask" out hidden features of the form `masker(mask, x)`. It takes a
            single input sample and a binary mask and returns a matrix of masked samples. These
            masked samples will then be evaluated using the model function and the outputs averaged.
            As a shortcut for the standard masking using by SHAP you can pass a background data matrix
            instead of a function and that matrix will be used for masking. Domain specific masking
            functions are available in shap such as shap.maksers.Image for images and shap.maskers.Text
            for text.

        partition_tree : None or function or numpy.array
            A hierarchical clustering of the input features represented by a matrix that follows the format
            used by scipy.cluster.hierarchy (see the notebooks/partition_explainer directory an example).
            If this is a function then the function produces a clustering matrix when given a single input
            example. If you are using a standard SHAP masker object then you can pass masker.partition_tree
            to use that masker's built-in clustering of the features, or if partition_tree is None then
            masker.partition_tree will be used by default.
        """

        super(Partition, self).__init__(model, masker, algorithm="partition")

        warnings.warn("explainers.Partition is still in an alpha state, so use with caution...")
        
        # convert dataframes
        # if safe_isinstance(masker, "pandas.core.frame.DataFrame"):
        #     masker = TabularMasker(masker)
        # elif safe_isinstance(masker, "numpy.ndarray") and len(masker.shape) == 2:
        #     masker = TabularMasker(masker)
        # elif safe_isinstance(masker, "transformers.PreTrainedTokenizer"):
        #     masker = TextMasker(masker)
        # self.masker = masker

        # TODO: maybe? if we have a tabular masker then we build a PermutationExplainer that we
        # will use for sampling
        self.input_shape = masker.shape[1:] if hasattr(masker, "shape") else None
        self.output_names = output_names

        self.model = lambda x: np.array(model(x))
        self.expected_value = None
        if getattr(self.masker, "partition_tree", None) is None:
            raise ValueError("The passed masker must have a .partition_tree attribute defined! Try shap.maskers.Tabular(data, clustering=\"correlation\") for example.")
        # if partition_tree is None:
        #     if not hasattr(masker, "partition_tree"):
        #         raise ValueError("The passed masker does not have masker.partition_tree, so the partition_tree must be passed!")
        #     self.partition_tree = masker.partition_tree
        # else:
        #     self.partition_tree = partition_tree
        
        # handle higher dimensional tensor inputs
        if self.input_shape is not None and len(self.input_shape) > 1:
            self._reshaped_model = lambda x: self.model(x.reshape(x.shape[0], *self.input_shape))
        else:
            self._reshaped_model = self.model

        # if we don't have a dynamic clustering algorithm then can precowe mpute
        # a lot of information
        if not callable(self.masker.partition_tree):
            self.create_cluster_matrices(self.masker.partition_tree)

    def __call__(self, X, y=None, max_evals=1000, output_indexes=None,
                 batch_size=10, hierarchical=None, silent=False):
        """ Explain the model's prediction on the given set of instances.

        Parameters
        ----------
        X : numpy.array or pandas.DataFrame or [TODO] pytorch or TF array
            An array of samples for which to compute explanations.

        y : numpy.array or pandas.DataFrame or [TODO] pytorch or TF array
            An optional parallel array to X that provides the labels of the samples.

        npartitions : int
            How many times we partition the data, and hence evaluate the underlying model. More partitions lead to more
            fine-grained explanations, but require more run-time.

        hierarchical : None (default) or bool
            Whether to leave all interaction effects on the internal nodes of the partition tree, or to instead
            use Shapley values to split them up among lower nodes. By default this behavior is chosen
            automatically by the masker object type, but this can be overridden by setting to True of False directly.
        """

        npartitions = max_evals // 2
        
        # convert dataframes
        feature_names = None
        if safe_isinstance(X, "pandas.core.frame.DataFrame"):
            feature_names = list(X.columns)
            X = X.values
        elif safe_isinstance(X, "pandas.core.series.Series"):
            X = X.values

        # get hierarchical setting
        if hierarchical is None:
            hierarchical = getattr(self.masker, "hierarchical", False)
            
        # convert strings
        if type(X[0]) is str:
            #X_str = X
            tokenized = [self.masker.tokenize(s) for s in X]
            X = np.array([t["input_ids"] for t in tokenized])
            #offsets = [t["offset_mapping"] for t in tokenized]
        
        # handle list inputs that are not multiple inputs by auto-wrapping them as np.arrays
        if type(X) is list and not hasattr(X[0], "__len__"):
            X = np.array(X)
        elif type(X) is np.array and type(X[0]) is list:
            X = np.array([np.array(v) for v in X])
        
        # check we got just a single instance (TODO: use the masker to make this check robust to multi-dim inputs)
        # single_instance = False
        # if not hasattr(X[0], "__len__"):
        #     single_instance = True
        #     X = X.reshape(1,-1)
            
        # handle higher dimensional tensor inputs
        orig_shape = X.shape[1:]
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
            self._reshaped_model = lambda X: self.model(X.reshape(X.shape[0], *orig_shape))
        else:
            self._reshaped_model = self.model
            
        out = []
        out_inds = []
        pbar = None
        start_time = time.time()
        for i in range(X.shape[0]):
            row_out,oinds = self.explain(X[i], hierarchical, npartitions, output_indexes, 0, batch_size, silent)
            out.append(row_out)
            if output_indexes is not None:
                out_inds.append(oinds)
            if pbar is None and time.time() - start_time > 5:
                pbar = tqdm(total=X.shape[0], disable=silent, leave=False)
                pbar.update(i)
            if pbar is not None:
                pbar.update(1)
        if pbar is not None:
            pbar.close()
        out = np.array(out)
        out_inds = np.array(out_inds)
        out_len = 0 if self.output_shape == tuple() else self.output_shape[0]
        if not hierarchical:
            if out_len == 0:
                out = out.reshape(out.shape[0], *orig_shape)
            else:
                out = out.reshape(out.shape[0], *orig_shape, out_len)
        
        # the output shape
#         out_len = 0 if self.output_shape == tulpe() else out_line
#         if self.multi_output:
#             out_len = len(self.curr_expected_value) if output_indexes is None else output_indexes_len(output_indexes)
#         output_shape = tuple(range(out_len))
        
        if output_indexes is not None:
            output_names = [[self.output_names[j] for j in out_inds[i]] for i in range(out_len)]
        else:
            output_names = self.output_names
        
        return Explanation(
            self.expected_value, out, X, input_names=feature_names, output_shape=self.output_shape,
            output_indexes=out_inds, output_names=output_names
        )
        
#         if self.multi_output:
#             n = np.prod(orig_shape)
#             out_len = len(self.curr_expected_value) if output_indexes is None else output_indexes_len(output_indexes)
#             out = out.reshape(out.shape[0], np.prod(orig_shape), out_len)
#             out = [out[:,:,i].reshape(out.shape[0], *orig_shape) for i in range(out_len)]
                
#             if output_indexes is not None:
#                 return out, out_inds
#             else:
#                 return out
#         else:
#             out = out.reshape(out.shape[0], *orig_shape)
#             if single_instance:
#                 return out[0]
#             else:
#                 return out
            
            
            
            
            
            
            
            
            
#         # convert strings
#         if type(X[0]) is str:
#             X_str = X
#             tokenized = [self.masker.tokenize(s) for s in X]
#             X = np.array([t["input_ids"] for t in tokenized])
#             offsets = [t["offset_mapping"] for t in tokenized]
            
#         if hierarchical:
#             v = self.hierarchical_shap_values(
#                 X, npartitions=npartitions, interaction_tolerance=interaction_tolerance,
#                 batch_size=batch_size, output_indexes=output_indexes
#             )
#             out_inds = None
#             if output_indexes is not None:
#                 out_inds = v[1]
#                 v = v[0]
#             output_shape = tuple()
#             if type(v) is list:
#                 output_shape = (len(v),)
#                 v = np.stack(v, axis=-1) # put outputs at the end
#             output_names = None
#             if out_inds is not None:
#                 output_names = [[self.output_names[j] for j in out_inds[i]] for i in range(v.shape[0])]
#             e = Explanation(
#                 self.expected_value, v, X, input_names=feature_names, output_shape=output_shape,
#                 output_indexes=out_inds, output_names=output_names
#             )
#         else:
#             v = self.shap_values(
#                 X, npartitions=npartitions, interaction_tolerance=interaction_tolerance,
#                 batch_size=batch_size, output_indexes=output_indexes
#             )
#             out_inds = None
#             if output_indexes is not None:
#                 out_inds = v[1]
#                 v = v[0]
#             output_shape = tuple()
#             if type(v) is list:
#                 output_shape = (len(v),)
#                 v = np.stack(v, axis=-1) # put outputs at the end
#             output_names = None
#             if out_inds is not None:
#                 output_names = [[self.output_names[j] for j in out_inds[i]] for i in range(v.shape[0])]
#             e = Explanation(
#                 self.expected_value, v, X, input_names=feature_names, output_shape=output_shape,
#                 output_indexes=out_inds, output_names=output_names
#             )
# #         else:
# #             assert False, "hierarchical not implemented yet for __call__... simple fix TBD."
#         return e

    def create_cluster_matrices(self, partition_tree):
        """ Build clustering dependent reuseable variables.
        """
        
        self.mask_matrix = make_masks(partition_tree)
        self.merge_clusters = -np.ones((2 * partition_tree.shape[0] + 1, 3), dtype=np.int64)
        self.merge_clusters[partition_tree.shape[0] + 1:,:2] = partition_tree[:,:2]
        for i in range(self.merge_clusters.shape[0]):
            if self.merge_clusters[i,0] < 0:
                self.merge_clusters[i,2] = 1
            else:
                self.merge_clusters[i,2] = self.merge_clusters[self.merge_clusters[i,0],2] + self.merge_clusters[self.merge_clusters[i,1],2]
    
    def hierarchical_shap_values(self, x, npartitions=100, output_indexes=None,
                                 context=0, batch_size=10, silent=False):
        
        # convert dataframes
        if safe_isinstance(x, "pandas.core.series.Series"):
            x = x.values
        elif safe_isinstance(x, "pandas.core.frame.DataFrame"):
            x = x.values
            
        # convert strings
        if type(x[0]) is str:
            x_str = x
            tokenized = [self.masker.tokenize(s) for s in x]
            x = np.array([np.array(t["input_ids"]) for t in tokenized])
            #offsets = [t["offset_mapping"] for t in tokenized]

        # handle list inputs that are not multiple inputs by auto-wrapping them as np.arrays
        if type(x) is list and not hasattr(x[0], "__len__"):
            x = np.array(x)

        single_instance = False
        if not hasattr(x[0], "__len__"):
            single_instance = True
            x = x.reshape(1, -1)
        
        # handle higher dimensional tensor inputs
        if hasattr(x[0][0], "__len__"):
            orig_shape = x[0].shape
            x = x.reshape(len(x), -1)
            self._reshaped_model = lambda x: self.model(x.reshape(len(x), *orig_shape))
        else:
            self._reshaped_model = self.model
            orig_shape = None

        out = []
        out_inds = []
        pbar = None
        start_time = time.time()
        for i in range(x.shape[0]):
            row_out,oinds = self.explain(
                x[i], "hierarchical", npartitions, output_indexes, -1, context, batch_size, silent
            )
            out.append(row_out)
            if output_indexes is not None:
                out_inds.append(oinds)
            if pbar is None and time.time() - start_time > 5:
                pbar = tqdm(total=x.shape[0], disable=silent, leave=False)
                pbar.update(i)
            if pbar is not None:
                pbar.update(1)
        if pbar is not None:
            pbar.close()
        out = np.array(out)
        out_inds = np.array(out_inds)

        if single_instance:
            return out[0]
        else:
            return out
        
    def shap_values(self, x, npartitions=100, interaction_tolerance=0, output_indexes=None,
                    batch_size=10, silent=False):
        
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
            
        # handle higher dimensional tensor inputs
        orig_shape = x.shape[1:]
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
            self._reshaped_model = lambda x: self.model(x.reshape(x.shape[0], *orig_shape))
        else:
            self._reshaped_model = self.model
            
        out = []
        out_inds = []
        pbar = None
        start_time = time.time()
        for i in range(x.shape[0]):
            row_out,oinds = self.explain(
                x[i], "marginal", npartitions, output_indexes,
                interaction_tolerance, 0, batch_size, silent
            )
            out.append(row_out)
            if output_indexes is not None:
                out_inds.append(oinds)
            if pbar is None and time.time() - start_time > 5:
                pbar = tqdm(total=x.shape[0], disable=silent, leave=False)
                pbar.update(i)
            if pbar is not None:
                pbar.update(1)
        if pbar is not None:
            pbar.close()
        out = np.array(out)
        out_inds = np.array(out_inds)
        
        if self.multi_output:
            n = np.prod(orig_shape)
            out_len = len(self.curr_expected_value) if output_indexes is None else output_indexes_len(output_indexes)
            out = out.reshape((out.shape[0], np.prod(orig_shape), out_len))
            out = [out[:,:,i].reshape(out.shape[0], *orig_shape) for i in range(out_len)]
                
            if output_indexes is not None:
                return out, out_inds
            else:
                return out
        else:
            out = out.reshape(out.shape[0], *orig_shape)
            if single_instance:
                return out[0]
            else:
                return out

    def explain(self, x, hierarchical, npartitions, output_indexes, context, batch_size, silent):
        
        if getattr(self.masker, "variable_background", False) or getattr(self, "curr_expected_value", None) is None:
            self.eval_time = time.time()
            self.curr_expected_value = self._reshaped_model(self.masker(np.zeros(x.shape, dtype=np.bool), x)).mean(0)
            self.eval_time = time.time() - self.eval_time
            if hasattr(self.curr_expected_value, 'shape') and len(self.curr_expected_value.shape) > 0:
                self.multi_output = True
                out_len = len(self.curr_expected_value) if output_indexes is None else output_indexes_len(output_indexes)
                self.output_shape = (out_len,)
            else:
                self.multi_output = False
                self.output_shape = tuple()
            
            if not getattr(self.masker, "variable_background", False):
                self.expected_value = self.curr_expected_value
            
        
        if callable(self.masker.partition_tree):
            self.create_cluster_matrices(self.masker.partition_tree(x))
        
        # allocate space for our outputs
        if self.multi_output:
            if output_indexes is not None:
                out_shape = (self.merge_clusters.shape[0], output_indexes_len(output_indexes))
            else:
                out_shape = (self.merge_clusters.shape[0], len(self.curr_expected_value))
        else:
            out_shape = (self.merge_clusters.shape[0],)
        self.values = np.zeros(out_shape)
        self.dvalues = np.zeros(out_shape)
        oinds = self.owen(x, npartitions, output_indexes, hierarchical, context, batch_size, silent)

        if hierarchical:
            if self.multi_output:
                return [self.dvalues[:,i] for i in range(self.dvalues.shape[1])], oinds
            else:
                return self.dvalues.copy(), oinds   
        else:
            # drop the interaction terms down onto self.values
            self.values[:] = self.dvalues
            M = len(x)
            def lower_credit(i, value=0):
                if i < M:
                    self.values[i] += value
                    return
                li = self.merge_clusters[i,0]
                ri = self.merge_clusters[i,1]
                group_size = self.merge_clusters[i,2]
                lsize = self.merge_clusters[li,2]
                rsize = self.merge_clusters[ri,2]
                assert lsize+rsize == group_size
                self.values[i] += value
                lower_credit(li, self.values[i] * lsize / group_size)
                lower_credit(ri, self.values[i] * rsize / group_size)
            lower_credit(len(self.dvalues) - 1)
            
            return self.values[:len(x)].copy(), oinds
        
    def single_context_owen(self, x, npartitions, output_indexes, interaction_tolerance, context, silent):
        """ Compute a nested set of recursive Owen values.
        """
        
        f = self._reshaped_model
        r = self.masker
        m00 = np.zeros(self.mask_matrix.shape[1], dtype=np.bool)
        f00 = self.curr_expected_value
        f11 = self._reshaped_model(r(~m00, x)).mean(0)
        ind = len(self.values)-1

        if context == 1:
            tmp = f00
            f00 = f11
            f11 = tmp

        q = queue.PriorityQueue()
        q.put((0, (m00, f00, f11, ind)))
        eval_count = 0
        total_evals = min(npartitions, (len(x)-1)*len(x))
        pbar = None
        start_time = time.time()
        while not q.empty():

            # if we passed our execution limit then leave everything else on the internal nodes
            if eval_count > npartitions:
                while not q.empty():
                    m00, f00, f11, ind = q.get()[1]
                    self.dvalues[ind] = f11 - f00
                break

            # get our next set of arguments
            m00, f00, f11, ind = q.get()[1]

            # get the left are right children of this cluster
            lind = self.merge_clusters[ind, 0]
            rind = self.merge_clusters[ind, 1]

            # check if we are a leaf node or terminated our decent early and dumping credit at an internal node
            if lind < 0:
                self.dvalues[ind] = f11 - f00
                continue

            # build the masks
            m10 = m00.copy() # we separate the copy from the add so as to not get converted to a matrix
            m10[:] += self.mask_matrix[lind, :]
            m01 = m00.copy()
            m01[:] += self.mask_matrix[rind, :]

            # evaluate the model on the two new masked inputs
            if context == 0:
                # assert r(m10, x)[0][0] == 101
                # assert r(m01, x)[0][0] == 101
                f10 = f(r(m10, x)).mean(0)
                f01 = f(r(m01, x)).mean(0)
            else:
                f10 = f(r(~m10, x)).mean(0)
                f01 = f(r(~m01, x)).mean(0)
            
            # update our progress indicator
            eval_count += 2
            if pbar is None and time.time() - start_time > 5:
                pbar = tqdm(total=total_evals, disable=silent, leave=False)
                pbar.update(eval_count)
            if pbar is not None:
                pbar.update(2)

            # update our dividends
            self.dvalues[ind] += f11 - f01 - f10 + f00

            # recurse on the left node
            args = (m00, f00, f10, lind)
            q.put((-abs(f10 - f00), args)) #/ np.sqrt(m10.sum()

            # recurse on the right node
            args = (m00, f00, f01, rind)
            q.put((-abs(f01 - f00), args))
        
        if pbar is not None:
            pbar.close()

        if context == 1:
            self.dvalues *= -1

        return None
    
    def owen(self, x, npartitions, output_indexes, interaction_tolerance, context, batch_size, silent):
        """ Compute a nested set of recursive Owen values based on an ordering recursion of paried permutations.

        TODO: This is not done yet, but when it is done it should outperform our current solution of just using
        the PermutationExplainer for tabular data.
        """
        
        f = self._reshaped_model
        r = self.masker
        m00 = np.zeros(self.mask_matrix.shape[1], dtype=np.bool)
        f00 = self.curr_expected_value
        f11 = self._reshaped_model(r(~m00, x)).mean(0)
        ind = len(self.values)-1
        scaling = np.ones(4*npartitions)
        sparent = np.ones(4*npartitions, dtype=np.int) * -1
        scount = 0
        required_evals = 0

        # make sure output_indexes is a list of indexes
        if output_indexes is not None:
            assert self.multi_output, "output_indexes is only valid for multi-output models!"
            
            out_len = output_indexes_len(output_indexes)
            if output_indexes.startswith("max("):
                output_indexes = np.argsort(-f11)[:out_len]
            elif output_indexes.startswith("min("):
                output_indexes = np.argsort(f11)[:out_len]
            elif output_indexes.startswith("max(abs("):
                output_indexes = np.argsort(np.abs(f11))[:out_len]
        
            f00 = f00[output_indexes]
            f11 = f11[output_indexes]
        
        q = queue.PriorityQueue()
        q.put((0, 0, 0, (m00, f00, f11, ind, scount, -1, True, self.merge_clusters[-1, 2], 0), 1.0))
        
        scount += 1
        eval_count = 0
        total_evals = min(npartitions, (len(x)-1)*len(x)) # TODO: (len(x)-1)*len(x) is only right for balanced partition trees
        pbar = None
        start_time = time.time()
        done_list = []
        skip_next_reverse = False
        while not q.empty():

            # if we passed our execution limit then leave everything else on the internal nodes
            # if eval_count >= npartitions:
            #     while not q.empty():
            #         done_list.append(q.get())
            #         # m00, f00, f11, ind, sind, twin_ind = q.get()[2]
            #         # self.dvalues[ind] += (f11 - f00) * compute_weight(sind, sparent, scaling)
            #     break

            # create a batch of work to do
            batch_args = []
            batch_data = []
            batch_positions = []
            batch_pos = 0
            will_eval = 0
            while not q.empty() and len(batch_data) < batch_size:# and eval_count <= npartitions:
                
                # get our next set of arguments
                parts = q.get()
                m00, f00, f11, ind, sind, twin_ind, forward, eval_cost, depth = parts[3]
                prev_weight = parts[4]
                curr_weight = compute_weight(sind, sparent, scaling)

                # if our weight has changed since we last added this node then we put the node back on the
                # queue with an updated weight and then start over
                if abs(prev_weight - curr_weight) > 1e-5:
                    q.put((parts[0], parts[1] * curr_weight / prev_weight, parts[2], parts[3], curr_weight))
                    continue

                # get the left are right children of this cluster
                lind = self.merge_clusters[ind, 0]
                rind = self.merge_clusters[ind, 1]

                # if we shouldn't be expanding reverse nodes anymore then we skip all of them
                if parts[0] == 1 and skip_next_reverse:
                    skip_next_reverse = False
                    continue
                
                if eval_cost > 0:
                    if npartitions - required_evals < eval_cost:
                        skip_next_reverse = True # we skip the next reverse node since it is coupled with us
                        continue
                    required_evals += eval_cost
                
                

                # if we have a twin, then we lower that twin's weight since it now needs to share it with us :)
                
                if twin_ind >= 0:
                    #print(npartitions, required_evals, eval_cost, ind, forward)
                    # cluster_size = self.merge_clusters[ind, 2]
                    # evals_left = npartitions - required_evals
                    # print("cs", cluster_size, evals_left, ind, self.merge_clusters.shape[0])
                    # if evals_left < cluster_size or not (ind == self.merge_clusters[-1,0] or ind == self.merge_clusters[-1,1]):
                    #     continue # we don't expand reverse orderings we can't finish...
                    # else:
                    #     print("ASDFASDF", required_evals, cluster_size)
                    scaling[twin_ind] /= 2

                # print()
                # print("run", ind, "forward" if forward else "reverse", parts[0], twin_ind, lind, rind, self.merge_clusters[-1,:])
                # print(m00 * 1)
                # print(m00 + self.mask_matrix[ind, :].toarray()[0] * 1)
                # print(npartitions, required_evals, eval_cost)

                
                
                
                # check if we are a leaf node and so terminating our decent
                if lind < 0:
                    done_list.append(parts)
                    continue

                # build the masks
                m10 = m00.copy() # we separate the copy from the add so as to not get converted to a matrix
                m10[:] += self.mask_matrix[lind, :]
                m01 = m00.copy()
                m01[:] += self.mask_matrix[rind, :]
                
                # print(m10 * 1, twin_ind)
                # print(m01 * 1, twin_ind)

                
                batch_args.append((m00, m10, m01, f00, f11, ind, lind, rind, sind, twin_ind, forward, eval_cost, depth))
                
                d = r(m10, x)
                batch_data.append(d)
                batch_positions.append(batch_pos)
                batch_pos += d.shape[0]
                will_eval += 1
                
                d = r(m01, x)
                batch_data.append(d)
                batch_positions.append(batch_pos)
                batch_pos += d.shape[0]
                will_eval += 1
                
            batch_positions.append(batch_pos)
                
            # run the batch
            if len(batch_args) > 0:
                if safe_isinstance(batch_data[0], "pandas.core.frame.DataFrame"):
                    fout = f(pd.concat(batch_data, axis=0))
                    #print(pd.concat(batch_data, axis=0))
                else:
                    fout = f(np.concatenate(batch_data, axis=0))
                    #print(np.concatenate(batch_data, axis=0))
                if output_indexes is not None:
                    fout = fout[:,output_indexes]
                    
                eval_count += len(batch_data)
                    
                if pbar is None and time.time() - start_time > 5:
                    pbar = tqdm(total=total_evals, disable=silent, leave=False)
                    pbar.update(eval_count)
                if pbar is not None:
                    pbar.update(len(batch_data))
            
            # use the results of the batch to add new nodes
            for i in range(len(batch_args)):
                
                m00, m10, m01, f00, f11, ind, lind, rind, sind, twin_ind, forward, eval_cost, depth = batch_args[i]
                
                # evaluate the model on the two new masked inputs
                f10 = fout[batch_positions[2*i]:batch_positions[2*i+1]].mean(0)
                f01 = fout[batch_positions[2*i+1]:batch_positions[2*i+2]].mean(0)

                weight = compute_weight(sind, sparent, scaling)
                lsize = self.merge_clusters[lind, 2]
                rsize = self.merge_clusters[rind, 2]

                if forward:

                    # recurse on the left node with zero context
                    args = (m00, f00, f10, lind, scount, -1, True, 0, depth+1)
                    q.put((0, -np.max(np.abs(f10 - f00)) / 2**(depth+1), scount, args, weight))
                    sparent[scount] = sind
                    scaling[scount] = 1.0
                    scount += 1

                    # recurse on the right node with one context
                    args = (m10, f10, f11, rind, scount, -1, True, 0, depth+1)
                    q.put((0, -np.max(np.abs(f11 - f10)) / 2**(depth+1), scount, args, weight))
                    sparent[scount] = sind
                    scaling[scount] = 1.0
                    scount += 1

                    # recurse on the right node with zero context
                    # this is the node that when it gets pulled off "pays" for the other nodes
                    args = (m00, f00, f01, rind, scount, scount-1, False, lsize+rsize, depth+1) # NOTE: the fourth to last arg is the scaling index of the other node that needs its weight reduced when we pull this one off
                    interaction_value = 1#f11 - f10 - f01 + f00
                    q.put((1, -np.max(np.abs(interaction_value)) / 2**(depth+1), scount, args, weight))
                    sparent[scount] = sind
                    scaling[scount] = 0.5
                    scount += 1

                    # recurse on the left node with one context
                    args = (m01, f01, f11, lind, scount, scount-3, False, 0, depth+1) 
                    q.put((1, -np.max(np.abs(interaction_value)) / 2**(depth+1), scount, args, weight))
                    sparent[scount] = sind
                    scaling[scount] = 0.5
                    scount += 1

                else:
                    
                    # recurse on the right node with zero context
                    args = (m00, f00, f01, rind, scount, -1, False, 0, depth+1)
                    q.put((0, -np.max(np.abs(f01 - f00)) / 2**(depth+1), scount, args, weight))
                    sparent[scount] = sind
                    scaling[scount] = 1.0
                    scount += 1

                    # recurse on the left node with one context
                    args = (m01, f01, f11, lind, scount, -1, False, 0, depth+1) # NOTE: the last arg is the scaling index of the other node that needs its weight reduced when we pull this one off
                    q.put((0, -np.max(np.abs(f10 - f00)) / 2**(depth+1), scount, args, weight))
                    sparent[scount] = sind
                    scaling[scount] = 1.0
                    scount += 1

                    # recurse on the left node with zero context
                    # this is the node that when it gets pulled off "pays" for the other nodes
                    args = (m00, f00, f10, lind, scount, scount - 1, True, lsize+rsize, depth+1)
                    interaction_value = 1#f11 - f10 - f01 + f00
                    q.put((1, -np.max(np.abs(interaction_value)) / 2**(depth+1), scount, args, weight))
                    sparent[scount] = sind
                    scaling[scount] = 0.5
                    scount += 1

                    # recurse on the right node with one context
                    args = (m10, f10, f11, rind, scount, scount - 3, True, 0, depth+1)
                    q.put((1, -np.max(np.abs(interaction_value)) / 2**(depth+1), scount, args, weight))
                    sparent[scount] = sind
                    scaling[scount] = 0.5
                    scount += 1
        
        # finish any undone twin scaling
        # for parts in done_list:
        #     twin_ind = parts[2][5]
        #     if twin_ind >= 0:
        #         scaling[twin_ind] /= 2
        
        # dump the values from the list of done nodes into dvalues
        count = 0
        for parts in done_list:
            m00, f00, f11, ind, sind, twin_ind, forward, eval_cost, depth = parts[3]
            lind = self.merge_clusters[ind, 0]
            if twin_ind == -1 or lind < 0:
                # if lind >= 0:
                #     print("internal!!", ind)
                # else:
                #     print(ind)
                # print(ind, twin_ind, f00, f11, sind)
                # print(m00 + self.mask_matrix[ind, :] + 0)
                # print("-")
                # print(m00.reshape(1,-1) + 0)
                # print()
                self.dvalues[ind] += (f11 - f00) * compute_weight(sind, sparent, scaling)
                count += 1
            else:
                pass#self.dvalues[ind] += (f11 - f00) * compute_weight(sind, sparent, scaling)
        # print("count", count)
        if pbar is not None:
            pbar.close()
        
        return output_indexes



    def owen3(self, x, npartitions, output_indexes, hierarchical, context, batch_size, silent):
        """ Compute a nested set of recursive Owen values based on an ordering recursion.
        """
        
        f = self._reshaped_model
        r = self.masker
        m00 = np.zeros(self.mask_matrix.shape[1], dtype=np.bool)
        f00 = self.curr_expected_value
        f11 = self._reshaped_model(r(~m00, x)).mean(0)
        ind = len(self.values)-1

        # make sure output_indexes is a list of indexes
        if output_indexes is not None:
            assert self.multi_output, "output_indexes is only valid for multi-output models!"
            
            out_len = output_indexes_len(output_indexes)
            if output_indexes.startswith("max("):
                output_indexes = np.argsort(-f11)[:out_len]
            elif output_indexes.startswith("min("):
                output_indexes = np.argsort(f11)[:out_len]
            elif output_indexes.startswith("max(abs("):
                output_indexes = np.argsort(np.abs(f11))[:out_len]
        
            f00 = f00[output_indexes]
            f11 = f11[output_indexes]
        
        q = queue.PriorityQueue()
        q.put((0, 0, (m00, f00, f11, ind, 1.0)))
        eval_count = 0
        total_evals = min(npartitions, (len(x)-1)*len(x)) # TODO: (len(x)-1)*len(x) is only right for balanced partition trees
        pbar = None
        start_time = time.time()
        while not q.empty():

            # if we passed our execution limit then leave everything else on the internal nodes
            if eval_count >= npartitions:
                while not q.empty():
                    m00, f00, f11, ind, weight = q.get()[2]
                    self.dvalues[ind] += (f11 - f00) * weight
                break

            # create a batch of work to do
            batch_args = []
            batch_data = []
            batch_positions = []
            batch_pos = 0
            while not q.empty() and len(batch_data) < batch_size and eval_count < npartitions:
                
                # get our next set of arguments
                m00, f00, f11, ind, weight = q.get()[2]

                # get the left are right children of this cluster
                lind = self.merge_clusters[ind, 0]
                rind = self.merge_clusters[ind, 1]

                # check if we are a leaf node and so terminating our decent
                if lind < 0:
                    self.dvalues[ind] += (f11 - f00) * weight
                    continue

                # build the masks
                m10 = m00.copy() # we separate the copy from the add so as to not get converted to a matrix
                m10[:] += self.mask_matrix[lind, :]
                m01 = m00.copy()
                m01[:] += self.mask_matrix[rind, :]
                
                batch_args.append((m00, m10, m01, f00, f11, ind, lind, rind, weight))
                
                d = r(m10, x)
                batch_data.append(d)
                batch_positions.append(batch_pos)
                batch_pos += d.shape[0]
                
                d = r(m01, x)
                batch_data.append(d)
                batch_positions.append(batch_pos)
                batch_pos += d.shape[0]
                
            batch_positions.append(batch_pos)
                
            # run the batch
            if len(batch_args) > 0:
                if safe_isinstance(batch_data[0], "pandas.core.frame.DataFrame"):
                    fout = f(pd.concat(batch_data, axis=0))
                else:
                    fout = f(np.concatenate(batch_data, axis=0))
                if output_indexes is not None:
                    fout = fout[:,output_indexes]
                    
                eval_count += len(batch_data)
                    
                if pbar is None and time.time() - start_time > 5:
                    pbar = tqdm(total=total_evals, disable=silent, leave=False)
                    pbar.update(eval_count)
                if pbar is not None:
                    pbar.update(len(batch_data))
            
            # use the results of the batch to add new nodes
            for i in range(len(batch_args)):
                
                m00, m10, m01, f00, f11, ind, lind, rind, weight = batch_args[i]
                
                # evaluate the model on the two new masked inputs
                f10 = fout[batch_positions[2*i]:batch_positions[2*i+1]].mean(0)
                f01 = fout[batch_positions[2*i+1]:batch_positions[2*i+2]].mean(0)

                new_weight = weight
                if not hierarchical:
                    new_weight /= 2
                else:
                    self.dvalues[ind] += (f11 - f10 - f01 + f00) * weight # leave the interaction effect on the internal node

                # recurse on the left node with zero context
                args = (m00, f00, f10, lind, new_weight)
                q.put((-np.max(np.abs(f10 - f00)) * new_weight, np.random.randn(), args))

                # recurse on the right node with zero context
                args = (m00, f00, f01, rind, new_weight)
                q.put((-np.max(np.abs(f01 - f00)) * new_weight, np.random.randn(), args))

                if not hierarchical:
                    # recurse on the left node with one context
                    args = (m01, f01, f11, lind, new_weight)
                    q.put((-np.max(np.abs(f11 - f01)) * new_weight, np.random.randn(), args))

                    # recurse on the right node with one context
                    args = (m10, f10, f11, rind, new_weight)
                    q.put((-np.max(np.abs(f11 - f10)) * new_weight, np.random.randn(), args))
        if pbar is not None:
            pbar.close()
        
        return output_indexes
    
def compute_weight(sind, sparent, scaling):
    weight = 1.0
    while sind >= 0:
        weight *= scaling[sind]
        sind = sparent[sind]
    return weight

def output_indexes_len(output_indexes):
    if output_indexes.startswith("max("):
        return int(output_indexes[4:-1])
    elif output_indexes.startswith("min("):
        return int(output_indexes[4:-1])
    elif output_indexes.startswith("max(abs("):
        return int(output_indexes[8:-2])
    elif type(output_indexes) is not str:
        return len(output_indexes)
    
def rec_fill_masks(mask_matrix, cluster_matrix, M, ind=None):
    if ind is None:
        ind = cluster_matrix.shape[0] - 1 + M
        
    if ind < M:
        mask_matrix[ind] = np.array([ind])
        return

    lind = int(cluster_matrix[ind-M,0])
    rind = int(cluster_matrix[ind-M,1])
    
    rec_fill_masks(mask_matrix, cluster_matrix, M, lind)
    mask_matrix[ind] = mask_matrix[lind]

    rec_fill_masks(mask_matrix, cluster_matrix, M, rind)
    mask_matrix[ind] = np.concatenate((mask_matrix[ind], mask_matrix[rind]))

# import scipy.sparse
# def make_masks(cluster_matrix):

#     # build the mask matrix recursively as an array of index lists
#     global count
#     count = 0
#     M = cluster_matrix.shape[0] + 1
#     mask_matrix_inds = np.zeros(2 * M - 1, dtype=np.object)
#     rec_fill_masks(mask_matrix_inds, cluster_matrix, M)
    
#     # convert the array of index lists into CSR format
#     indptr = np.zeros(len(mask_matrix_inds) + 1, dtype=np.int)
#     indices = np.zeros(np.sum([len(v) for v in mask_matrix_inds]), dtype=np.int)
#     pos = 0
#     for i in range(len(mask_matrix_inds)):
#         inds = mask_matrix_inds[i]
#         indices[pos:pos+len(inds)] = inds
#         pos += len(inds)
#         indptr[i+1] = pos
#     mask_matrix = scipy.sparse.csr_matrix(
#         (np.ones(len(indices), dtype=np.bool), indices, indptr),
#         shape=(len(mask_matrix_inds), M)
#     )

#     return mask_matrix
import pandas as pd
import scipy as sp
import numpy as np
import warnings
import time
from tqdm.auto import tqdm
import queue
from ..common import assert_import, record_import_error, safe_isinstance

try:
    import cv2
except ImportError as e:
    record_import_error("cv2", "cv2 could not be imported!", e)


class PartitionExplainer():
    
    def __init__(self, model, masker, partition_tree):
        """ Uses the Partition SHAP method to explain the output of any function.

        Partition SHAP computes Shapley values recursively through a hierarchy of features, this
        hierarchy defines feature coalitions and results in the Owen values from game theory. The
        PartitionExplainer has two particularly nice properties: 1) PartitionExplainer is
        model-agnostic but when using a balanced partition tree only has quadradic exact runtime 
        (in term of the number of input features). This is in contrast to the exponential exact
        runtime of KernalExplainer or SamplingExplainer. 2) PartitionExplainer always assigns to groups of
        correlated features the credit that set of features would have had when treated as a group. This
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
            computes the output of the model for those samples.

        masker : function or numpy.array or pandas.DataFrame
            The function used to "mask" out hidden features of the form `masker(x, mask)`. It takes a
            single input sample and a binary mask and returns a matrix of masked samples. These
            masked samples will then be evaluated using the model function and the outputs averaged.
            As a shortcut for the standard masking using by SHAP you can pass a background data matrix
            instead of a function and that matrix will be used for masking. Domain specific masking
            functions are available in shap such as shap.ImageMasker for images and shap.TokenMasker
            for text.

        partition_tree : function or numpy.array
            A hierarchical clustering of the input features represented by a matrix that follows the format
            used by scipy.cluster.hierarchy (see the notebooks/partition_explainer directory an example).
            If this is a function then the function produces a clustering matrix when given a single input
            example. If you are using a standard SHAP masker object then you can pass masker.partition_tree
            to use that masker's built-in clustering of the features.
        """

        warnings.warn("PartitionExplainer is still in an alpha state, so use with caution...")
        
        # convert dataframes
        if safe_isinstance(masker, "pandas.core.series.Series"):
            masker = masker.values
        elif safe_isinstance(masker, "pandas.core.frame.DataFrame"):
            masker = masker.values

        # If the user just gave a dataset as the masker
        # then we make a masker that perturbs features independently
        self.input_shape = masker.shape[1:] if hasattr(masker, "shape") else None
        if type(masker) == np.ndarray:
            self.masker_data = masker
            self.masker = lambda x, mask: x * mask + self.masker_data * np.invert(mask)
        else:
            self.masker = masker

        self.model = lambda x: np.array(model(x))
        self.expected_value = None
        self.partition_tree = partition_tree
        
        # handle higher dimensional tensor inputs
        if self.input_shape is not None and len(self.input_shape) > 1:
            self._reshaped_model = lambda x: self.model(x.reshape(x.shape[0], *self.input_shape))
        else:
            self._reshaped_model = self.model

        # if we don't have a dynamic clustering algorithm then we can precompute
        # a lot of information
        if not callable(self.partition_tree):
            self.create_cluster_matrices(self.partition_tree)

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
    
    def hierarchical_shap_values(self, x, nsamples=100, output_indexes=None,
                                 context=0, batch_size=10, silent=False):
        
        # convert dataframes
        if safe_isinstance(x, "pandas.core.series.Series"):
            x = x.values
        elif safe_isinstance(x, "pandas.core.frame.DataFrame"):
            x = x.values

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
                x[i], "hierarchical", nsamples, output_indexes, -1, context, batch_size, silent
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
        
    def shap_values(self, x, nsamples=100, interaction_tolerance=0, output_indexes=None,
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
                x[i], "marginal", nsamples, output_indexes,
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
            out = out.reshape(out.shape[0], np.prod(orig_shape), out_len)
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

    def explain(self, x, output_type, nsamples, output_indexes,
                interaction_tolerance, context, batch_size, silent):
        
        if self.masker.variable_background or self.curr_expected_value is None:
            self.eval_time = time.time()
            self.curr_expected_value = self._reshaped_model(self.masker(x, np.zeros(x.shape, dtype=np.bool))).mean(0)
            self.eval_time = time.time() - self.eval_time
            if hasattr(self.curr_expected_value, 'shape') and len(self.curr_expected_value.shape) > 0:
                self.multi_output = True
            else:
                self.multi_output = False
            
            if not self.masker.variable_background:
                self.expected_value = self.curr_expected_value
            
        
        if callable(self.partition_tree):
            self.create_cluster_matrices(self.partition_tree(x))
        
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

        if output_type == "hierarchical":
            oinds = self.single_context_owen(x, nsamples, output_indexes, interaction_tolerance, context, silent)
            
            if self.multi_output:
                return [self.dvalues[:,i] for i in range(self.dvalues.shape[1])], oinds
            else:
                return self.dvalues.copy(), oinds
                
        elif output_type == "marginal":
            oinds = self.owen(x, nsamples, output_indexes, interaction_tolerance, context, batch_size, silent)

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
                assert lsize+rsize == group_size, "Ah"
                self.values[i] += value
                lower_credit(li, self.values[i] * lsize / group_size)
                lower_credit(ri, self.values[i] * rsize / group_size)
            lower_credit(len(self.dvalues) - 1)
            
            return self.values[:len(x)].copy(), oinds
        
    def single_context_owen(self, x, nsamples, output_indexes, interaction_tolerance, context, silent):
        """ Compute a nested set of recursive Owen values.
        """
        
        f = self._reshaped_model
        r = self.masker
        m00 = np.zeros(self.mask_matrix.shape[1], dtype=np.bool)
        f00 = self.curr_expected_value
        f11 = self._reshaped_model(x.reshape(1,len(x)))[0]
        ind = len(self.values)-1

        if context == 1:
            tmp = f00
            f00 = f11
            f11 = tmp

        q = queue.PriorityQueue()
        q.put((0, (m00, f00, f11, ind)))
        eval_count = 0
        total_evals = min(nsamples, (len(x)-1)*len(x))
        pbar = None
        start_time = time.time()
        while not q.empty():

            # if we passed our execution limit then leave everything else on the internal nodes
            if eval_count > nsamples:
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
                assert r(x, m10)[0][0] == 101
                assert r(x, m01)[0][0] == 101
                f10 = f(r(x, m10)).mean(0)
                f01 = f(r(x, m01)).mean(0)
            else:
                f10 = f(r(x, ~m10)).mean(0)
                f01 = f(r(x, ~m01)).mean(0)
            
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
            print("closing pbar")
            pbar.close()

        if context == 1:
            self.dvalues *= -1

        return None
    
    def owen(self, x, nsamples, output_indexes, interaction_tolerance, context, batch_size, silent):
        """ Compute a nested set of recursive Owen values based on an ordering recursion.
        """
        
        f = self._reshaped_model
        r = self.masker
        m00 = np.zeros(self.mask_matrix.shape[1], dtype=np.bool)
        f00 = self.curr_expected_value
        f11 = self._reshaped_model(x.reshape(1,len(x)))[0]
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
        total_evals = min(nsamples, (len(x)-1)*len(x))
        pbar = None
        start_time = time.time()
        #pbar = tqdm(total=total_evals, disable=silent or self.eval_time * total_evals < 5, leave=False)
        while not q.empty():

            # if we passed our execution limit then leave everything else on the internal nodes
            if eval_count >= nsamples:
                while not q.empty():
                    m00, f00, f11, ind, weight = q.get()[2]
                    self.dvalues[ind] += (f11 - f00) * weight
                break

            # create a batch of work to do
            batch_args = []
            batch_data = []
            batch_positions = []
            batch_pos = 0
            while not q.empty() and len(batch_data) < batch_size and eval_count < nsamples:
                
                # get our next set of arguments
                m00, f00, f11, ind, weight = q.get()[2]

                # get the left are right children of this cluster
                lind = self.merge_clusters[ind, 0]
                rind = self.merge_clusters[ind, 1]

                # check if we are a leaf node or terminated our decent early and dumping credit at an internal node
                if lind < 0:
                    self.dvalues[ind] += (f11 - f00) * weight
                    continue

                # build the masks
                m10 = m00.copy() # we separate the copy from the add so as to not get converted to a matrix
                m10[:] += self.mask_matrix[lind, :]
                m01 = m00.copy()
                m01[:] += self.mask_matrix[rind, :]
                
                batch_args.append((m00, m10, m01, f00, f11, ind, lind, rind, weight))
                
                d = r(x, m10)
                batch_data.append(d)
                batch_positions.append(batch_pos)
                batch_pos += d.shape[0]
                
                d = r(x, m01)
                batch_data.append(d)
                batch_positions.append(batch_pos)
                batch_pos += d.shape[0]
                
            batch_positions.append(batch_pos)
                
            # run the batch
            if len(batch_args) > 0:
                fout = f(np.concatenate(batch_data, axis=0))
                if output_indexes is not None:
                    fout = fout[:,output_indexes]
                    
                eval_count += fout.shape[0]
                    
                if pbar is None and time.time() - start_time > 5:
                    pbar = tqdm(total=total_evals, disable=silent, leave=False)
                    pbar.update(eval_count)
                if pbar is not None:
                    pbar.update(fout.shape[0])
            
            # use the results of the batch to add new nodes
            for i in range(len(batch_args)):
                
                m00, m10, m01, f00, f11, ind, lind, rind, weight = batch_args[i]

                # evaluate the model on the two new masked inputs
                f10 = fout[batch_positions[2*i]:batch_positions[2*i+1]].mean(0)
                f01 = fout[batch_positions[2*i+1]:batch_positions[2*i+2]].mean(0)

                iratio_left = np.abs(((f10 - f00) - (f11 - f01)) / (np.abs(f10 - f00) + 1e-8))
                iratio_right = np.abs(((f11 - f10) - (f01 - f00)) / (np.abs(f11 - f10) + 1e-8))

                iratio = np.max([np.max(iratio_left), np.max(iratio_right)])

                new_weight = weight
                if iratio >= interaction_tolerance:
                    new_weight /= 2

                # recurse on the left node with zero context
                args = (m00, f00, f10, lind, new_weight)
                q.put((-np.max(np.abs(f10 - f00)) * new_weight, np.random.randn(), args))

                # recurse on the right node with one context
                args = (m10, f10, f11, rind, new_weight)
                q.put((-np.max(np.abs(f11 - f10)) * new_weight, np.random.randn(), args))

                if iratio >= interaction_tolerance:
                    # recurse on the right node with zero context
                    args = (m00, f00, f01, rind, new_weight)
                    q.put((-np.max(np.abs(f01 - f00)) * new_weight, np.random.randn(), args))

                    # recurse on the left node with one context
                    args = (m01, f01, f11, lind, new_weight)
                    q.put((-np.max(np.abs(f11 - f01)) * new_weight, np.random.randn(), args))
        if pbar is not None:
            pbar.close()
        
        return output_indexes

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

import scipy.sparse
def make_masks(cluster_matrix):

    # build the mask matrix recursively as an array of index lists
    global count
    count = 0
    M = cluster_matrix.shape[0] + 1
    mask_matrix_inds = np.zeros(2 * M - 1, dtype=np.object)
    rec_fill_masks(mask_matrix_inds, cluster_matrix, M)
    
    # convert the array of index lists into CSR format
    indptr = np.zeros(len(mask_matrix_inds) + 1, dtype=np.int)
    indices = np.zeros(np.sum([len(v) for v in mask_matrix_inds]), dtype=np.int)
    pos = 0
    for i in range(len(mask_matrix_inds)):
        inds = mask_matrix_inds[i]
        indices[pos:pos+len(inds)] = inds
        pos += len(inds)
        indptr[i+1] = pos
    mask_matrix = scipy.sparse.csr_matrix(
        (np.ones(len(indices), dtype=np.bool), indices, indptr),
        shape=(len(mask_matrix_inds), M)
    )

    return mask_matrix

    

class ImageMasker():
    def __init__(self, mask_value, shape=None):
        """ This masks out image regions according to the given tokenizer. 
        
        Parameters
        ----------
        mask_value : np.array, "blur(kernel_xsize, kernel_xsize)", "inpaint_telea", or "inpaint_ns"
            The value used to mask hidden regions of the image.

        shape : None or tuple
            If the mask_value is an auto-generated masker instead of a dataset then the input
            image shape needs to be provided.
        """
        if shape is None:
            if type(mask_value) is str:
                raise TypeError("When the mask_value is a string the shape parameter must be given!")
            self.shape = mask_value.shape
        else:
            self.shape = shape
            
        if issubclass(type(mask_value), np.ndarray):
            self.mask_value = mask_value.flatten()
        elif type(mask_value) is str:
            assert_import("cv2")
            self.mask_value = mask_value
            if mask_value.startswith("blur("):
                self.blur_kernel = tuple(map(int, mask_value[5:-1].split(",")))
        else:
            self.mask_value = np.ones(self.shape).flatten() * mask_value
        self.build_partition_tree()

        # note if this masker can use different background for different samples
        self.variable_background = type(self.mask_value) is str
        
        self.scratch_mask = np.zeros(self.shape[:-1], dtype=np.bool)
        self.last_xid = None
    
    def __call__(self, x, mask=None):

        # unwrap single element lists (which are how single input models look in multi-input format)
        if type(x) is list and len(x) == 1:
            x = x[0]
        
        # we preserve flattend inputs as flattened and full-shaped inputs as their original shape
        in_shape = x.shape
        if len(x.shape) > 1:
            x = x.flatten()
        
        # if not mask is given then we mask the whole image
        if mask is None:
            mask = np.zeros(np.prod(x.shape), dtype=np.bool)
            
        if type(self.mask_value) is str:
            if self.blur_kernel is not None:
                if self.last_xid != id(x):
                    self.blur_value = cv2.blur(x.reshape(self.shape), self.blur_kernel).flatten()
                    self.last_xid = id(x)
                out = x.copy()
                out[~mask] = self.blur_value[~mask]
                
            elif self.mask_value == "inpaint_telea":
                out = self.inpaint(x, ~mask, "INPAINT_TELEA")
            elif self.mask_value == "inpaint_ns":
                out = self.inpaint(x, ~mask, "INPAINT_NS")
        else:
            out = x.copy()
            out[~mask] = self.mask_value[~mask]

        return out.reshape(1, *in_shape)
        
    def blur(self, x, mask):
        cv2.blur()
        
    def inpaint(self, x, mask, method):
        reshaped_mask = mask.reshape(self.shape).astype(np.uint8).max(2)
        if reshaped_mask.sum() == np.prod(self.shape[:-1]):
            out = x.reshape(self.shape).copy()
            out[:] = out.mean((0,1))
            return out.flatten()
        else:
            return cv2.inpaint(
                x.reshape(self.shape).astype(np.uint8),
                reshaped_mask,
                inpaintRadius=3,
                flags=getattr(cv2, method)
            ).astype(x.dtype).flatten()

    def build_partition_tree(self):
        """ This partitions an image into a herarchical clustering based on axis-aligned splits.
        """
        
        xmin = 0
        xmax = self.shape[0]
        ymin = 0
        ymax = self.shape[1]
        zmin = 0
        zmax = self.shape[2]
        total_xwidth = xmax - xmin
        total_ywidth = ymax - ymin
        total_zwidth = zmax - zmin
        q = queue.PriorityQueue()
        M = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
        self.partition_tree = np.zeros((M - 1, 2))
        q.put((0, xmin, xmax, ymin, ymax, zmin, zmax, -1, False))
        ind = len(self.partition_tree) - 1
        while not q.empty():
            _, xmin, xmax, ymin, ymax, zmin, zmax, parent_ind, is_left = q.get()
            
            if parent_ind >= 0:
                self.partition_tree[parent_ind, 0 if is_left else 1] = ind

            # make sure we line up with a flattened indexing scheme
            if ind < 0:
                assert -ind - 1 ==  xmin * total_ywidth * total_zwidth + ymin * total_zwidth + zmin

            xwidth = xmax - xmin
            ywidth = ymax - ymin
            zwidth = zmax - zmin
            if xwidth == 1 and ywidth == 1 and zwidth == 1:
                pass
            else:

                # by default our ranges remain unchanged
                lxmin = rxmin = xmin
                lxmax = rxmax = xmax
                lymin = rymin = ymin
                lymax = rymax = ymax
                lzmin = rzmin = zmin
                lzmax = rzmax = zmax

                # split the xaxis if it is the largest dimension
                if xwidth >= ywidth and xwidth > 1:
                    xmid = xmin + xwidth // 2
                    lxmax = xmid
                    rxmin = xmid

                # split the yaxis
                elif ywidth > 1:
                    ymid = ymin + ywidth // 2
                    lymax = ymid
                    rymin = ymid

                # split the zaxis only when the other ranges are already width 1
                else:
                    zmid = zmin + zwidth // 2
                    lzmax = zmid
                    rzmin = zmid

                lsize = (lxmax - lxmin) * (lymax - lymin) * (lzmax - lzmin)
                rsize = (rxmax - rxmin) * (rymax - rymin) * (rzmax - rzmin)

                q.put((-lsize, lxmin, lxmax, lymin, lymax, lzmin, lzmax, ind, True))
                q.put((-rsize, rxmin, rxmax, rymin, rymax, rzmin, rzmax, ind, False))

            ind -= 1
        self.partition_tree += int(M)




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

        # note if this masker can use different background for different samples
        self.variable_background = self.mask_token is not None
    
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
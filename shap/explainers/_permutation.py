from ..utils import partition_tree_shuffle, MaskedModel
from .._explanation import Explanation
from ._explainer import Explainer
import numpy as np
import pandas as pd
import scipy as sp


class Permutation(Explainer):
    """ This method approximates the Shapley values by iterating through permutations of the inputs.

    This is a model agnostic explainer that gurantees local accuracy (additivity) by iterating completely
    through an entire permutatation of the features in both forward and reverse directions. If we do this
    once, then we get the exact SHAP values for models with up to second order interaction effects. We can
    iterate this many times over many random permutations to get better SHAP value estimates for models
    with higher order interactions. This sequential ordering formulation also allows for easy reuse of
    model evaluations and the ability to effciently avoid evaluating the model when the background values
    for a feature are the same as the current input value. We can also account for hierarchial data
    structures with partition trees, something not currently implemented for KernalExplainer or SamplingExplainer.
    """

    def __init__(self, model, masker):
        """ Build an explainers.Permutation object for the given model using the given masker object.

        Parameters
        ----------
        model : function
            A callable python object that executes the model given a set of input data samples.

        masker : function or numpy.array or pandas.DataFrame
            A callable python object used to "mask" out hidden features of the form `masker(x, mask)`.
            It takes a single input sample and a binary mask and returns a matrix of masked samples. These
            masked samples are evaluated using the model function and the outputs are then averaged.
            As a shortcut for the standard masking using by SHAP you can pass a background data matrix
            instead of a function and that matrix will be used for masking. To use a clustering
            game structure you can pass a shap.maksers.Tabular(data, clustering=\"correlation\") object.
        """
        super(Permutation, self).__init__(model, masker)


    def explain_row(self, *row_args, max_evals, main_effects, error_bounds, silent):
        """ Explains a single row and returns the tuple (row_values, row_expected_values, row_mask_shapes).
        """

        # build a masked version of the model for the current input sample
        fm = MaskedModel(self.model, self.masker, *row_args)

        # by default we run 10 permutations forward and backward
        if max_evals == "auto":
            max_evals = 10 * 2 * len(fm)

        # loop over many permutations
        inds = fm.varying_inputs()
        inds_mask = np.zeros(len(fm), dtype=np.bool)
        inds_mask[inds] = True
        masks = np.zeros(2*len(inds)+1, dtype=np.int)
        masks[0] = MaskedModel.delta_mask_noop_value
        npermutations = max_evals // (2*len(inds)+1)
        row_values = np.zeros(len(fm))
        for _ in range(npermutations):

            # shuffle the indexes so we get a random permutation ordering
            if getattr(self.masker, "partition_tree", None) is not None:
                # [TODO] This is shuffle does not work when inds is not a complete set of integers from 0 to M
                #assert len(inds) == len(fm), "Need to support partition shuffle when not all the inds vary!!"
                partition_tree_shuffle(inds, inds_mask, self.masker.partition_tree)
            else:
                np.random.shuffle(inds)

            # create a large batch of masks to evaluate
            i = 1
            for ind in inds:
                masks[i] = ind
                i += 1
            for ind in inds:
                masks[i] = ind
                i += 1
            
            # evaluate the masked model
            outputs = fm(masks)
            
            # update our SHAP value estimates
            for i,ind in enumerate(inds):
                row_values[ind] += outputs[i+1] - outputs[i]
            for i,ind in enumerate(inds):
                row_values[ind] += outputs[i+1] - outputs[i]
        
        expected_value = outputs[0]

        # compute the main effects if we need to
        main_effect_values = fm.main_effects(inds) if main_effects else None
        
        return row_values / (2 * npermutations), expected_value, fm.mask_shapes, main_effect_values
    

    def shap_values(self, X, npermutations=10, main_effects=False, error_bounds=False, batch_evals=True, silent=False):
        """ Legacy interface to estimate the SHAP values for a set of samples.

        Parameters
        ----------
        X : numpy.array or pandas.DataFrame or any scipy.sparse matrix
            A matrix of samples (# samples x # features) on which to explain the model's output.

        npermutations : int
            Number of times to cycle through all the features, re-evaluating the model at each step.
            Each cycle evaluates the model function 2 * (# features + 1) times on a data matrix of
            (# background data samples) rows. An exception to this is when PermutationExplainer can
            avoid evaluating the model because a feature's value is the same in X and the background
            dataset (which is common for example with sparse features).

        Returns
        -------
        For models with a single output this returns a matrix of SHAP values
        (# samples x # features). Each row sums to the difference between the model output for that
        sample and the expected value of the model output (which is stored as expected_value
        attribute of the explainer). For models with vector outputs this returns a list
        of such matrices, one for each output.
        """

        explanation = self(X, max_evals=npermutations * X.shape[1], main_effects=main_effects)
        return explanation._old_format()

    #     # convert dataframes
    #     self.dataframe_columns = None
    #     if str(type(X)).endswith("pandas.core.series.Series'>"):
    #         X = X.values
    #     elif str(type(X)).endswith("'pandas.core.frame.DataFrame'>"):
    #         self.dataframe_columns = list(X.columns)
    #         X = X.values
        
    #     x_type = str(type(X))
    #     arr_type = "'numpy.ndarray'>"
    #     # if sparse, convert to lil for performance
    #     if sp.sparse.issparse(X) and not sp.sparse.isspmatrix_lil(X):
    #         X = X.tolil()
    #     assert x_type.endswith(arr_type) or sp.sparse.isspmatrix_lil(X), "Unknown instance type: " + x_type
    #     assert len(X.shape) == 1 or len(X.shape) == 2, "Instance must have 1 or 2 dimensions!"

    #     # single instance
    #     if len(X.shape) == 1:
    #         data = X.reshape((1, X.shape[0]))
    #         row_phi, row_phi_min, row_phi_max, row_phi_main = self.explain(
    #             data, npermutations=npermutations, main_effects=main_effects, batch_evals=batch_evals,
    #             error_bounds=error_bounds
    #         )

    #         # vector-output
    #         s = row_phi.shape
    #         if len(s) == 2:
    #             outs = [np.zeros(s[0]) for j in range(s[1])]
    #             for j in range(s[1]):
    #                 outs[j] = row_phi[:, j]
    #             return outs

    #         # single-output
    #         else:
    #             out = np.zeros(s[0])
    #             out[:] = row_phi
    #             return out

    #     # explain the whole dataset
    #     elif len(X.shape) == 2:
    #         explanations = []
    #         explanations_min = []
    #         explanations_max = []
    #         explanations_main = []
    #         for i in tqdm(range(X.shape[0]), disable=silent):
    #             data = X[i:i + 1, :]
    #             row_phi, row_phi_min, row_phi_max, row_phi_main = self.explain(
    #                 data, npermutations=npermutations, main_effects=main_effects,
    #                 batch_evals=batch_evals, error_bounds=error_bounds
    #             )
    #             explanations.append(row_phi)
    #             explanations_min.append(row_phi_min)
    #             explanations_max.append(row_phi_max)
    #             explanations_main.append(row_phi_main)

    #         # vector-output
    #         s = explanations[0].shape
    #         if len(s) == 2:
    #             outs = [np.zeros((X.shape[0], s[0])) for j in range(s[1])]
    #             outs_min = copy.deepcopy(outs)
    #             outs_max = copy.deepcopy(outs)
    #             outs_main = copy.deepcopy(outs)
    #             for i in range(X.shape[0]):
    #                 for j in range(s[1]):
    #                     outs[j][i] = explanations[i][:, j]
    #                     if error_bounds:
    #                         outs_min[j][i] = explanations_min[i][:, j]
    #                         outs_max[j][i] = explanations_max[i][:, j]
    #                     if main_effects:
    #                         outs_main[j][i] = explanations_main[i][:, j]
    #             if error_bounds or main_effects:
    #                 return outs, outs_min, outs_max, out_main
    #             else:
    #                 return outs

    #         # single-output
    #         else:
    #             out = np.zeros((X.shape[0], s[0]))
    #             out_min = copy.deepcopy(out)
    #             out_max = copy.deepcopy(out)
    #             out_main = copy.deepcopy(out)
    #             for i in range(X.shape[0]):
    #                 out[i] = explanations[i]
    #                 if error_bounds:
    #                     out_min[i] = explanations_min[i]
    #                     out_max[i] = explanations_max[i]
    #                 if main_effects:
    #                     out_main[i] = explanations_main[i]
    #             if error_bounds or main_effects:
    #                 return out, out_min, out_max, out_main
    #             else:
    #                 return out

    # def explain(self, incoming_instance, **kwargs):
    #     # convert incoming input to a standardized iml object
    #     instance = convert_to_instance(incoming_instance)
    #     match_instance_to_data(instance, self.data)
    #     error_bounds = True

    #     #assert len(self.data.groups) == self.P, "PermutationExplainer does not support feature groups!"

    #     # find the feature groups we will test. If a feature does not change from its
    #     # current value then we know it doesn't impact the model
    #     self.varyingInds = self.varying_groups(instance.x)
    #     #self.varyingFeatureGroups = [self.data.groups[i] for i in self.varyingInds]
    #     self.M = len(self.varyingInds)

    #     # find f(x)
    #     if self.keep_index:
    #         model_out = self.model.f(instance.convert_to_df())
    #     else:
    #         model_out = self.model.f(instance.x)
    #     if isinstance(model_out, (pd.DataFrame, pd.Series)):
    #         model_out = model_out.values[0]
    #     self.fx = model_out[0]

    #     if not self.vector_out:
    #         self.fx = np.array([self.fx])

    #     # if no features vary then there no feature has an effect
    #     if self.M == 0:
    #         phi = np.zeros((len(self.data.groups), self.D))
    #         phi_min = phi.copy() + 1e10
    #         phi_max = phi.copy() - 1e10
    #         # phi_var = np.zeros((len(self.data.groups), self.D))

    #     # if only one feature varies then it has all the effect
    #     elif self.M == 1:
    #         phi = np.zeros((len(self.data.groups), self.D))
    #         # phi_var = np.zeros((len(self.data.groups), self.D))
    #         diff = self.fx - self.fnull
    #         for d in range(self.D):
    #             phi[self.varyingInds[0],d] = diff[d]
    #         phi_min = phi.copy() + 1e10
    #         phi_max = phi.copy() - 1e10

    #     # if more than one feature varies then we have to do real work
    #     else:

    #         # flatten singleton groups
    #         groups = []
    #         for g in self.data.groups:
    #             if len(g) == 1:
    #                 groups.append(g[0])
    #             else:
    #                 groups.append(g)

    #         # pick a reasonable number of samples if the user didn't specify how many they wanted
    #         self.npermutations = kwargs.get("npermutations", 1)

    #         # find which rows actually vary between x and self.data.data, that way we only need to re-evaluate
    #         # when the data is different, which for categorical data can be a big win
    #         varying_rows = []
    #         for g in groups:
    #             diffs = (self.data.data[:, g] != instance.x[0, g])
    #             if hasattr(g, "shape") and len(g.shape) > 0:
    #                 diffs = diffs.sum(1)
    #             group_rows = np.where(diffs > 0)[0]

    #             # use fast full slicing when inds is more more than 90% of the whole set of rows
    #             if len(group_rows) > 0.9 * self.data.data.shape[0]:
    #                 group_rows = slice(None,None,None) 
    #             elif len(group_rows) == 0:
    #                 group_rows = None

    #             varying_rows.append(group_rows)

    #         phi = np.zeros((len(groups), self.D))
    #         phi_min = phi.copy() + 1e10
    #         phi_max = phi.copy() - 1e10
    #         phi_main = phi.copy()
    #         inds = np.arange(len(groups))
    #         X_masked = self.data.data.copy()
    #         evals_prev = self.model.f(X_masked)
    #         evals = evals_prev.copy()
            
    #         total_rows = np.sum([0 if g is None else X_masked[g,:].shape[0] for g in varying_rows])
            
    #         #batch_size = 100
    #         X_batch = np.zeros((total_rows * 2, self.data.data.shape[1]))

    #         #nbackground = self.data.data.shape[0]
    #         main_effects = kwargs.get("main_effects", False)
    #         if main_effects:

    #             # fill in the batch array
    #             batch_pos = 0
    #             for i in inds:
    #                 rows = varying_rows[i] # we only need to run the model for rows where this feature differs between x and the background
    #                 if rows is not None:
    #                     g = groups[i]
    #                     X_masked[:, g] = instance.x[0, g]
    #                     l = X_masked[rows].shape[0]
    #                     X_batch[batch_pos:batch_pos+l] = X_masked[rows]
    #                     batch_pos += l
    #                     X_masked[:, g] = self.data.data[:,g]
                
    #             # run the model
    #             batch_out = self.model.f(X_batch[:batch_pos])

    #             # forward permutation fill out
    #             batch_pos = 0
    #             for i in inds:
    #                 rows = varying_rows[i] # we only need to run the model for rows where this feature differs between x and the background
    #                 if rows is not None:
    #                     l = evals[rows].shape[0]
    #                     evals[rows] = batch_out[batch_pos:batch_pos+l]
    #                     delta = evals.mean(0) - evals_prev.mean(0)
    #                     phi_main[i] += delta
    #                     batch_pos += l
    #                     evals[:] = evals_prev
    #             X_masked = self.data.data.copy()
            
    #         if kwargs.get("batch_evals", True):
    #             for _ in range(self.npermutations):
                    
    #                 if getattr(self.masker, "partition_tree", None) is not None:
    #                     partition_tree_shuffle(inds, self.masker.partition_tree)
    #                 else:
    #                     np.random.shuffle(inds)

    #                 # forward permutation fill out
    #                 batch_pos = 0
    #                 for i in inds:
    #                     rows = varying_rows[i] # we only need to run the model for rows where this feature differs between x and the background
    #                     if rows is not None:
    #                         g = groups[i]
    #                         X_masked[:, g] = instance.x[0, g]
    #                         l = X_masked[rows].shape[0]
    #                         X_batch[batch_pos:batch_pos+l] = X_masked[rows]
    #                         batch_pos += l
    #                 # reverse permutation undo (leaves X_masked in the state is was before the forward pass)
    #                 for i in inds:
    #                     rows = varying_rows[i]
    #                     if rows is not None:
    #                         g = groups[i]
    #                         X_masked[:, g] = self.data.data[:, g]
    #                         l = X_masked[rows].shape[0]
    #                         X_batch[batch_pos:batch_pos+l] = X_masked[rows]
    #                         batch_pos += l
                    
    #                 # run the model
    #                 batch_out = self.model.f(X_batch)
                    
    #                 # forward permutation fill out
    #                 batch_pos = 0
    #                 for i in inds:
    #                     rows = varying_rows[i] # we only need to run the model for rows where this feature differs between x and the background
    #                     if rows is not None:
    #                         l = evals[rows].shape[0]
    #                         evals[rows] = batch_out[batch_pos:batch_pos+l]# + self.model.f(X_masked[rows])
    #                         delta = evals.mean(0) - evals_prev.mean(0)
    #                         phi[i] += delta
    #                         if error_bounds:
    #                             phi_max[i] = np.maximum(phi_max[i], delta)
    #                             phi_min[i] = np.minimum(phi_min[i], delta)
    #                         evals_prev[:] = evals
    #                         batch_pos += l
                    
    #                 # reverse permutation undo (leaves X_masked in the state is was before the forward pass)
    #                 for i in inds:
    #                     rows = varying_rows[i]
    #                     if rows is not None:
    #                         l = evals[rows].shape[0]
    #                         evals[rows] = batch_out[batch_pos:batch_pos+l]
    #                         delta = evals_prev.mean(0) - evals.mean(0)
    #                         phi[i] += delta
    #                         if error_bounds:
    #                             phi_max[i] = np.maximum(phi_max[i], delta)
    #                             phi_min[i] = np.minimum(phi_min[i], delta)
    #                         evals_prev[:] = evals
    #                         batch_pos += l
            
    #         # this is if we are not batching our evals
    #         else:
    #             for _ in range(self.npermutations):

    #                 if getattr(self.masker, "partition_tree", None) is not None:
    #                     partition_tree_shuffle(inds, self.masker.partition_tree)
    #                 else:
    #                     np.random.shuffle(inds)

    #                 # forward permutation fill out
    #                 for i in inds:
    #                     rows = varying_rows[i] # we only need to run the model for rows where this feature differs between x and the background
    #                     if rows is not None:
    #                         g = groups[i]
    #                         X_masked[:, g] = instance.x[0, g]
    #                         evals[rows] = self.model.f(X_masked[rows])
    #                         delta = evals.mean(0) - evals_prev.mean(0)
    #                         phi[i] += delta
    #                         if error_bounds:
    #                             phi_max[i] = np.maximum(phi_max[i], delta)
    #                             phi_min[i] = np.minimum(phi_min[i], delta)
    #                         evals_prev[:] = evals
                    
    #                 # reverse permutation undo (leaves X_masked in the state is was before the forward pass)
    #                 for i in inds:
    #                     rows = varying_rows[i]
    #                     if rows is not None:
    #                         g = groups[i]
    #                         X_masked[:, g] = self.data.data[:, g]
    #                         evals[rows] = self.model.f(X_masked[rows])
    #                         delta = evals_prev.mean(0) - evals.mean(0)
    #                         phi[i] += delta
    #                         if error_bounds:
    #                             phi_max[i] = np.maximum(phi_max[i], delta)
    #                             phi_min[i] = np.minimum(phi_min[i], delta)
    #                         evals_prev[:] = evals

    #         phi /= self.npermutations * 2
        
    #     if phi.shape[1] == 1:
    #         phi = phi[:,0]
    #         if error_bounds:
    #             phi_min = phi_min[:,0]
    #             phi_max = phi_max[:,0]
    #         if main_effects:
    #             phi_main = phi_main[:,0]
        
    #     # self.phi_min = phi_min
    #     # self.phi_max = phi_max
        
    #     return phi, phi_min, phi_max, phi_main
    

    # def varying_groups(self, x):
    #     if not sp.sparse.issparse(x):
    #         varying = np.zeros(self.data.groups_size)
    #         for i in range(0, self.data.groups_size):
    #             inds = self.data.groups[i]
    #             x_group = x[0, inds]
    #             if sp.sparse.issparse(x_group):
    #                 if all(j not in x.nonzero()[1] for j in inds):
    #                     varying[i] = False
    #                     continue
    #                 x_group = x_group.todense()
    #             num_mismatches = np.sum(np.invert(np.isclose(x_group, self.data.data[:, inds], equal_nan=True)))
    #             varying[i] = num_mismatches > 0
    #         varying_indices = np.nonzero(varying)[0]
    #         return varying_indices
    #     else:
    #         varying_indices = []
    #         # go over all nonzero columns in background and evaluation data
    #         # if both background and evaluation are zero, the column does not vary
    #         varying_indices = np.unique(np.union1d(self.data.data.nonzero()[1], x.nonzero()[1]))
    #         remove_unvarying_indices = []
    #         for i in range(0, len(varying_indices)):
    #             varying_index = varying_indices[i]
    #             # now verify the nonzero values do vary
    #             data_rows = self.data.data[:, [varying_index]]
    #             nonzero_rows = data_rows.nonzero()[0]

    #             if nonzero_rows.size > 0:
    #                 background_data_rows = data_rows[nonzero_rows]
    #                 if sp.sparse.issparse(background_data_rows):
    #                     background_data_rows = background_data_rows.toarray()
    #                 num_mismatches = np.sum(np.abs(background_data_rows - x[0, varying_index]) > 1e-7)
    #                 # Note: If feature column non-zero but some background zero, can't remove index
    #                 if num_mismatches == 0 and not \
    #                     (np.abs(x[0, [varying_index]][0, 0]) > 1e-7 and len(nonzero_rows) < data_rows.shape[0]):
    #                     remove_unvarying_indices.append(i)
    #         mask = np.ones(len(varying_indices), dtype=bool)
    #         mask[remove_unvarying_indices] = False
    #         varying_indices = varying_indices[mask]
    #         return varying_indices
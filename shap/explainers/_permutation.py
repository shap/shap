from ..utils import partition_tree_shuffle, MaskedModel
from .._explanation import Explanation
from ._explainer import Explainer
from ..utils import safe_isinstance, show_progress
import numpy as np
import pandas as pd
import scipy as sp
import pickle
import datetime
import cloudpickle
from .. import links
from .. import maskers
from ..maskers import Masker
from ..models import Model

def add_pair(pair_dict, inds, iind, iind2, k):
    k1=inds[iind]
    k2=inds[iind2]
    if k1 in pair_dict:
        pair_dict[k1][k2]=k
    else:
        pair_dict[k1]={k2:k}

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

    def __init__(self, model, masker, link=links.identity, feature_names=None):
        """ Build an explainers.Permutation object for the given model using the given masker object.

        Parameters
        ----------
        model : function
            A callable python object that executes the model given a set of input data samples.

        masker : function or numpy.array or pandas.DataFrame
            A callable python object used to "mask" out hidden features of the form `masker(binary_mask, x)`.
            It takes a single input sample and a binary mask and returns a matrix of masked samples. These
            masked samples are evaluated using the model function and the outputs are then averaged.
            As a shortcut for the standard masking using by SHAP you can pass a background data matrix
            instead of a function and that matrix will be used for masking. To use a clustering
            game structure you can pass a shap.maksers.Tabular(data, clustering=\"correlation\") object.
        """
        super(Permutation, self).__init__(model, masker, link=link, feature_names=feature_names)
        self.mode = ""
        if not isinstance(model, Model):
            self.model = Model(model)

    def explain_full(self, args, max_evals, error_bounds, batch_size, outputs, silent, feature_group_list=None, main_effects=False, need_interactions=False, **kwargs):
        """ Explains a full set of samples (by groups of features) and returns the tuple (row_values, row_expected_values, row_mask_shapes, main_effects, interactions).

        Parameters
        ----------
        args : numpy.array
            samples x # features. Where # features shuold be a 1D array

        batch_size : int
            count of samples to send to model to predict as a batch. every sample would "unfold" into a number of masked samples.
            if main effects and/or interactions are switched on, more masked samples are going to be in a batch.

        feature_groupd_list : dict
             dictionary of "group_feature_name": [list of feature index]
             eg: {"BP": [0, 4, 8], "AG": [1, 5, 9], "CQ": [2, 6, 10], "DD": [3, 7, 11]}  <== this could be a grouping for a RNN of a 4x3 input.

        main_effects : boolean
            if a main_effects would be calculated and returned.

        need_interactions : boolean
            if a interaction value array would be calculated and returned for every sample.

        """
        if batch_size == "auto":
            batch_size = 1
        # by default we run 10 permutations forward and backward
        if feature_group_list is None:
            feature_mask_list = [[i] for i in range(self.masker.shape[1])]
        else:
            feature_mask_list = feature_group_list
        mask_group_len=len(feature_mask_list)
        if max_evals == "auto":
            max_evals = 10 * 2 * mask_group_len

        inds_mask = np.zeros(self.masker.shape[1], dtype=np.bool)
        npermutations = max_evals // (2*mask_group_len+1)
        #last_time = datetime.datetime.now()
        #print("shap calculation started at {}".format(last_time))
        # build a masked version of the model for the current input sample
        fm = MaskedModel(self.model, self.masker, self.link, *args, mode='full')
        #inds = fm.varying_inputs()
        #ind_len=len(inds) # feature_count
        ginds= [i for i in range(mask_group_len)]
        gind_len = mask_group_len #number of feature groups
        ind_len = args[0].shape[1]
        ginds_mask = np.zeros(mask_group_len, dtype=np.bool)
        ginds_len = 0 # shap valid (need to average) masks count 

        def indlist_from_gindlist(ginds, addnew=True):
            new_mask = []
            if type(ginds) is not list:
                ginds=[ginds,]
            for igind in ginds:
                for j,jgind in enumerate(feature_mask_list[igind]):
                    if addnew:
                        if j+1 == len(feature_mask_list[igind]):
                            new_mask.append(jgind)
                        else:
                            new_mask.append(-jgind-1)
                    else:
                        new_mask.append(-jgind-1)
            return new_mask

        if gind_len > 0:

            k=0 #total mask counter
            row_count = args[0].shape[0]
            shapmask_len = 0 # mask count for base_value and shap value
            row_values = None
            main_effect_values = None
            interaction_values = None
            
            for row in show_progress(range(row_count), row_count, self.__class__.__name__+" explainer (full mode)", silent): #generate masks for the whole dataset at once
                if row % batch_size == 0: #send for model prediction in batches
                    base_row = row
                    batch_rows=0
                    masks=[]
                    pair_dict=[]
                    ginds_list=[]

                # loop over many permutations
                batch_rows += 1
                ginds_mask[ginds] = True
                masks += [MaskedModel.delta_mask_noop_value,] #first one, base value
                k+=1
                ginds_list.append([])
                for _ in range(npermutations):
                    
                    # shuffle the indexes so we get a random permutation ordering
                    if getattr(self.masker, "clustering", None) is not None:
                        # [TODO] This is shuffle does not work when inds is not a complete set of integers from 0 to M TODO: still true?
                        #assert ind_len == len(fm), "Need to support partition shuffle when not all the inds vary!!"
                        partition_tree_shuffle(ginds, ginds_mask, self.masker.clustering)
                        # [jnjn] this is not well supported yet in 'full' mode
                    else:
                        np.random.shuffle(ginds)

                    # create a large batch of masks to evaluate

                    masks += indlist_from_gindlist(list(ginds)) + indlist_from_gindlist(list(ginds), addnew = False) + [MaskedModel.delta_mask_noop_value,]
                    k+=gind_len+1
                    ginds_list[batch_rows-1] += list(ginds) + [MaskedModel.delta_mask_noop_value,]
                if row == 0:
                    shapmask_len = k
                    ginds_len=len(ginds_list[batch_rows-1])

                if main_effects:
                    masks_mei = []
                    pair_dict.append({})
                    k_mei = 0 # mei (main_effects and interactions段的masks计数器)
                    for iind in range(gind_len):
                        masks_mei += indlist_from_gindlist(ginds[iind]) #开main effects feature mask
                        add_pair(pair_dict[batch_rows-1], ginds, iind, iind, k_mei)
                        k_mei += 1
                        if need_interactions:
                            for iind2 in range(iind+1, gind_len):
                                masks_mei += indlist_from_gindlist(ginds[iind2])
                                add_pair(pair_dict[batch_rows-1], ginds, iind, iind2, k_mei)
                                k_mei += 1
                                masks_mei += indlist_from_gindlist(ginds[iind2], addnew=False)
                        masks_mei += indlist_from_gindlist(ginds[iind], addnew=False)  #关main effects feature mask
                    k += k_mei
                    masks += masks_mei
                if row == 0:
                    mask_len = k
                    ps_len = len(masks)
                    
                if ((row + 1) % batch_size == 0) or (row + 1 == row_count):

                    masks=np.array(masks, dtype=np.int)
                    outputs = fm(masks, ps_len=ps_len, start_row=base_row)
                    outputs2=outputs.reshape(batch_rows, mask_len, -1)

                    if row_values is None:
                        row_values = np.zeros((row_count, gind_len,) + outputs2.shape[2:])
                        expected_values = np.zeros((row_count,) + outputs2.shape[2:])
                        if main_effects:
                            main_effect_values = np.zeros((row_count, gind_len,) + outputs2.shape[2:])
                            if need_interactions:
                                interaction_values = np.zeros((row_count, gind_len, gind_len,) + outputs2.shape[2:])         

                    expected_values[base_row:base_row + batch_rows]=outputs2[:,0]
                    for batch_row in range(batch_rows):
                            
                        for j in range(ginds_len):
                            d=ginds_list[batch_row][j]
                            if d != MaskedModel.delta_mask_noop_value:
                                row_values[base_row + batch_row, d] += (outputs2[batch_row,j+1] - outputs2[batch_row,j])
                        if main_effects:
                            for j in range(gind_len):
                                main_effect_values[base_row + batch_row, j] = outputs2[batch_row, shapmask_len + pair_dict[batch_row][j][j]] - outputs2[batch_row, 0]
                            
                                if need_interactions:
                                    for j2 in range(j,gind_len):
                                        if j != j2:
                                            d=None
                                            if j2 in pair_dict[batch_row][j]:
                                                d=pair_dict[batch_row][j][j2]
                                            else:
                                                d=pair_dict[batch_row][j2][j]
                                            if d:
                                                itt=(outputs2[batch_row,shapmask_len + d] + outputs2[batch_row,0] - outputs2[batch_row, shapmask_len + pair_dict[batch_row][j][j]] - outputs2[batch_row, shapmask_len + pair_dict[batch_row][j2][j2]])/2
                                                interaction_values[base_row + batch_row,j,j2]=itt
                                                interaction_values[base_row + batch_row,j2,j]=itt   
                                        else:
                                            interaction_values[base_row + batch_row,j,j2]=main_effect_values[base_row + batch_row, j]
                    #present_time =  datetime.datetime.now()
                    #if last_time:
                    #    time_diff = (present_time - last_time).seconds
                    #    time_eta = (time_diff / batch_rows ) * (row_count - (row+1))
                    #    last_time = present_time
                    #    print(" --> done calculation of row {} (of {}) at {}, used {} seconds ({} seconds per sample), ETA: {} min left".format(row+1, row_count, datetime.datetime.now(), time_diff, time_diff / batch_rows, round(time_eta/60, 1)))    
                    
                    #if (type(need_temp_save) is int) and (((row + 1) % (batch_rows * need_temp_save) == 0) or (row + 1 == row_count)):
                    #    print(" ===> saving as directed at row {} (every {} batches)".format(row+1, need_temp_save))
                    #    with open("temp_shap_cal.save", mode="wb") as sf:
                    #        pickle.dump([expected_values, row_values, main_effect_values, interaction_values, row], sf)
                                
        return {
            "values": row_values / npermutations,
            "expected_values": expected_values,
            "mask_shapes": fm.mask_shapes,
            "main_effects": main_effect_values,
            "interactions": interaction_values,
            "clustering": getattr(self.masker, "clustering", None)
        }
    
    def explain_row(self, *row_args, max_evals, main_effects, error_bounds, batch_size, outputs, silent, **kwargs):
        """ Explains a single row and returns the tuple (row_values, row_expected_values, row_mask_shapes).
        """

        # build a masked version of the model for the current input sample
        fm = MaskedModel(self.model, self.masker, self.link, *row_args)

        # by default we run 10 permutations forward and backward
        if max_evals == "auto":
            max_evals = 10 * 2 * len(fm)
        
        # see if we need interactions
        need_interactions = kwargs.get("need_interactions", False)

        # compute any custom clustering for this row
        row_clustering = None
        if getattr(self.masker, "clustering", None) is not None:
            if isinstance(self.masker.clustering, np.ndarray):
                row_clustering = self.masker.clustering
            elif callable(self.masker.clustering):
                row_clustering = self.masker.clustering(*row_args)
            else:
                raise Exception("The masker passed has a .clustering attribute that is not yet supported by the Permutation explainer!")

        # loop over many permutations
        inds = fm.varying_inputs()
        inds_mask = np.zeros(len(fm), dtype=np.bool)
        inds_mask[inds] = True
        masks = np.zeros(2*len(inds)+1, dtype=np.int)
        masks[0] = MaskedModel.delta_mask_noop_value
        npermutations = max_evals // (2*len(inds)+1)
        row_values = None
        main_effect_values = None
        interaction_values = None
        if len(inds) > 0:
            for _ in range(npermutations):

                # shuffle the indexes so we get a random permutation ordering
                if row_clustering is not None:
                    # [TODO] This is shuffle does not work when inds is not a complete set of integers from 0 to M TODO: still true?
                    #assert len(inds) == len(fm), "Need to support partition shuffle when not all the inds vary!!"
                    partition_tree_shuffle(inds, inds_mask, row_clustering)
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
                outputs = fm(masks, batch_size=batch_size)

                if row_values is None:
                    row_values = np.zeros((len(fm),) + outputs.shape[1:])

                # update our SHAP value estimates
                for i,ind in enumerate(inds):
                    row_values[ind] += outputs[i+1] - outputs[i]
                for i,ind in enumerate(inds):
                    row_values[ind] += outputs[i+1] - outputs[i]

            if npermutations == 0:
                raise Exception("max_evals is too low for the Permutation explainer, it must be at least 2 * num_features + 1!")

            expected_value = outputs[0]

            # compute the main effects if we need to
            if main_effects:
                main_effect_values, interaction_values = fm.main_effects(inds, need_interactions=need_interactions)

        return {
            "values": row_values / (2 * npermutations),
            "expected_values": expected_value,
            "mask_shapes": fm.mask_shapes,
            "main_effects": main_effect_values,
            "interactions": interaction_values,
            "clustering": row_clustering,
            "output_names": self.model.output_names if hasattr(self.model, "output_names") else None
        }


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
        array or list
            For models with a single output this returns a matrix of SHAP values
            (# samples x # features). Each row sums to the difference between the model output for that
            sample and the expected value of the model output (which is stored as expected_value
            attribute of the explainer). For models with vector outputs this returns a list
            of such matrices, one for each output.
        """

        explanation = self(X, max_evals=npermutations * X.shape[1], main_effects=main_effects)
        return explanation._old_format()

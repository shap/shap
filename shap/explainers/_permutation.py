from ..utils import partition_tree_shuffle, MaskedModel
from .._explanation import Explanation
from ._explainer import Explainer
import numpy as np
import pandas as pd
import scipy as sp
import pickle
import cloudpickle
from .. import links
from .. import maskers
from ..maskers import Masker
from ..models import Model

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
        
        self.model = Model(model)


    def explain_row(self, *row_args, max_evals, main_effects, error_bounds, batch_size, outputs, silent):
        """ Explains a single row and returns the tuple (row_values, row_expected_values, row_mask_shapes).
        """

        # build a masked version of the model for the current input sample
        fm = MaskedModel(self.model, self.masker, self.link, *row_args)

        # by default we run 10 permutations forward and backward
        if max_evals == "auto":
            max_evals = 10 * 2 * len(fm)

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
            
            expected_value = outputs[0]

            # compute the main effects if we need to
            if main_effects:
                main_effect_values = fm.main_effects(inds)
        
        return {
            "values": row_values / (2 * npermutations),
            "expected_values": expected_value,
            "mask_shapes": fm.mask_shapes,
            "main_effects": main_effect_values,
            "clustering": row_clustering
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


    def save(self, out_file):
        """ Saves content of permutation explainer
        """

        super(Permutation, self).save(out_file)
        
        if callable(self.model.save):
            self.model.save(out_file, self.model.model)
        else:
            pickle.dump(None,out_file)
        
        if callable(self.masker.save):
            self.masker.save(out_file, self.masker)
        else:
            pickle.dump(None,out_file)
        
        cloudpickle.dump(self.link, out_file)
        cloudpickle.dump(self.link.inverse,out_file)
        pickle.dump(self.feature_names, out_file)

    @classmethod
    def load(cls, in_file, model_loader = None, masker_loader = None):
        explainer_type = pickle.load(in_file)
        if not explainer_type == cls:
            print("Warning: Saved explainer type not same as the one that's attempting to be loaded. Saved explainer type: ", explainer_type)
        
        return Permutation._load(in_file, model_loader, masker_loader)
    
    @classmethod
    def _load(cls, in_file, model_loader = None, masker_loader = None):
        if model_loader is None:
            model = Model.load(in_file)
        else:
            model = model_loader(in_file)
        
        if masker_loader is None:
            masker = Masker.load(in_file)
        else:
            masker = masker_loader(in_file)

        link = cloudpickle.load(in_file)
        link_inverse = cloudpickle.load(in_file)
        link.inverse = link_inverse
        feature_names = pickle.load(in_file)
        return Permutation(model, masker, link, feature_names)
        
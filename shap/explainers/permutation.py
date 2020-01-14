from ..common import convert_to_instance, convert_to_model, match_instance_to_data, match_model_to_data, convert_to_instance_with_index, convert_to_link, IdentityLink, convert_to_data, DenseData
from .kernel import KernelExplainer
import numpy as np
import pandas as pd
import logging

log = logging.getLogger('shap')


class PermutationExplainer(KernelExplainer):
    """ This method approximates the Shapley values by iteration through permutations of the inputs.

    This is an alternative to the KernelExplainer and the SamplingExplainer where we gurantee
    local accuracy (additivity) by iterating completely through an entire permutatation of the
    features in both forward and reverse directions. If we do this once, then we get the exact SHAP
    values for models with up to second order interaction effects. We can iterate this many times over
    many random permutations to get better SHAP value estimates for models we higher order interactions.
    """

    def __init__(self, model, data, **kwargs):
        super(PermutationExplainer, self).__init__(model, data, **kwargs)


    def explain(self, incoming_instance, **kwargs):
        # convert incoming input to a standardized iml object
        instance = convert_to_instance(incoming_instance)
        match_instance_to_data(instance, self.data)

        #assert len(self.data.groups) == self.P, "PermutationExplainer does not support feature groups!"

        # find the feature groups we will test. If a feature does not change from its
        # current value then we know it doesn't impact the model
        self.varyingInds = self.varying_groups(instance.x)
        #self.varyingFeatureGroups = [self.data.groups[i] for i in self.varyingInds]
        self.M = len(self.varyingInds)

        # find f(x)
        if self.keep_index:
            model_out = self.model.f(instance.convert_to_df())
        else:
            model_out = self.model.f(instance.x)
        if isinstance(model_out, (pd.DataFrame, pd.Series)):
            model_out = model_out.values[0]
        self.fx = model_out[0]

        if not self.vector_out:
            self.fx = np.array([self.fx])

        # if no features vary then there no feature has an effect
        if self.M == 0:
            phi = np.zeros((len(self.data.groups), self.D))
            # phi_var = np.zeros((len(self.data.groups), self.D))

        # if only one feature varies then it has all the effect
        elif self.M == 1:
            phi = np.zeros((len(self.data.groups), self.D))
            # phi_var = np.zeros((len(self.data.groups), self.D))
            diff = self.fx - self.fnull
            for d in range(self.D):
                phi[self.varyingInds[0],d] = diff[d]

        # if more than one feature varies then we have to do real work
        else:

            # pick a reasonable number of samples if the user didn't specify how many they wanted
            self.npermutations = kwargs.get("npermutations", 1)

            phi = np.zeros((len(self.data.groups), self.D))
            inds = np.arange(len(self.data.groups))
            X_masked = self.data.data.copy()
            evals_prev = self.model.f(X_masked)
            for _ in range(self.npermutations):

                # forward permuation fill out
                for i in inds:
                    g = self.data.groups[i]
                    X_masked[:, g] = instance.x[0, g]
                    evals = self.model.f(X_masked)
                    phi[i] += (evals - evals_prev).mean(0)
                    evals_prev = evals
                
                # reverse permuation undo (leaves X_masked in the state is was before the forward pass)
                for i in inds:
                    g = self.data.groups[i]
                    X_masked[:, g] = self.data.data[:, g]
                    evals = self.model.f(X_masked)
                    phi[i] += (evals_prev - evals).mean(0)
                    evals_prev = evals

                np.random.shuffle(inds)

            phi /= self.npermutations * 2
        
        if phi.shape[1] == 1:
            phi = phi[:,0]

        return phi

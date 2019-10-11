from ..common import convert_to_instance, convert_to_model, match_instance_to_data, match_model_to_data, convert_to_instance_with_index, convert_to_link, IdentityLink, convert_to_data, DenseData
from .kernel import KernelExplainer
import numpy as np
import pandas as pd
import logging
import scipy.special
import numpy as np
import itertools

log = logging.getLogger('shap')




class BruteForceExplainer(KernelExplainer):
    """ This is a brute force implementation of SHAP values intended for unit tests, etc.
    """

    def __init__(self, model, data, **kwargs):
        
        # silence warning about large datasets
        level = log.level
        log.setLevel(logging.ERROR)
        super(BruteForceExplainer, self).__init__(model, data, **kwargs)
        log.setLevel(level)

        assert str(self.link) == "identity", "BruteForceExplainer only supports the identity link not " + str(self.link)

    def explain(self, incoming_instance, **kwargs):
        
        # convert incoming input to a standardized iml object
        instance = convert_to_instance(incoming_instance)
        match_instance_to_data(instance, self.data)

        assert len(self.data.groups) == self.P, "BruteForceExplainer does not support feature groups!"

        # find the feature groups we will test. If a feature does not change from its
        # current value then we know it doesn't impact the model
        self.varyingInds = self.varying_groups(instance.x)
        #self.varyingFeatureGroups = [self.data.groups[i] for i in self.varyingInds]
        self.M = len(self.varyingInds)

        phi = np.zeros((len(self.data.groups), self.D))
        phi_var = np.zeros((len(self.data.groups), self.D))
        x = instance.x
        masker_data = self.data.data
        masker = lambda x, mask: x * mask + masker_data * np.invert(mask)
        mask = np.zeros(self.P, dtype=np.bool)
        f = self.model.f
        for i in range(self.P):
            for s in powerset(set(range(self.P)).difference([i])):
                weight = 1 / (scipy.special.comb(self.P - 1, len(s)) * self.P)
                mask[:] = 0
                mask[list(s)] = 1
                f_without_i = f(masker(x, mask)).mean(0)
                mask[i] = 1
                f_with_i = f(masker(x, mask)).mean(0)
                phi[i,:] += weight * (f_with_i - f_without_i)

        if phi.shape[1] == 1:
            phi = phi[:,0]

        return phi


def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

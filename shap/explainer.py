from iml.common import convert_to_instance, convert_to_model, match_instance_to_data, match_model_to_data
from iml.explanations import AdditiveExplanation
from iml.links import convert_to_link, IdentityLink
from iml.datatypes import convert_to_data, DenseData
from scipy.special import binom
import numpy as np
import logging
import copy
import itertools
from sklearn.linear_model import LassoLarsIC, Lasso
from tqdm import tqdm

log = logging.getLogger('shap')

class KernelExplainer:

    def __init__(self, model, data, link=IdentityLink(), **kwargs):

        # convert incoming inputs to standardized iml objects
        self.link = convert_to_link(link)
        self.model = convert_to_model(model)
        self.data = convert_to_data(data)
        match_model_to_data(self.model, self.data)

        # enforce our current input type limitations
        assert isinstance(self.data, DenseData), "Shap explainer only supports the DenseData input currently."
        assert not self.data.transposed, "Shap explainer does not support transposed DenseData currently."

        # init our parameters
        self.N = self.data.data.shape[0]
        self.P = self.data.data.shape[1]
        self.weights = kwargs.get("weights", np.ones(self.N)) # TODO: Use these weights!
        self.weights /= sum(self.weights)
        assert len(self.weights) == self.N,  "Provided 'weights' must match the number of representative data points {0}!".format(self.N)
        self.linkfv = np.vectorize(self.link.f)
        self.nsamplesAdded = 0
        self.nsamplesRun = 0

    def shap_values(self, X, **kwargs):

        # convert dataframes
        if str(type(X)) == "<class 'pandas.core.series.Series'>":
            X = X.as_matrix()
        elif str(type(X)) == "<class 'pandas.core.frame.DataFrame'>":
            X = X.as_matrix()

        if type(X) == np.ndarray:

            # single instance
            if len(X.shape) == 1:
                explanation = self.explain(X.reshape((1,X.shape[0])), **kwargs)

                # vector-output
                s = explanation.effects.shape
                if len(s) == 2:
                    outs = [np.zeros(s[0]+1) for j in range(s[1])]
                    for j in range(s[1]):
                        outs[j][:-1] = explanation.effects[:,j]
                        outs[j][-1] = explanation.base_value[j]
                    return outs

                # single-output
                else:
                    out = np.zeros(s[0]+1)
                    out[:-1] = explanation.effects
                    out[-1] = explanation.base_value
                    return out

            # explain the whole dataset
            elif len(X.shape) == 2:
                explanations = []
                for i in tqdm(range(X.shape[0])):
                    explanations.append(self.explain(X[i:i+1,:], **kwargs))

                # vector-output
                s = explanations[0].effects.shape
                if len(s) == 2:
                    outs = [np.zeros((X.shape[0],s[0]+1)) for j in range(s[1])]
                    for i in range(X.shape[0]):
                        for j in range(s[1]):
                            outs[j][i,:-1] = explanations[i].effects[:,j]
                            outs[j][i,-1] = explanations[i].base_value[j]
                    return outs

                # single-output
                else:
                    out = np.zeros((X.shape[0],s[0]+1))
                    for i in range(X.shape[0]):
                        out[i,:-1] = explanations[i].effects
                        out[i,-1] = explanations[i].base_value
                    return out
            else:
                assert False, "Instance must have 1 or 2 dimensions!"
        else:
            assert False, "Unknown instance type: "+str(type(X))

    def explain(self, incoming_instance, **kwargs):

        # convert incoming input to a standardized iml object
        instance = convert_to_instance(incoming_instance)
        match_instance_to_data(instance, self.data)

        # find the feature groups we will test. If a feature does not change from its
        # current value then we know it doesn't impact the model
        self.varyingInds = self.varying_groups(instance.x)
        self.varyingFeatureGroups = [self.data.groups[i] for i in self.varyingInds]
        self.M = len(self.varyingFeatureGroups)

        # find f(x) and E_x[f(x)]
        model_out = self.model.f(instance.x)
        self.fx = model_out[0]
        self.fnull = np.mean(self.model.f(self.data.data),0)
        self.vector_out = True
        if len(model_out.shape) == 1:
            self.vector_out = False
            self.D = 1
            self.fx = np.array([self.fx])
            self.fnull = np.array([self.fnull])
        else:
            self.D = model_out.shape[1]

        # if no features vary then there no feature has an effect
        if self.M == 0:
            phi = np.zeros(len(self.data.groups))
            phi_var = np.zeros(len(self.data.groups))
            return AdditiveExplanation(self.fnull, self.fx, phi, phi_var, instance, self.link, self.model, self.data)


        # if only one feature varies then it has all the effect
        elif self.M == 1:
            phi = np.zeros(len(self.data.groups))
            phi[self.varyingInds[0]] = self.link.f(self.fx) - self.link.f(self.fnull)
            phi_var = np.zeros(len(self.data.groups))
            return AdditiveExplanation(self.fnull, self.fx, phi, phi_var, instance, self.link, self.model, self.data)

        self.l1_reg = kwargs.get("l1_reg", "auto")

        # pick a reasonable number of samples if the user didn't specify how many they wanted
        self.nsamples = kwargs.get("nsamples", 0)
        if self.nsamples == 0:
            self.nsamples = 2*self.M+1000

        # if we have enough samples to enumerate all subsets then ignore the unneeded samples
        self.max_samples = 2**30
        if self.M <= 30 and self.nsamples > 2**self.M-2:
            self.nsamples = 2**self.M-2
            self.max_samples = self.nsamples

        # reserve space for some of our computations
        self.allocate()

        # weight the different subset sizes
        num_subset_sizes = np.int(np.ceil((self.M-1)/2.0))
        num_paired_subset_sizes = np.int(np.floor((self.M-1)/2.0))
        weight_vector = np.array([(self.M-1.0)/(i*(self.M-i)) for i in range(1,num_subset_sizes+1)])
        weight_vector[:num_paired_subset_sizes] *= 2
        weight_vector /= np.sum(weight_vector)
        log.debug("weight_vector = {0}".format(weight_vector))
        log.debug("num_subset_sizes = {0}".format(num_subset_sizes))
        log.debug("num_paired_subset_sizes = {0}".format(num_paired_subset_sizes))

        # fill out all the subset sizes we can completely enumerate
        # given nsamples*remaining_weight_vector[subset_size]
        num_full_subsets = 0
        num_samples_left = self.nsamples
        group_inds = np.arange(self.M, dtype='int64')
        mask = np.zeros(self.M)
        remaining_weight_vector = copy.copy(weight_vector)
        for subset_size in range(1,num_subset_sizes+1):

            # determine how many subsets (and their complements) are of the current size
            nsubsets = binom(self.M, subset_size)
            if subset_size <= num_paired_subset_sizes: nsubsets *= 2
            log.debug("subset_size = {0}".format(subset_size))
            log.debug("nsubsets = {0}".format(nsubsets))
            log.debug("self.nsamples*weight_vector[subset_size-1] = {0}".format(num_samples_left*remaining_weight_vector[subset_size-1]))
            log.debug("self.nsamples*weight_vector[subset_size-1/nsubsets = {0}".format(num_samples_left*remaining_weight_vector[subset_size-1]/nsubsets))

            # see if we have enough samples to enumerate all subsets of this size
            if num_samples_left*remaining_weight_vector[subset_size-1]/nsubsets >= 1.0-1e-8:
                num_full_subsets += 1
                num_samples_left -= nsubsets

                # rescale what's left of the remaining weight vector to sum to 1
                if remaining_weight_vector[subset_size-1] < 1.0:
                    remaining_weight_vector /= (1-remaining_weight_vector[subset_size-1])

                # add all the samples of the current subset size
                w = weight_vector[subset_size-1] / binom(self.M, subset_size)
                if subset_size <= num_paired_subset_sizes: w /= 2.0
                for inds in itertools.combinations(group_inds, subset_size):
                    mask[:] = 0.0
                    mask[np.array(inds, dtype='int64')] = 1.0
                    self.addsample(instance.x, mask, w)
                    if subset_size <= num_paired_subset_sizes:
                        mask[:] = np.abs(mask - 1)
                        self.addsample(instance.x, mask, w)
            else:
                break
        log.info("num_full_subsets = {0}".format(num_full_subsets))

        # add random samples from what is left of the subset space
        samples_left = self.nsamples - self.nsamplesAdded
        log.debug("samples_left = {0}".format(samples_left))
        if num_full_subsets != num_subset_sizes:
            weight_left = np.sum(weight_vector[num_full_subsets:])
            rand_sample_weight = weight_left/samples_left
            log.info("weight_left = {0}".format(weight_left))
            log.info("rand_sample_weight = {0}".format(rand_sample_weight))
            remaining_weight_vector = weight_vector[num_full_subsets:]
            remaining_weight_vector /= np.sum(remaining_weight_vector)
            log.info("remaining_weight_vector = {0}".format(remaining_weight_vector))
            log.info("num_paired_subset_sizes = {0}".format(num_paired_subset_sizes))
            ind_set = np.arange(len(remaining_weight_vector))
            while samples_left > 0:
                mask[:] = 0.0
                np.random.shuffle(group_inds)
                ind = np.random.choice(ind_set, 1, p=remaining_weight_vector)[0]
                mask[group_inds[:ind+num_full_subsets+1]] = 1.0
                samples_left -= 1
                self.addsample(instance.x, mask, rand_sample_weight)

                # add the compliment sample
                if samples_left > 0:
                    mask -= 1.0
                    mask[:] = np.abs(mask)
                    self.addsample(instance.x, mask, rand_sample_weight)
                    samples_left -= 1

        # execute the model on the synthetic samples we have created
        self.run()

        # solve then expand the feature importance (Shapley value) vector to contain the non-varying features
        phi = np.zeros((len(self.data.groups), self.D))
        phi_var = np.zeros((len(self.data.groups), self.D))
        for d in range(self.D):
            vphi,vphi_var = self.solve(self.nsamples/self.max_samples, d)
            phi[self.varyingInds,d] = vphi
            phi_var[self.varyingInds,d] = vphi_var

        if not self.vector_out:
            phi = np.squeeze(phi, axis=1)
            phi_var = np.squeeze(phi_var, axis=1)
            self.fx = self.fx[0]
            self.fnull = self.fnull[0]

        # return the Shapley values along with variances of the estimates
        # note that if features were eliminated by l1 regression their
        # variance will be 0, even though they are not perfectaly known
        return AdditiveExplanation(self.link.f(self.fnull), self.link.f(self.fx), phi, phi_var, instance, self.link, self.model, self.data)

    def varying_groups(self, x):
        varying = np.zeros(len(self.data.groups))
        for i in range(0,len(self.data.groups)):
            inds = self.data.groups[i]
            varying[i] = sum(sum(np.abs(x[0,inds] - self.data.data[:,inds]) < 1e-8) != len(inds))
        return np.nonzero(varying)[0]

    def allocate(self):
        self.synth_data = np.zeros((self.nsamples * self.N, self.P))
        self.maskMatrix = np.zeros((self.nsamples, self.M))
        self.kernelWeights = np.zeros(self.nsamples)
        self.y = np.zeros((self.nsamples * self.N, self.D))
        self.ey = np.zeros((self.nsamples, self.D))
        self.lastMask = np.zeros(self.nsamples)
        self.nsamplesAdded = 0
        self.nsamplesRun = 0

    def addsample(self, x, m, w):
        offset = self.nsamplesAdded * self.N
        for i in range(self.N):
            for j in range(self.M):
                for k in self.varyingFeatureGroups[j]:
                    if m[j] == 1.0:
                        self.synth_data[offset+i,k] = x[0,k]
                    else:
                        self.synth_data[offset+i,k] = self.data.data[i,k]

        self.maskMatrix[self.nsamplesAdded,:] = m
        self.kernelWeights[self.nsamplesAdded] = w
        self.nsamplesAdded += 1

    def run(self):
        num_to_run = self.nsamplesAdded*self.N - self.nsamplesRun*self.N
        modelOut = self.model.f(self.synth_data[self.nsamplesRun*self.N:self.nsamplesAdded*self.N,:])
        # if len(modelOut.shape) > 1:
        #     raise ValueError("The supplied model function should output a vector not a matrix!")
        self.y[self.nsamplesRun*self.N:self.nsamplesAdded*self.N,:] = np.reshape(modelOut, (num_to_run,self.D))

        # find the expected value of each output
        for i in range(self.nsamplesRun, self.nsamplesAdded):
            eyVal = np.zeros(self.D)
            for j in range(0, self.N):
                eyVal += self.y[i*self.N + j,:]

            self.ey[i,:] = eyVal/self.N
            self.nsamplesRun += 1

    def solve(self, fraction_evaluated, dim):
        eyAdj = self.linkfv(self.ey[:,dim]) - self.link.f(self.fnull[dim])

        s = np.sum(self.maskMatrix, 1)

        # do feature selection if we have not well enumerated the space
        nonzero_inds = np.arange(self.M)
        log.debug("fraction_evaluated = {0}".format(fraction_evaluated))
        if (self.l1_reg not in ["auto", False, 0]) or (fraction_evaluated < 0.2 and self.l1_reg == "auto"):
            w_aug = np.hstack((self.kernelWeights * (self.M-s), self.kernelWeights*s))
            log.info("np.sum(w_aug) = {0}".format(np.sum(w_aug)))
            log.info("np.sum(self.kernelWeights) = {0}".format(np.sum(self.kernelWeights)))
            w_sqrt_aug = np.sqrt(w_aug)
            eyAdj_aug = np.hstack((eyAdj, eyAdj - (self.link.f(self.fx[dim]) - self.link.f(self.fnull[dim]))))
            eyAdj_aug *= w_sqrt_aug
            mask_aug = np.transpose(w_sqrt_aug*np.transpose(np.vstack((self.maskMatrix, self.maskMatrix-1))))
            var_norms = np.array([np.linalg.norm(mask_aug[:,i]) for i in range(mask_aug.shape[1])])

            if self.l1_reg == "auto":
                model = LassoLarsIC(criterion="aic")
            elif self.l1_reg == "bic" or self.l1_reg == "aic":
                model = LassoLarsIC(criterion=self.l1_reg)
            else:
                model = Lasso(alpha=self.l1_reg)

            model.fit(mask_aug, eyAdj_aug)
            nonzero_inds = np.nonzero(model.coef_)[0]

        if len(nonzero_inds) == 0:
            return np.zeros(self.M),np.ones(self.M)

        # eliminate one variable with the constraint that all features sum to the output
        eyAdj2 = eyAdj - self.maskMatrix[:,nonzero_inds[-1]]*(self.link.f(self.fx[dim]) - self.link.f(self.fnull[dim]))
        etmp = np.transpose(np.transpose(self.maskMatrix[:,nonzero_inds[:-1]]) - self.maskMatrix[:,nonzero_inds[-1]])
        log.debug("etmp[:4,:] {0}".format(etmp[:4,:]))

        # solve a weighted least squares equation to estimate phi
        tmp = np.transpose(np.transpose(etmp) * np.transpose(self.kernelWeights))
        tmp2 = np.linalg.inv(np.dot(np.transpose(tmp),etmp))
        w = np.dot(tmp2,np.dot(np.transpose(tmp),eyAdj2))
        log.debug("np.sum(w) = {0}".format(np.sum(w)))
        log.debug("self.link(self.fx) - self.link(self.fnull) = {0}".format(self.link.f(self.fx[dim]) - self.link.f(self.fnull[dim])))
        log.debug("self.fx = {0}".format(self.fx[dim]))
        log.debug("self.link(self.fx) = {0}".format(self.link.f(self.fx[dim])))
        log.debug("self.fnull = {0}".format(self.fnull[dim]))
        log.debug("self.link(self.fnull) = {0}".format(self.link.f(self.fnull[dim])))
        phi = np.zeros(self.M)
        phi[nonzero_inds[:-1]] = w
        phi[nonzero_inds[-1]] = (self.link.f(self.fx[dim]) - self.link.f(self.fnull[dim])) - sum(w)
        log.info("phi = {0}".format(phi))

        # clean up any rounding errors
        for i in range(self.M):
            if np.abs(phi[i]) < 1e-10:
                phi[i] = 0

        return phi,np.ones(len(phi))

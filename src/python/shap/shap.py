from iml.common import convert_to_instance, convert_to_model, match_instance_to_data, match_model_to_data
from iml.explanations import AdditiveExplanation
from iml.links import convert_to_link, IdentityLink
from iml.datatypes import convert_to_data, DenseData
from iml import initjs
from scipy.special import binom
import numpy as np
import logging
import copy
import itertools
from sklearn.linear_model import LassoLarsIC


log = logging.getLogger('shap')


class KernelExplainer:

    #@profile
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

    #@profile
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
        self.fx = self.model.f(instance.x)[0]
        self.fnull = np.mean(self.model.f(self.data.data))

        # if no features vary then there no feature has an effect
        if self.M == 0:
            return self.fx,np.zeros(self.featureGroups.size),np.zeros(self.featureGroups.size)

        # if only one feature varies then it has all the effect
        elif self.M == 1:
            phi = np.zeros(len(self.featureGroups))
            phi[self.varyingInds[0]] = self.link(self.fx) - self.link(self.fnull)
            return self.fnull,phi,np.zeros(len(self.featureGroups))

        # pick a reasonable number of samples if the user didn't specify how many they wanted
        self.nsamples = kwargs.get("nsamples", 0)
        if self.nsamples == 0:
            self.nsamples = 2*self.M+1000

        # if we have enough samples to enumerate all subsets then ignore the unneeded samples
        if self.M <= 30 and self.nsamples > 2**self.M-2:
            self.nsamples = 2**self.M-2

        # reserve space for some of our computations
        self.allocate()

        # weight the different subset sizes
        num_subset_sizes = np.int(np.ceil((self.M-1)/2.0))
        num_paired_subset_sizes = np.int(np.floor((self.M-1)/2.0))
        weight_vector = np.array([(self.M-1)/(i*(self.M-i)) for i in range(1,num_subset_sizes+1)])
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
                if subset_size <= num_paired_subset_sizes: w /= 2
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
        vphi,vphiVar = self.solve()
        phi = np.zeros(len(self.data.groups))
        phi[self.varyingInds] = vphi
        phiVar = np.zeros(len(self.data.groups))
        phiVar[self.varyingInds] = vphiVar

        # return the Shapley values along with variances of the estimates
        # note that if features were eliminated by l1 regression their
        # variance will be 0, even though they are not perfectaly known
        return AdditiveExplanation(self.fnull, self.fx, phi, phiVar, instance, self.link, self.model, self.data)

    def varying_groups(self, x):
        varying = np.zeros(len(self.data.groups))
        for i in range(0,len(self.data.groups)):
            inds = self.data.groups[i]
            varying[i] = sum(sum(x[0,inds] == self.data.data[:,inds]) != len(inds))
        return np.nonzero(varying)[0]

    def allocate(self):
        self.synth_data = np.zeros((self.nsamples * self.N, self.P))
        self.maskMatrix = np.zeros((self.nsamples, self.M))
        self.kernelWeights = np.zeros(self.nsamples)
        self.y = np.zeros(self.nsamples * self.N)
        self.ey = np.zeros(self.nsamples)
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
        modelOut = self.model.f(self.synth_data[self.nsamplesRun*self.N:self.nsamplesAdded*self.N,:])
        self.y[self.nsamplesRun*self.N:self.nsamplesAdded*self.N] = modelOut

        # find the expected value of each output
        for i in range(self.nsamplesRun, self.nsamplesAdded):
            eyVal = 0.0
            for j in range(0, self.N):
                eyVal += self.y[i*self.N + j]

            self.ey[i] = eyVal/self.N
            self.nsamplesRun += 1

    def solve(self):
        count = 0.0
        for i in range(self.maskMatrix.shape[0]):
            if self.maskMatrix[i,0] == 1 and sum(self.maskMatrix[i,1:]) == 0:
                count += 1
        log.info("[1,0,0,0] ratio = {0}".format(count/self.maskMatrix.shape[0]))

        count = 0.0
        for i in range(self.maskMatrix.shape[0]):
            if sum(self.maskMatrix[i,:]) == 2 or sum(self.maskMatrix[i,:]) == 18:
                count += 1
        log.info("2 or 18 sum ratio = {0}".format(count/self.maskMatrix.shape[0]))

        count = 0.0
        for i in range(self.maskMatrix.shape[0]):
            if sum(self.maskMatrix[i,:]) == 3 or sum(self.maskMatrix[i,:]) == 17:
                count += 1
        log.info("3 or 17 sum ratio = {0}".format(count/self.maskMatrix.shape[0]))

        count = 0.0
        for i in range(self.maskMatrix.shape[0]):
            if sum(self.maskMatrix[i,:]) == 0:
                count += 1
        log.info("0 sum ratio = {0}".format(count/self.maskMatrix.shape[0]))

        count = 0.0
        for i in range(self.maskMatrix.shape[0]):
            if sum(self.maskMatrix[i,:]) == 10:
                count += 1
        log.info("10 sum ratio = {0}".format(count/self.maskMatrix.shape[0]))



        # self.maskMatrix = self.maskMatrix[:self.nsamplesAdded,:]
        # self.ey = self.ey[:self.nsamplesAdded]
        # self.kernelWeights = self.kernelWeights[:self.nsamplesAdded]
        log.debug("self.maskMatrix.shape = {0}".format(self.maskMatrix.shape))
        # adjust the y value according to the constraints for the offset and sum
        log.debug("self.link(self.fnull) = {0}".format(self.link.f(self.fnull)))
        log.debug("self.link(self.fx) = {0}".format(self.link.f(self.fx)))
        # for i in range(self.maskMatrix.shape[0]):
        #     log.debug("{0} {1} {2}".format(self.maskMatrix[i,:], self.ey[i], self.kernelWeights[i]))
        eyAdj = self.linkfv(self.ey) - self.link.f(self.fnull)

        s = np.sum(self.maskMatrix, 1)


        # do feature selection
        w_aug = np.hstack((self.kernelWeights * (self.M-s), self.kernelWeights*s))
        log.info("np.sum(w_aug) = {0}".format(np.sum(w_aug)))
        log.info("np.sum(self.kernelWeights) = {0}".format(np.sum(self.kernelWeights)))
        w_sqrt_aug = np.sqrt(w_aug)
        eyAdj_aug = np.hstack((eyAdj, eyAdj - (self.link.f(self.fx) - self.link.f(self.fnull))))
        eyAdj_aug *= w_sqrt_aug
        mask_aug = np.transpose(w_sqrt_aug*np.transpose(np.vstack((self.maskMatrix, self.maskMatrix-1))))
        var_norms = np.array([np.linalg.norm(mask_aug[:,i]) for i in range(mask_aug.shape[1])])
        #mask_aug /= var_norms
        # print(self.kernelWeights)
        # print(w_aug)


        model = LassoLarsIC(criterion='bic', normalize=True)#fit_intercept
        #model = Lasso(alpha=self.l1reg, fit_intercept=True)
        model.fit(mask_aug, eyAdj_aug)
        nonzero_inds = np.nonzero(model.coef_)[0]
        # for i in range(mask_aug.shape[0]):
        #     log.info("{0} {1} {2}".format(mask_aug[i,:], self.ey[i], self.kernelWeights[i]))
        log.info("model.get_params() = {0}".format(model.get_params()))
        #log.info("model.alpha_ = {0}".format(model.alpha_))
        log.info("model.coef_ = {0}".format(model.coef_))
        log.info("nonzero_inds = {0}".format(nonzero_inds))

        w1 = np.dot(np.linalg.inv(np.dot(np.transpose(mask_aug),mask_aug)),np.dot(np.transpose(mask_aug), eyAdj_aug))
        log.info("w1 = {0}".format(w1))

        w1 = np.dot(np.linalg.inv(np.dot(np.transpose(mask_aug),mask_aug)),np.dot(np.transpose(mask_aug), eyAdj_aug))
        log.info("w1 = {0}".format(w1))

        #np.transpose(self.maskMatrix) * self.kernelWeights

        #w = np.dot(np.linalg.inv(np.dot(np.transpose(mask_aug),mask_aug)),np.dot(np.transpose(mask_aug), eyAdj_aug))

        # eyAdj1 = eyAdj - self.maskMatrix[:,-1]*(self.link(self.fx) - self.link(self.fnull))
        # etmp = self.maskMatrix[:,:-1] - self.maskMatrix[:,-1:]
        # var_norms = np.array([np.linalg.norm(etmp[:,i]) for i in range(etmp.shape[1])])
        # etmp /= var_norms
        # print(var_norms)
        # model_bic = LassoLarsIC(criterion='bic')
        # model_bic.fit(etmp, eyAdj1)
        # nonzero_inds = np.nonzero(model_bic.coef_)[0]
        # print(nonzero_inds.shape)
        # # solve a weighted least squares equation to estimate phi
        # print(self.maskMatrix[:,nonzero_inds[-1]].shape)
        # print(nonzero_inds)
        #nonzero_inds = np.arange(self.M)

        eyAdj2 = eyAdj - self.maskMatrix[:,nonzero_inds[-1]]*(self.link.f(self.fx) - self.link.f(self.fnull))
        etmp = np.transpose(np.transpose(self.maskMatrix[:,nonzero_inds[:-1]]) - self.maskMatrix[:,nonzero_inds[-1]])
        #print(self.maskMatrix)
        log.debug("etmp[1:4,:] {0}".format(etmp[0:4,:]))
        # etmp = self.maskMatrix
        # eyAdj2 = eyAdj
        # solve a weighted least squares equation to estimate phi
        tmp = np.transpose(np.transpose(etmp) * np.transpose(self.kernelWeights))
        #tmp = etmp
        # log.debug("tmp.shape", tmp.shape)
        # log.debug("tmp.shape", tmp.shape)
        tmp2 = np.linalg.inv(np.dot(np.transpose(tmp),etmp))
        w = np.dot(tmp2,np.dot(np.transpose(tmp),eyAdj2))
        #log.info("w = {0}".format(w))
        log.debug("np.sum(w) = {0}".format(np.sum(w)))
        log.debug("self.link(self.fx) - self.link(self.fnull) = {0}".format(self.link.f(self.fx) - self.link.f(self.fnull)))
        phi = np.zeros(self.M)
        phi[nonzero_inds[:-1]] = w
        phi[nonzero_inds[-1]] = (self.link.f(self.fx) - self.link.f(self.fnull)) - sum(w)
        log.info("phi = {0}".format(phi))

        # yHat = np.dot(self.maskMatrix, w)
        # phiVar = np.var(yHat - eyAdj) * np.diag(tmp2)
        # phiVar = np.hstack((phiVar, max(phiVar))) # since the last weight is inferred we use a pessimistic guess of its variance

        # a finite sample adjustment based on how much of the weight is left in the sample space
        # fractionWeightLeft = 1 - sum(self.kernelWeights)/sum(np.array([(self.M-1)/(s*(self.M-s)) for s in range(1, self.M)]))

        return phi,np.ones(len(phi))#phiVar*fractionWeightLeft

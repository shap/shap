'''
This module extends the Kernel SHAP explainer as proposed by Scott Lundberg* and Su-In Lee**, in their paper
"A Unified Approach to Interpreting Model Predictions". The here defined function extends the Kernel SHAP method from
computing local feature importance to computing global feature importance values. Moreover, the global feature importance measures
are 'standardised' in terms of the increase/decrease in accuracy they provide to the model. This latter point
is inspired by the approach taken by Paolo Giudici+ and Emanueala Raffinetti++, in their paper
"Shapley-Lorenz decompositions in exPlainable Artificial Intelligence", where they standardise
Shapley values, by considering their Gini coefficient, i.e. how much variance of the true variance, can be
attributed to each feature.

*slund1@cs.washington.edu
**suinlee@cs.washington.edu
+paolo.giudici@unipv.it
++emanuela.raffinetti@unimi.it
'''

import auxiliary_functions
import numpy as np
import pandas as pd
import scipy as sp
import itertools
import copy
from tqdm.auto import tqdm
import logging
import warnings

log = logging.getLogger('altShap')

class altShapExplainer:
    '''
    Uses the SHAP approach as implemented by Lundberg and Lee, but instead of calculating
    local feature value contributions, this method calculates global feature value
    contributions. Moreover, the feature value contributions are in terms of contribution
    to a measure of accuracy of choice. In addition, the global feature importance measures
    account for feature dependence.

    Parameters:
    ------------------------------------------------------------------------------------
    model : function or iml.Modle
        function that has a matrix of sample as input and outputs a predition for those
        samples
    
    X_background : nump.array or pandas.DataFrame
        background data used to mask features, i.e. integrate out features.

    '''
    def __init__(self, model, X_background):
        self.model = convert_to_model(model) # standardise model
        self.data = convert_to_data(X_background) # standardise data format
        self.N = self.data.data.shape[0]
        self.M = self.data.data.shape[1]

        if self.N > 100:
            log.warning('A large background dataset of size ' + str(self.N) + ' could cause +'
            + 'could cause slow run times. A background dataset of size 100 is recommended.')
        
        # initiate counters
        self.nsamplesAdded = 0
        self.nsamplesRun = 0


    def altshap_values(self, X, y, score_measure = accuracy_score, **kwargs):
        '''
        Estimates alternative global SHAP values for the given sample.

        Parameters:
        ------------------------------------------------------------------------------------
        X : numpy.array or pandas.DataFrame
            matrix of samples (#samples x #features) for which the model output is explained
        
        y : numpy.array
            vector of samples corresponding to the observed output, given the input matrix X
        
        score_measure : str (default = 'accuracy_score')
            method to use as measure for model accuracy

        Returns:
        ------------------------------------------------------------------------------------
        A M x 1 vector, corresponding to the estimated global feature importance values.
        '''

        # Standardise data
        if str(type(X)).endswith("pandas.core.series.Series'>"):
            X = X.values
        elif str(type(X)).endswith("pandas.core.frame.DataFrame'>"):
            index_value = X.index.values
            index_name = X.index.name
            column_names = X.columns.tolist()
            X = X.values

        # Conditions on X
        assert(X.shape[0] == self.N), 'background data and feature matrix need to be of same length'
        assert len(X.shape) == 2, 'input data needs to be a matrix, i.e. at least two features'

        # Conditions on y
        assert str(type(y)).endswith('numpy.array') or str(type(y)).endswith("series.Series'>"), 'response variable needs to be of numpy array or pandas.core.series.Series format'
        assert len(y) == self.N, 'response vector needs to have same # of obs as feature matrix'
        self.ytrue = y
    
        # Null model
        self.fnull = self.model.f(self.data.data)
        self.measure_null = accuracy_score(self.ytrue, self.fnull)

        # Full model
        self.fX = self.model.f(X)
        self.measure_fX = accuracy_score(self.ytrue, self.fX)

        # Explanations
        explanations = []
        explanations.append(self.glob_explain(X))

        #s = explanations[0].shape
        #out = np.zeros(s[0])
        #out[:] = explanations
        return np.array([explanations]).reshape((self.M,1))

    # Explain dataset 
    def glob_explain(self, X, **kwargs):

        # 1. BUILD SYNTHETIC DATA

        # Define number of samples to take from background data
        self.nsamples = kwargs.get('nsamples', 'auto')
        if self.nsamples == 'auto':
            self.nsamples = 2*self.M+2**11

        # make some preperations
        self.prep()

        # create weight vector
        num_subset_sizes = np.int(np.ceil(self.M - 1) / 2.0) # only half of all possible
                                                            # subset sizes, due to symmetrical property
                                                            # of binomial coefficient
        num_paired_subset_sizes = np.int(np.floor(self.M - 1) / 2.0)
        # weight vector --> SHAP kernel without binomial coefficient: (M-1) / (S*(M-S))
        weight_vector = np.array([(self.M - 1.0) / (i*(self.M-i)) for i in range(1, num_subset_sizes + 1)])
        weight_vector[:num_paired_subset_sizes] *= 2 # for the inverse (e.g. inverse of S=1 is
                                                        # S=M-1 --> have same kernel)
        weight_vector /= np.sum(weight_vector) # normalise s.t. between [0,1], to use for sampling within
                                                # subsets later on

        # create synthetic data for subsets that can be completely enumerated (i.e. fill out
        # subsets of size s, s.t. MChooseS <= nsamplesleft)
        num_full_subsets = 0
        num_samples_left = self.nsamples
        mask = np.zeros(self.M) # vector to mask features for current subset size (later transposed)
        remaining_weight_vector = weight_vector.copy() # to reweight weight_vector if one subset completed

        # loop over subsets (N.B. zero subset and full subset are left out, as kernel weight is infinite
        # for S = 0 and S = M)
        for subset_size in range(1, num_subset_sizes + 1):

            # find number of subsets and their inverse for current subset_size
            nsubsets = binom(self.M, subset_size)
            if subset_size <= num_paired_subset_sizes: nsubsets *= 2 # times 2 to account for inverse

            # check if have enough samples to completely enumerate number of subsets
            if num_samples_left * remaining_weight_vector[subset_size - 1] / nsubsets >= 1.0 - 1e-8:
                num_full_subsets += 1
                num_samples_left -= nsubsets

                # rescale weight vector s.t. remaining weight vector, i.e. from next subset size onwards
                # sums to 1 - iff last weight is not yet 1 already
                if remaining_weight_vector[subset_size - 1] < 1.0:
                    remaining_weight_vector /= (1 - remaining_weight_vector[subset_size - 1])

                # add samples of current subset size to the synthetic data set and create corresponding
                # masks
                w = weight_vector[subset_size - 1] / binom(self.M, subset_size) # create kernel
                if subset_size <= num_paired_subset_sizes: w/= 2.0 # get weight for one subset
                                                                    # previously was for two
                # loop over all possible subset combinations of size subset_size
                group_indx = np.arange(self.M, dtype='int64')
                for groups in itertools.combinations(group_indx, subset_size):
                    mask[:] = 0.0 # reset to zero
                    mask[np.array(groups, dtype = 'int64')] = 1.0 # [*]
                    self.addsample(X, mask, w) # use addsample function to add samples to the
                                                # background (masking) dataset, i.e. to unmask
                                                # features, as defined by mask
                    if subset_size <= num_paired_subset_sizes:
                        mask[:] = np.abs(mask - 1) # get the inverse/complement to [*]
                        self.addsample(X, mask, w)
            else:
                break

        # from left-over nsamples, enumerate over random subset spaces, iff any subset spaces left
        nfixed_samples = self.nsamplesAdded
        samples_left = self.nsamples - nfixed_samples

        # check that the number of full subsets has not reached the number of subset sizes
        if num_full_subsets != num_subset_sizes:
            remaining_weight_vector = weight_vector.copy() # reset weight vector
            remaining_weight_vector[:num_paired_subset_sizes] /= 2 # get weight for one subset
                                                                    # previously was for two
            remaining_weight_vector = remaining_weight_vector[num_full_subsets:] # throw away
                                                            # weight vector for full subsizes
            remaining_weight_vector /= np.sum(remaining_weight_vector) # reweight to sum to 1
            indx_set = np.random.choice(len(remaining_weight_vector), 6 * samples_left,\
                                            p = remaining_weight_vector) # create a weighted
                                            # random sample of subsets of size
                                            # 6*samples_left to randomly choose from
                                            # to enumerate (weights are given by weight vector)
            indx_pos = 0
            used_masks = {} # create a dictionary of used masks, to keep tab on used
                            # subset sizes
            # loop over left over samples
            while samples_left > 0 and indx_pos < len(indx_set):
                mask.fill(0.0) # reset to zero
                indx = indx_set[indx_pos] # pick a random subset size generated by indx_set
                                            # (only generated once to save time here)
                indx_pos += 1
                subset_size = indx + num_full_subsets + 1 # adjust subset size, for
                                            # already considered subset sizes, s.t.
                                            # already fully enumerated subset
                                            # sizes are not considered again
                mask[np.random.permutation(self.M)[:subset_size]] = 1.0 # randomly
                                                # switch on features, s.t. total switched
                                                # on features is equal the selected subset_size
                
                # check if a sample of the current subset size has already been addded.
                # If so, a previous sample's weight is incremented
                mask_tuple = tuple(mask)
                new_sample = False
                if mask_tuple not in used_masks:
                    new_sample = True
                    used_masks[mask_tuple] = self.nsamplesAdded # update dicitonary of seen
                                                                # samples
                    samples_left -= 1
                    self.addsample(X, mask, 1.0) # add samples to the background data set
                else:
                    self.kernelWeights[used_masks[mask_tuple]] += 1.0

                # inverse selected features and add sample
                if samples_left > 0 and subset_size <= num_paired_subset_sizes:
                    mask[:] = np.abs(mask - 1) # inverse selected features
                    
                    # again check if sample of current subset size has already been added
                    if new_sample:
                        samples_left -= 1
                        self.addsample(X, mask, 1.0) # add samples to the background data set
                    else:
                        # compliment sample is always the next one thus just skip to next row
                        self.kernelWeights[used_masks[mask_tuple] + 1] += 1.0

            # normalise kernel weights
            weights_left_sum = np.sum(weight_vector[num_full_subsets:])
            self.kernelWeights[nfixed_samples:] *= weights_left_sum / self.kernelWeights[nfixed_samples:]

        # 2. GET PREDICTIONS AND CORRESPONDING SCORE MEASURE

        self.run()

        # 3. SOLVE FOR ALTERNATIVE GLOBAL SHAP IMPORTANCE VALUES

        phi = np.zeros((self.M))
        vphi, vphi_var = self.solve()

        phi = vphi

        return phi


    # Notes
    # sample from background data large sample


    # Auxiliary functions

    # -- create synthetic data and containers
    def prep(self):
        # synthetic data
        self.synth_data = np.tile(self.data.data, (self.nsamples,1))

        # containers
        self.maskMatrix = np.zeros((self.nsamples, self.M))
        self.kernelWeights = np.zeros(self.nsamples)
        self.nonzero_indx = np.zeros(self.nsamples)
        self.measure = np.zeros((self.nsamples,1))
        self.y = np.zeros((self.nsamples*self.N,1))

        # counters
        self.nsamplesAdded = 0
        self.nsamplesRun = 0

    # -- unmask features
    def addsample(self, X, m, w):
        shift = self.nsamplesAdded * self.N
        #dist = self.dist_parameters(X, indx)
        #X_sbar = np.random.multivariate_normal(dist.mu_sbar_s, sigma_sbar_s, self.N)
        for k in range(self.M):
            if m[k] == 1.0:
                self.synth_data[shift:shift+self.N, k] = X[:, k]

        self.maskMatrix[self.nsamplesAdded, :] = m
        self.kernelWeights[self.nsamplesAdded] = w
        self.nsamplesAdded += 1

    # -- run model
    def run(self):
        num_to_run = self.N * (self.nsamplesAdded - self.nsamplesRun)
        data = self.synth_data[self.nsamplesRun*self.N : self.nsamplesAdded*self.N,:]
        
        modelOut = self.model.f(data)
        if isinstance(modelOut, (pd.DataFrame, pd.Series)):
            modelOut = modelOut.values
        self.y[:self.nsamples*self.N] = np.reshape(modelOut,(num_to_run, 1))

        for i in range(1,self.nsamples):
            #print(measure[self.nsamplesRun].shape)
            #print(accuracy_score(self.ytrue, self.y[self.nsamplesRun*self.N: i*self.N]))
            self.measure[self.nsamplesRun] = accuracy_score(self.ytrue, self.y[self.nsamplesRun*self.N: i*self.N])
            self.nsamplesRun += 1

    def solve(self):
        #for i in range(self.maskMatrix.shape[0]):
            #print(self.maskMatrix.shape)
         #   self.nonzero_indx[i] = np.array(np.nonzero(self.maskMatrix[i,:])[0].tolist())
        nonzero_indx = np.arange(self.M)
        measure_k = self.measure - self.measure_null
        measure_dif = measure_k - np.array([self.maskMatrix[:, nonzero_indx[-1]]*(self.measure_fX - self.measure_null)]).reshape((self.nsamples, 1)) # nsamples x 1
        Z = np.transpose(np.transpose(self.maskMatrix[:, nonzero_indx[:-1]])\
                        - self.maskMatrix[:, nonzero_indx[-1]]) # nsamples x M
        ZW = np.transpose(np.transpose(Z) * np.transpose(self.kernelWeights)) # nsamples x M
        ZWZ_inv = np.linalg.inv(np.dot(np.transpose(ZW), Z)) # M x M
        wlsq_result = np.dot(ZWZ_inv, np.dot(np.transpose(ZW), measure_dif)) # M x 1

        result = np.zeros(self.M)
        result[nonzero_indx[:-1]] = wlsq_result.reshape(-1)
        result[nonzero_indx[-1]] = (self.measure_fX - self.measure_null) - sum(wlsq_result)

        return np.array([result]), np.ones(len(result))

    # -- find multivariate distribution
    def dist_parameters(self, X, indx):
        m = indx
        setall = np.arange(self.M)
        m_bar = [item for item in setall if item not in m]
        X_s = X[:,m]
        X_sbar = self.data.data[:,m_bar]
        mu_s = np.mean(X_s)
        mu_sbar = np.mean(X_sbar)

        self.mu_sbar_s = np.zeros((X.shape[0], self.M))
        self.sigma_sbar_s = np.zeros((self.M, self.M))
        
        if len(m) == 1:
            sigma_ss = np.var(X_s)
        else:
            sigma_ss = np.cov(X_s, rowvar = False ,bias = False)
        sigma_s_sbar = np.cov(X_s, y = X_sbar, rowvar = False, bias = False)
        sigma_sbar_s = np.cov(X_sbar, y = X_s, rowvar = False, bias = False)
        if len(m_bar) == 1:
            sigma_sbar_sbar = np.var(X_sbar)
        else:
            sigma_sbar_sbar = np.cov(X_sbar, rowvar = False, bias = False)

        #print('sigma: {}'.format(sigma_ss))
        #print('X_sbar shape: {}'.format(X_sbar.shape))
        print('m_bar: {}'.format(m_bar))
        print('m_bar: {}'.format(len(m_bar)))

        for i in range(X_s.shape[1]):
            if len(m) == 1:
                self.mu_sbar_s[:,i] = mu_sbar + (sigma_sbar_s/sigma_ss) * (X_s[:,i] - mu_s)
            else:
                self.mu_sbar_s[:,i] = mu_sbar + np.dot(sigma_sbar_s, np.linalg.inv(sigma_ss)) * (X_s[:,i] - mu_s)
        
        if len(m) == 1:
            self.sigma_sbar_s = sigma_sbar_sbar - np.dot((sigma_sbar_s / sigma_ss), sigma_s_sbar)
        else:
            self.sigma_sbar_s = sigma_sbar_sbar - np.dot(np.dot(sigma_sbar_s, np.linalg.inv(sigma_ss)), sigma_s_sbar)

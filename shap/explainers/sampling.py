from ..common import convert_to_instance, convert_to_model, match_instance_to_data, match_model_to_data, convert_to_instance_with_index, convert_to_link, IdentityLink, convert_to_data, DenseData
from .kernel import KernelExplainer
import numpy as np
import pandas as pd
import logging

log = logging.getLogger('shap')


class SamplingExplainer(KernelExplainer):
    """ This is an extension of the Shapley sampling values explanation method (aka. IME)

    SamplingExplainer computes SHAP values under the assumption of feature independence and is an
    extension of the algorithm proposed in "An Efficient Explanation of Individual Classifications
    using Game Theory", Erik Strumbelj, Igor Kononenko, JMLR 2010. It is a good alternative to
    KernelExplainer when you want to use a large background set (as opposed to a single reference
    value for example).

    Parameters
    ----------
    model : function
        User supplied function that takes a matrix of samples (# samples x # features) and
        computes a the output of the model for those samples. The output can be a vector
        (# samples) or a matrix (# samples x # model outputs).

    data : numpy.array or pandas.DataFrame
        The background dataset to use for integrating out features. To determine the impact
        of a feature, that feature is set to "missing" and the change in the model output
        is observed. Since most models aren't designed to handle arbitrary missing data at test
        time, we simulate "missing" by replacing the feature with the values it takes in the
        background dataset. So if the background dataset is a simple sample of all zeros, then
        we would approximate a feature being missing by setting it to zero. Unlike the
        KernelExplainer this data can be the whole training set, even if that is a large set. This
        is because SamplingExplainer only samples from this background dataset.
    """

    def __init__(self, model, data, **kwargs):
        # silence warning about large datasets
        level = log.level
        log.setLevel(logging.ERROR)
        super(SamplingExplainer, self).__init__(model, data, **kwargs)
        log.setLevel(level)

        assert str(self.link) == "identity", "SamplingExplainer only supports the identity link not " + str(self.link)

    def explain(self, incoming_instance, **kwargs):
        # convert incoming input to a standardized iml object
        instance = convert_to_instance(incoming_instance)
        match_instance_to_data(instance, self.data)

        assert len(self.data.groups) == self.P, "SamplingExplainer does not support feature groups!"

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
            phi_var = np.zeros((len(self.data.groups), self.D))

        # if only one feature varies then it has all the effect
        elif self.M == 1:
            phi = np.zeros((len(self.data.groups), self.D))
            phi_var = np.zeros((len(self.data.groups), self.D))
            diff = self.fx - self.fnull
            for d in range(self.D):
                phi[self.varyingInds[0],d] = diff[d]

        # if more than one feature varies then we have to do real work
        else:

            # pick a reasonable number of samples if the user didn't specify how many they wanted
            self.nsamples = kwargs.get("nsamples", "auto")
            if self.nsamples == "auto":
                self.nsamples = 1000 * self.M
            assert self.nsamples % 2 == 0, "nsamples must be divisible by 2!"

            min_samples_per_feature = kwargs.get("min_samples_per_feature", 100)
            round1_samples = self.nsamples
            round2_samples = 0
            if round1_samples > self.M * min_samples_per_feature:
                round2_samples = round1_samples - self.M * min_samples_per_feature
                round1_samples -= round2_samples

            # divide up the samples among the features for round 1
            nsamples_each1 = np.ones(self.M, dtype=np.int64) * 2 * (round1_samples // (self.M * 2))
            for i in range((round1_samples % (self.M * 2)) // 2):
                nsamples_each1[i] += 2

            # explain every feature in round 1
            phi = np.zeros((self.P, self.D))
            phi_var = np.zeros((self.P, self.D))
            self.X_masked = np.zeros((nsamples_each1.max(), self.data.data.shape[1]))
            for i,ind in enumerate(self.varyingInds):
                phi[ind,:],phi_var[ind,:] = self.sampling_estimate(ind, self.model.f, instance.x, self.data.data, nsamples=nsamples_each1[i])

            # optimally allocate samples according to the variance
            if phi_var.sum() == 0:
                phi_var += 1 # spread samples uniformally if we found no variability
            phi_var /= phi_var.sum()
            nsamples_each2 = (phi_var[self.varyingInds,:].mean(1) * round2_samples).astype(np.int)
            for i in range(len(nsamples_each2)):
                if nsamples_each2[i] % 2 == 1: nsamples_each2[i] += 1
            for i in range(len(nsamples_each2)):
                if nsamples_each2.sum() > round2_samples:
                    nsamples_each2[i] -= 2
                elif nsamples_each2.sum() < round2_samples:
                    nsamples_each2[i] += 2
                else:
                    break

            self.X_masked = np.zeros((nsamples_each2.max(), self.data.data.shape[1]))
            for i,ind in enumerate(self.varyingInds):
                if nsamples_each2[i] > 0:
                    val,var = self.sampling_estimate(ind, self.model.f, instance.x, self.data.data, nsamples=nsamples_each2[i])

                    total_samples = nsamples_each1[i] + nsamples_each2[i]
                    phi[ind,:] = (phi[ind,:] * nsamples_each1[i] + val * nsamples_each2[i]) / total_samples
                    phi_var[ind,:] = (phi_var[ind,:] * nsamples_each1[i] + var * nsamples_each2[i]) / total_samples

            # convert from the variance of the differences to the variance of the mean (phi)
            for i,ind in enumerate(self.varyingInds):
                phi_var[ind,:] /= np.sqrt(nsamples_each1[i] + nsamples_each2[i])

            # correct the sum of the SHAP values to equal the output of the model using a linear
            # regression model with priors of the coefficents equal to the estimated variances for each
            # SHAP value (note that 1e6 is designed to increase the weight of the sample and so closely
            # match the correct sum)
            sum_error = self.fx - phi.sum(0) - self.fnull
            for i in range(self.D):
                # this is a ridge regression with one sample of all ones with sum_error[i] as the label
                # and 1/v as the ridge penalties. This simlified (and stable) form comes from the
                # Sherman-Morrison formula
                v = (phi_var[:,i] / phi_var[:,i].max()) * 1e6
                adj = sum_error[i] * (v - (v * v.sum()) / (1 + v.sum()))
                phi[:,i] += adj

        if phi.shape[1] == 1:
            phi = phi[:,0]

        return phi

    def sampling_estimate(self, j, f, x, X, nsamples=10):
        assert nsamples % 2 == 0, "nsamples must be divisible by 2!"
        X_masked = self.X_masked[:nsamples,:]
        inds = np.arange(X.shape[1])

        for i in range(0, nsamples//2):
            np.random.shuffle(inds)
            pos = np.where(inds == j)[0][0]
            rind = np.random.randint(X.shape[0])
            X_masked[i, :] = x
            X_masked[i, inds[pos+1:]] = X[rind, inds[pos+1:]]
            X_masked[-(i+1), :] = x
            X_masked[-(i+1), inds[pos:]] = X[rind, inds[pos:]]

        evals = f(X_masked)
        evals_on = evals[:nsamples//2]
        evals_off = evals[nsamples//2:][::-1]
        d = evals_on - evals_off

        return np.mean(d, 0), np.var(d, 0)

from iml.common import convert_to_instance, convert_to_model, match_instance_to_data, match_model_to_data
from iml.explanations import AdditiveExplanation
from iml.links import convert_to_link, IdentityLink
from iml.datatypes import convert_to_data, DenseData
from iml import initjs, Instance, Model
import iml
from scipy.special import binom
import numpy as np
import logging
import copy
import itertools
from sklearn.linear_model import LassoLarsIC

try:
    import xgboost
except ImportError:
    pass

try:
    import lightgbm
except ImportError:
    pass

try:
    import matplotlib.pyplot as pl
    from matplotlib.colors import LinearSegmentedColormap

    cdict1 = {'red':   ((0.0, 0.11764705882352941, 0.11764705882352941),
                        (1.0, 0.9607843137254902, 0.9607843137254902)),

             'green': ((0.0, 0.5333333333333333, 0.5333333333333333),
                       (1.0, 0.15294117647058825, 0.15294117647058825)),

             'blue':  ((0.0, 0.8980392156862745, 0.8980392156862745),
                       (1.0, 0.3411764705882353, 0.3411764705882353))
            }
    red_blue = LinearSegmentedColormap('RedBlue', cdict1)
except ImportError:
    pass

log = logging.getLogger('shap')


def joint_plot(x, y, joint_shap_values, xname, yname, axis_color="#000000", show=True):
    if type(x[0]) == str:
        xnames = list(set(x))
        xnames.sort()
        name_map = {n: i for i,n in enumerate(xnames)}
        xv = [name_map[v] for v in x]
    else:
        xv = x

    if type(y[0]) == str:
        ynames = list(set(y))
        ynames.sort()
        name_map = {n: i for i,n in enumerate(ynames)}
        yv = [name_map[v] for v in y]
    else:
        yv = y

    sc = pl.scatter(x, y, s=20, c=joint_shap_values, edgecolor='', alpha=1, cmap=red_blue)
    pl.xlabel(xname, color=axis_color)
    pl.ylabel(yname, color=axis_color)
    pl.colorbar(sc, label="Joint SHAP value")

    pl.gca().tick_params(color=axis_color, labelcolor=axis_color)
    for spine in pl.gca().spines.values():
        spine.set_edgecolor(axis_color)
    if type(x[0]) == str:
        pl.xticks([name_map[n] for n in xnames], xnames, rotation='vertical')
    if show:
        pl.show()


def plot(x, shap_values, name, color="#ff0052", axis_color="#333333", alpha=1, title=None, show=True):
    if type(x[0]) == str:
        xnames = list(set(x))
        xnames.sort()
        name_map = {n: i for i,n in enumerate(xnames)}
        xv = [name_map[v] for v in x]
    else:
        xv = x

    pl.plot(xv, shap_values, ".", markersize=5, color=color, alpha=alpha)

    # make the plot more readable
    pl.xlabel(name, color=axis_color)
    pl.ylabel("SHAP value for "+name, color=axis_color)
    if title != None:
        pl.title("SHAP plot for "+name, color=axis_color, fontsize=11)
    pl.gca().xaxis.set_ticks_position('bottom')
    pl.gca().yaxis.set_ticks_position('left')
    pl.gca().spines['right'].set_visible(False)
    pl.gca().spines['top'].set_visible(False)
    pl.gca().tick_params(color=axis_color, labelcolor=axis_color)
    for spine in pl.gca().spines.values():
        spine.set_edgecolor(axis_color)
    if type(x[0]) == str:
        pl.xticks([name_map[n] for n in xnames], xnames, rotation='vertical')
    if show:
        pl.show()

def interactions(X, shap_values, index):
    """ Order other features by how much interaction they have with the feature at the given index. """
    if X.shape[0] > 10000:
        a = np.arange(X.shape[0])
        np.random.shuffle(a)
        inds = a[:10000]
    else:
        inds = np.arange(X.shape[0])

    x = X[inds,index]
    srt = np.argsort(x)
    shap_ref = shap_values[inds,index]
    shap_ref = shap_ref[srt]
    inc = int(len(x)/10.0)
    interactions = []
    for i in range(X.shape[1]):
        shap_other = shap_values[inds,i][srt]

        if i == index or np.sum(np.abs(shap_other)) < 1e-8:
            v = 0
        else:
            v = np.sum(np.abs([np.corrcoef(shap_ref[i:i+inc],shap_other[i:i+inc])[0,1] for i in range(0,len(x),inc)]))
        interactions.append(v)

    return np.argsort(-np.abs(interactions))

def interaction_plot(ind, X, shap_value_matrix, feature_names=None, show_interaction=False, color="#ff0052", axis_color="#333333", alpha=1, title=None, show=True):

    # convert from a DataFrame if we got one
    if str(type(X)) == "<class 'pandas.core.frame.DataFrame'>":
        if feature_names is None:
            feature_names = X.columns
        X = X.as_matrix()

    x = X[:,ind]
    name = feature_names[ind]
    shap_values = shap_value_matrix[:,ind]
    if type(x[0]) == str:
        xnames = list(set(x))
        xnames.sort()
        name_map = {n: i for i,n in enumerate(xnames)}
        xv = [name_map[v] for v in x]
    else:
        xv = x

    top_interaction = interactions(X, shap_value_matrix, ind)[0]
    pl.scatter(xv, shap_values, s=12.0, linewidth=0, c=shap_value_matrix[:,top_interaction], cmap=red_blue, alpha=alpha)
    cb = pl.colorbar(label="SHAP value for "+feature_names[top_interaction])
    cb.set_alpha(1)
    cb.draw_all()
    # make the plot more readable
    pl.xlabel(name, color=axis_color)
    pl.ylabel("SHAP value for "+name, color=axis_color)
    if title != None:
        pl.title("SHAP plot for "+name, color=axis_color, fontsize=11)
    pl.gca().xaxis.set_ticks_position('bottom')
    pl.gca().yaxis.set_ticks_position('left')
    pl.gca().spines['right'].set_visible(False)
    pl.gca().spines['top'].set_visible(False)
    pl.gca().tick_params(color=axis_color, labelcolor=axis_color)
    for spine in pl.gca().spines.values():
        spine.set_edgecolor(axis_color)
    if type(x[0]) == str:
        pl.xticks([name_map[n] for n in xnames], xnames, rotation='vertical')
    if show:
        pl.show()

def summary_plot(shap_values, feature_names, max_display=20, color="#ff0052", axis_color="#333333", title=None, show=True):
    ind_order = np.argsort(np.sum(np.abs(shap_values), axis=0)[:-1])
    ind_order = ind_order[-min(max_display,len(ind_order)):]
    fig = pl.figure(figsize=(5,len(ind_order)*0.35))
    pl.axvline(x=0, color="#999999")

    for pos,i in enumerate(ind_order):
        pl.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1,5))
        pl.plot(shap_values[:,i], np.ones(shap_values.shape[0])*pos, ".", color=color, markersize=7, alpha=0.1, markeredgewidth=0)
    pl.gca().xaxis.set_ticks_position('bottom')
    pl.gca().yaxis.set_ticks_position('none')
    pl.gca().spines['right'].set_visible(False)
    pl.gca().spines['top'].set_visible(False)
    pl.gca().spines['left'].set_visible(False)
    pl.gca().tick_params(color=axis_color, labelcolor=axis_color)
    pl.yticks(range(len(ind_order)), [feature_names[i] for i in ind_order])
    pl.gca().tick_params('y', length=20, width=0.5, which='major')
    pl.ylim(-1, len(ind_order))
    pl.xlabel("SHAP value (impact on model output)")
    pl.show()

def visualize(shap_values, feature_names=None, data=None, out_names=None):
    if type(shap_values) != np.ndarray:
        return iml.visualize(shap_values)

    if len(shap_values.shape) == 1:
        shap_values = np.reshape(shap_values, (1,len(shap_values)))

    if out_names is None:
        out_names = ["output value"]

    if shap_values.shape[0] == 1:
        if feature_names is None:
            feature_names = ["" for i in range(shap_values.shape[1]-1)]
        if data is None:
            data = ["" for i in range(len(feature_names))]
        if type(data) == np.ndarray:
            data = data.flatten()

        instance = Instance(np.zeros((1,len(feature_names))), data)
        e = AdditiveExplanation(
            shap_values[0,-1],
            np.sum(shap_values[0,:]),
            shap_values[0,:-1],
            None,
            instance,
            IdentityLink(),
            Model(None, out_names),
            DenseData(np.zeros((1,len(feature_names))), list(feature_names))
        )
        return e

    else:
        exps = []
        for i in range(shap_values.shape[0]):
            if feature_names is None:
                feature_names = ["" for i in range(shap_values.shape[1]-1)]
            if data is None:
                display_data = ["" for i in range(len(feature_names))]
            else:
                display_data = data[i,:]

            instance = Instance(np.ones((1,len(feature_names))), display_data)
            e = AdditiveExplanation(
                shap_values[i,-1],
                np.sum(shap_values[i,:]),
                shap_values[i,:-1],
                None,
                instance,
                IdentityLink(),
                Model(None, out_names),
                DenseData(np.ones((1,len(feature_names))), list(feature_names))
            )
            exps.append(e)
        return exps

def explain(model, data=None, feature_names=None, out_names=None):
    if data is None:
        return explain_model(model, feature_names, out_names)
    elif len(data.shape) == 1:
        return explain_instance(model, np.reshape(data, (1,len(data))), feature_names, out_names)
    elif data.shape[0] == 1:
        return explain_instance(model, data, feature_names, out_names)
    else:
        return explain_instances(model, data, feature_names, out_names)


def explain_instance(model, data, feature_names, out_names):
    if out_names is None:
        out_names = ["model output"]
    if feature_names is None:
        feature_names = [(i+1)+"" for i in range(data.shape[1])]

    if type(model) == xgboost.core.Booster:
        contribs = model.predict(xgboost.DMatrix(data), pred_contribs=True)
    elif type(model) == lightgbm.basic.Booster:
        contribs = model.predict(data, pred_contrib=True)
    else:
        return None

    instance = Instance(data[0:1,:], data[0,:])
    e = AdditiveExplanation(
        contribs[0,-1],
        np.sum(contribs[0,:]),
        contribs[0,:-1],
        None,
        instance,
        IdentityLink(),
        Model(None, out_names),
        DenseData(np.zeros((1,data.shape[1])), list(feature_names))
    )
    return e

def explain_instances(model, data, feature_names, out_names):
    if out_names is None:
        out_names = ["model output"]
    if feature_names is None:
        feature_names = [(i+1)+"" for i in range(data.shape[1])]

    if type(model) == xgboost.core.Booster:
        exps = []
        contribs = model.predict(xgboost.DMatrix(data), pred_contribs=True)
        for i in range(data.shape[0]):
            instance = Instance(data[i:i+1,:], data[i,:])
            e = AdditiveExplanation(
                contribs[i,-1],
                np.sum(contribs[i,:]),
                contribs[i,:-1],
                None,
                instance,
                IdentityLink(),
                Model(None, out_names),
                DenseData(np.zeros((1,data.shape[1])), list(feature_names))
            )
            exps.append(e)
        return exps

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
            phi = np.zeros(len(self.data.groups))
            phi_var = np.zeros(len(self.data.groups))
            return AdditiveExplanation(self.fnull, self.fx, phi, phi_var, instance, self.link, self.model, self.data)


        # if only one feature varies then it has all the effect
        elif self.M == 1:
            phi = np.zeros(len(self.data.groups))
            phi[self.varyingInds[0]] = self.link.f(self.fx) - self.link.f(self.fnull)
            phi_var = np.zeros(len(self.data.groups))
            return AdditiveExplanation(self.fnull, self.fx, phi, phi_var, instance, self.link, self.model, self.data)

        self.use_l1 = kwargs.get("use_l1", None)

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
        vphi,vphi_var = self.solve(self.nsamples/self.max_samples)
        phi = np.zeros(len(self.data.groups))
        phi[self.varyingInds] = vphi
        phi_var = np.zeros(len(self.data.groups))
        phi_var[self.varyingInds] = vphi_var

        # return the Shapley values along with variances of the estimates
        # note that if features were eliminated by l1 regression their
        # variance will be 0, even though they are not perfectaly known
        return AdditiveExplanation(self.link.f(self.fnull), self.link.f(self.fx), phi, phi_var, instance, self.link, self.model, self.data)

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

    def solve(self, fraction_evaluated):
        eyAdj = self.linkfv(self.ey) - self.link.f(self.fnull)

        s = np.sum(self.maskMatrix, 1)

        # do feature selection if we have not well enumerated the space
        nonzero_inds = np.arange(self.M)
        #print("fraction_evaluated", fraction_evaluated)
        if self.use_l1 == True or (fraction_evaluated < 0.2 and self.use_l1 != False):
            #print("using feature selection...")
            w_aug = np.hstack((self.kernelWeights * (self.M-s), self.kernelWeights*s))
            log.info("np.sum(w_aug) = {0}".format(np.sum(w_aug)))
            log.info("np.sum(self.kernelWeights) = {0}".format(np.sum(self.kernelWeights)))
            w_sqrt_aug = np.sqrt(w_aug)
            eyAdj_aug = np.hstack((eyAdj, eyAdj - (self.link.f(self.fx) - self.link.f(self.fnull))))
            eyAdj_aug *= w_sqrt_aug
            mask_aug = np.transpose(w_sqrt_aug*np.transpose(np.vstack((self.maskMatrix, self.maskMatrix-1))))
            var_norms = np.array([np.linalg.norm(mask_aug[:,i]) for i in range(mask_aug.shape[1])])


            model = LassoLarsIC(criterion='bic', normalize=True)#fit_intercept
            #model = Lasso(alpha=self.l1reg, fit_intercept=True)
            model.fit(mask_aug, eyAdj_aug)
            nonzero_inds = np.nonzero(model.coef_)[0]

        if len(nonzero_inds) == 0:
            return np.zeros(self.M),np.ones(self.M)

        eyAdj2 = eyAdj - self.maskMatrix[:,nonzero_inds[-1]]*(self.link.f(self.fx) - self.link.f(self.fnull))
        etmp = np.transpose(np.transpose(self.maskMatrix[:,nonzero_inds[:-1]]) - self.maskMatrix[:,nonzero_inds[-1]])
        log.debug("etmp[1:4,:] {0}".format(etmp[0:4,:]))

        # solve a weighted least squares equation to estimate phi
        tmp = np.transpose(np.transpose(etmp) * np.transpose(self.kernelWeights))
        tmp2 = np.linalg.inv(np.dot(np.transpose(tmp),etmp))
        w = np.dot(tmp2,np.dot(np.transpose(tmp),eyAdj2))
        #log.info("w = {0}".format(w))
        log.debug("np.sum(w) = {0}".format(np.sum(w)))
        log.debug("self.link(self.fx) - self.link(self.fnull) = {0}".format(self.link.f(self.fx) - self.link.f(self.fnull)))
        phi = np.zeros(self.M)
        phi[nonzero_inds[:-1]] = w
        phi[nonzero_inds[-1]] = (self.link.f(self.fx) - self.link.f(self.fnull)) - sum(w)
        log.info("phi = {0}".format(phi))

        # clean up any rounding errors
        for i in range(self.M):
            if np.abs(phi[i]) < 1e-10:
                phi[i] = 0

        # yHat = np.dot(self.maskMatrix, w)
        # phi_var = np.var(yHat - eyAdj) * np.diag(tmp2)
        # phi_var = np.hstack((phi_var, max(phi_var))) # since the last weight is inferred we use a pessimistic guess of its variance

        # a finite sample adjustment based on how much of the weight is left in the sample space
        # fractionWeightLeft = 1 - sum(self.kernelWeights)/sum(np.array([(self.M-1)/(s*(self.M-s)) for s in range(1, self.M)]))

        return phi,np.ones(len(phi))#phi_var*fractionWeightLeft

def test_blank():
    pass

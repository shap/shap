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
from sklearn.linear_model import LassoLarsIC, Lasso
import pandas as pd

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


def joint_plot(ind, X, shap_value_matrix, feature_names=None, other_ind=None, other_auto_ind=0, alpha=1, axis_color="#000000", show=True):

    # convert from a DataFrame if we got one
    if str(type(X)) == "<class 'pandas.core.frame.DataFrame'>":
        if feature_names is None:
            feature_names = X.columns
        X = X.as_matrix()
    if feature_names is None:
        feature_names = ["Feature %d"%i for i in range(X.shape[1])]

    x = X[:,ind]
    xname = feature_names[ind]

    if other_ind is None:
        other_ind = interactions(X, shap_value_matrix, ind)[other_auto_ind]

    y = X[:,other_ind]
    yname = feature_names[other_ind]

    joint_shap_values = shap_value_matrix[:,ind] + shap_value_matrix[:,other_ind]

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

    sc = pl.scatter(x, y, s=20, c=joint_shap_values, edgecolor='', alpha=alpha, cmap=red_blue)
    pl.xlabel(xname, color=axis_color)
    pl.ylabel(yname, color=axis_color)
    cb = pl.colorbar(sc, label="Joint SHAP value")
    cb.set_alpha(1)
    cb.draw_all()

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

    pl.plot(xv, shap_values, ".", markersize=5, color=color, alpha=alpha, markeredgewidth=0)

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

def approx_interactions(X, shap_values, index):
    """ Order other features by how much interaction they seem to have with the feature at the given index.

    This just bins the SHAP values for a feature along that feature's value. For true Shapley interaction
    index values for SHAP see the interaction_contrib option implemented in XGBoost.
    """
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
    inc = min(int(len(x)/10.0), 50)
    interactions = []
    for i in range(X.shape[1]):
        val_other = X[inds,i][srt]

        if i == index or np.sum(np.abs(val_other)) < 1e-8:
            v = 0
        else:
            v = np.sum(np.abs([np.corrcoef(shap_ref[i:i+inc],val_other[i:i+inc])[0,1] for i in range(0,len(x),inc)]))
        interactions.append(v)

    return np.argsort(-np.abs(interactions))

def interaction_plot(ind, X, shap_value_matrix, feature_names=None, interaction_index=None, color="#ff0052", axis_color="#333333", alpha=1, title=None, dot_size=12, show=True):

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

    if interaction_index is None:
        interaction_index = approx_interactions(X, shap_value_matrix, ind)[0]
    pl.scatter(xv, shap_values, s=dot_size, linewidth=0, c=X[:,interaction_index], cmap=red_blue, alpha=alpha)
    cb = pl.colorbar(label=feature_names[interaction_index])
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
import matplotlib.pyplot as pl
from scipy.stats import gaussian_kde
def summary_plot(shap_values, features, feature_names=None, max_display=10, color=None, axis_color="#333333", title=None, alpha=1, violin=True, distributed=True, show=True, max_num_bins=20, size=(10, 10), width=0.7):
    """
    Use cases:
        - scatter plot: TODO
        - scatter plot split: TODO
        - violin plot: color="#ff0052"
        - violin plot split: color="coolwarm", features != None
    """
    
    # NOTE: the sum of individual kdes may not be the full kde ...
    
    
    ind_order = np.argsort(np.sum(np.abs(shap_values), axis=0)[:-1])
    ind_order = ind_order[-min(max_display,len(ind_order)):]
    pl.gcf().set_size_inches(size)
    pl.axvline(x=0, color="#999999")
    
    # features can be a pandas dataframe or a matrix: handle both cases:
    # TODO: could be more error handling here e.g. provide feature matrix of wrong dimension, etc.
    if isinstance(features, pd.DataFrame):
        feature_matrix = features.as_matrix()
        # use dataframe columns if no names provided:
        if feature_names is None:
            feature_names = features.columns
    else:
        feature_matrix = features
        if feature_names is None:
            raise ValueError("if provided features as a matrix, you must provide feature_names")
    
    # set default colors:
    if color is None:
        color = "coolwarm" if distributed else "#ff0052"
    
    # check stuff:
    if distributed:
        if not color in pl.cm.datad:
            raise ValueError("'%s' is not a known colormap" % color)
        if feature_matrix is None:
            raise ValueError("features must be provided if doing a distributed plot")
    
    if violin:
        num_x_points = 200
        for pos, i in enumerate(ind_order):
            pl.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1,5))
        if distributed:
            bins = np.linspace(0, features.shape[0], max_num_bins + 1).round(0).astype('int') # the indices of the feature data corresponding to each bin
            x_points = np.linspace(-1, 1, num_x_points)
            cmap = pl.get_cmap(color)
            # loop through each feature and plot:
            for pos, ind in enumerate(ind_order):
                # decide how to handle: if #unique < max_num_bins then split by unique value, otherwise use bins/percentiles.
                # to keep simpler code, in the case of uniques, we just adjust the bins to align with the unique counts.
                feature = feature_matrix[:, ind]
                unique, counts = np.unique(feature, return_counts=True)
                if unique.shape[0] <= max_num_bins:
                    order = np.argsort(unique)
                    thesebins = np.cumsum(counts[order])
                    thesebins = np.insert(thesebins, 0, 0)
                else:
                    thesebins = bins
                nbins = thesebins.shape[0] - 1
                # order the feature data so we can apply percentiling
                order = np.argsort(feature)
                # x axis is located at y0 = pos, with pos being there for offset
                y0 = np.ones(num_x_points) * pos
                # calculate kdes:
                ys = np.zeros((nbins, num_x_points))
                for i in range(nbins):
                    # get shap values in this bin:
                    shaps = shap_values[order[bins[i]:bins[i+1]], ind]
                    # save kde of them: note that we add a tiny bit of gaussian noise to avoid singular matrix errors
                    ys[i, :] = gaussian_kde(shaps + np.random.normal(loc=0, scale=0.001, size=shaps.shape[0]))(x_points)
                # now plot 'em. We don't plot the individual strips, as this can leave whitespace between them.
                # instead, we plot the full kde, then remove outer strip and plot over it, etc., to ensure no
                # whitespace
                ys = np.cumsum(ys, axis=0)
                scale = ys.max() * 2 / width # 2 is here as we plot both sides of x axis
                for i in range(nbins - 1, -1, -1):
                    y = ys[i, :] / scale
                    c = cmap(i / (nbins - 1))
                    pl.fill_between(x_points, pos - y, pos + y, facecolor=c)
        else:
            parts = pl.violinplot(shap_values[:,ind_order], range(len(ind_order)), points=num_x_points, vert=False, widths=width,
                              showmeans=False, showextrema=False, showmedians=False)
            for pc in parts['bodies']:
                pc.set_facecolor(color)
                pc.set_edgecolor('none')
                pc.set_alpha(alpha)
    else:
        for pos,i in enumerate(ind_order):
            pl.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1,5))
            if distributed:
                # set y values as random values bounded between pos +-width                
                y = pos + width * (1 / (1 + np.exp(-np.random.normal(size=shap_values.shape[0]))) - 0.5)
                # get colors: the order of the item in the sorted list
                colors = np.argsort(np.argsort(feature_matrix[:, i]))
                cmap = pl.get_cmap(color)
                pl.scatter(shap_values[:, i], y, c=colors, cmap=cmap, alpha=alpha, linewidth=0)
            else:
                pl.scatter(shap_values[:,i], np.ones(shap_values.shape[0])*pos, color=color, alpha=alpha, linewidth=0)
    
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
    """ Visualize the given SHAP values with an additive force layout. """

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
            varying[i] = sum(sum(x[0,inds] == self.data.data[:,inds]) != len(inds))
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

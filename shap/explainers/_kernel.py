import copy
import gc
import itertools
import logging
import time
import warnings

import numpy as np
import pandas as pd
import scipy.sparse
import sklearn
from _kernel_lib import _exp_val
from packaging import version
from scipy.special import binom
from sklearn.linear_model import Lasso, LassoLarsIC, lars_path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from .._explanation import Explanation
from ..utils import safe_isinstance
from ..utils._exceptions import DimensionError
from ..utils._legacy import (
    DenseData,
    SparseData,
    convert_to_data,
    convert_to_instance,
    convert_to_instance_with_index,
    convert_to_link,
    convert_to_model,
    match_instance_to_data,
    match_model_to_data,
)
from ._explainer import Explainer

log = logging.getLogger("shap")


class KernelExplainer(Explainer):
    """Uses the Kernel SHAP method to explain the output of any function.

    Kernel SHAP is a method that uses a special weighted linear regression
    to compute the importance of each feature. The computed importance values
    are Shapley values from game theory and also coefficients from a local linear
    regression.

    Parameters
    ----------
    model : function or iml.Model
        User supplied function that takes a matrix of samples (# samples x # features) and
        computes the output of the model for those samples. The output can be a vector
        (# samples) or a matrix (# samples x # model outputs).

    data : numpy.array or pandas.DataFrame or shap.common.DenseData or any scipy.sparse matrix
        The background dataset to use for integrating out features. To determine the impact
        of a feature, that feature is set to "missing" and the change in the model output
        is observed. Since most models aren't designed to handle arbitrary missing data at test
        time, we simulate "missing" by replacing the feature with the values it takes in the
        background dataset. So if the background dataset is a simple sample of all zeros, then
        we would approximate a feature being missing by setting it to zero. For small problems,
        this background dataset can be the whole training set, but for larger problems consider
        using a single reference value or using the ``kmeans`` function to summarize the dataset.
        Note: for the sparse case, we accept any sparse matrix but convert to lil format for
        performance.

    feature_names : list
        The names of the features in the background dataset. If the background dataset is
        supplied as a pandas.DataFrame, then ``feature_names`` can be set to ``None`` (default),
        and the feature names will be taken as the column names of the dataframe.

    link : "identity" or "logit"
        A generalized linear model link to connect the feature importance values to the model
        output. Since the feature importance values, phi, sum up to the model output, it often makes
        sense to connect them to the output with a link function where link(output) = sum(phi).
        Default is "identity" (a no-op).
        If the model output is a probability, then "logit" can be used to transform the SHAP values
        into log-odds units.

    Examples
    --------
    See :ref:`Kernel Explainer Examples <kernel_explainer_examples>`.

    """

    def __init__(self, model, data, feature_names=None, link="identity", **kwargs):
        if feature_names is not None:
            self.data_feature_names = feature_names
        elif isinstance(data, pd.DataFrame):
            self.data_feature_names = list(data.columns)

        # convert incoming inputs to standardized iml objects
        self.link = convert_to_link(link)
        self.keep_index = kwargs.get("keep_index", False)
        self.keep_index_ordered = kwargs.get("keep_index_ordered", False)
        self.model = convert_to_model(model, keep_index=self.keep_index)
        self.data = convert_to_data(data, keep_index=self.keep_index)
        model_null = match_model_to_data(self.model, self.data)

        # enforce our current input type limitations
        if not isinstance(self.data, (DenseData, SparseData)):
            emsg = "Shap explainer only supports the DenseData and SparseData input currently."
            raise TypeError(emsg)
        if self.data.transposed:
            emsg = "Shap explainer does not support transposed DenseData or SparseData currently."
            raise DimensionError(emsg)

        # warn users about large background data sets
        if len(self.data.weights) > 100:
            log.warning(
                "Using "
                + str(len(self.data.weights))
                + " background data samples could cause "
                + "slower run times. Consider using shap.sample(data, K) or shap.kmeans(data, K) to "
                + "summarize the background as K samples."
            )

        # init our parameters
        self.N = self.data.data.shape[0]
        self.P = self.data.data.shape[1]
        self.linkfv = np.vectorize(self.link.f)
        self.nsamplesAdded = 0
        self.nsamplesRun = 0

        # find E_x[f(x)]
        if isinstance(model_null, (pd.DataFrame, pd.Series)):
            model_null = np.squeeze(model_null.values)
        if safe_isinstance(model_null, "tensorflow.python.framework.ops.EagerTensor"):
            model_null = model_null.numpy()
        elif safe_isinstance(model_null, "tensorflow.python.framework.ops.SymbolicTensor"):
            model_null = self._convert_symbolic_tensor(model_null)
        self.fnull = np.sum((model_null.T * self.data.weights).T, 0)
        self.expected_value = self.linkfv(self.fnull)

        # see if we have a vector output
        self.vector_out = True
        if len(self.fnull.shape) == 0:
            self.vector_out = False
            self.fnull = np.array([self.fnull])
            self.D = 1
            self.expected_value = float(self.expected_value)
        else:
            self.D = self.fnull.shape[0]

    @staticmethod
    def _convert_symbolic_tensor(symbolic_tensor) -> np.ndarray:
        import tensorflow as tf

        if tf.__version__ >= "2.0.0":
            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                tensor_as_np_array = sess.run(symbolic_tensor)
        else:
            # this is untested
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                tensor_as_np_array = sess.run(symbolic_tensor)
        return tensor_as_np_array

    def __call__(self, X, l1_reg="num_features(10)", silent=False):
        start_time = time.time()

        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
        else:
            feature_names = getattr(self, "data_feature_names", None)

        v = self.shap_values(X, l1_reg=l1_reg, silent=silent)
        if isinstance(v, list):
            v = np.stack(v, axis=-1)  # put outputs at the end

        # the explanation object expects an expected value for each row
        if hasattr(self.expected_value, "__len__"):
            ev_tiled = np.tile(self.expected_value, (v.shape[0], 1))
        else:
            ev_tiled = np.tile(self.expected_value, v.shape[0])

        return Explanation(
            v,
            base_values=ev_tiled,
            data=X.to_numpy() if isinstance(X, pd.DataFrame) else X,
            feature_names=feature_names,
            compute_time=time.time() - start_time,
        )

    def shap_values(self, X, **kwargs):
        """Estimate the SHAP values for a set of samples.

        Parameters
        ----------
        X : numpy.array or pandas.DataFrame or any scipy.sparse matrix
            A matrix of samples (# samples x # features) on which to explain the model's output.

        nsamples : "auto" or int
            Number of times to re-evaluate the model when explaining each prediction. More samples
            lead to lower variance estimates of the SHAP values. The "auto" setting uses
            `nsamples = 2 * X.shape[1] + 2048`.

        l1_reg : "num_features(int)", "aic", "bic", or float
            The l1 regularization to use for feature selection. The estimation
            procedure is based on a debiased lasso.

            * "num_features(int)" selects a fixed number of top features.
            * "aic" and "bic" options use the AIC and BIC rules for regularization.
            * Passing a float directly sets the "alpha" parameter of the
              ``sklearn.linear_model.Lasso`` model used for feature selection.
            * "auto" (deprecated): uses "aic" when less than
              20% of the possible sample space is enumerated, otherwise it uses
              no regularization.

            .. versionchanged:: 0.47.0
                The default value changed from ``"auto"`` to ``"num_features(10)"``.

        silent: bool
            If True, hide tqdm progress bar. Default False.

        gc_collect : bool
           Run garbage collection after each explanation round. Sometime needed for memory intensive explanations (default False).

        Returns
        -------
        np.array or list
            Estimated SHAP values, usually of shape ``(# samples x # features)``.

            Each row sums to the difference between the model output for that
            sample and the expected value of the model output (which is stored as the ``expected_value``
            attribute of the explainer).

            The type and shape of the return value depends on the number of model inputs and outputs:

            * one input, one output: array of shape ``(#num_samples, *X.shape[1:])``.
            * one input, multiple outputs: array of shape ``(#num_samples, *X.shape[1:], #num_outputs)``
            * multiple inputs: list of arrays of corresponding shape above.

            .. versionchanged:: 0.45.0
                Return type for models with multiple outputs and one input changed from list to np.ndarray.

        """
        # convert dataframes
        if isinstance(X, pd.Series):
            X = X.values
        elif isinstance(X, pd.DataFrame):
            if self.keep_index:
                index_value = X.index.values
                index_name = X.index.name
                column_name = list(X.columns)
            X = X.values

        x_type = str(type(X))
        arr_type = "'numpy.ndarray'>"
        # if sparse, convert to lil for performance
        if scipy.sparse.issparse(X) and not scipy.sparse.isspmatrix_lil(X):
            X = X.tolil()
        assert x_type.endswith(arr_type) or scipy.sparse.isspmatrix_lil(X), "Unknown instance type: " + x_type

        # single instance
        if len(X.shape) == 1:
            data = X.reshape((1, X.shape[0]))
            if self.keep_index:
                data = convert_to_instance_with_index(data, column_name, index_name, index_value)
            explanation = self.explain(data, **kwargs)

            # vector-output
            s = explanation.shape
            out = np.zeros(s)
            out[:] = explanation
            return out

        # explain the whole dataset
        elif len(X.shape) == 2:
            explanations = []
            for i in tqdm(range(X.shape[0]), disable=kwargs.get("silent", False)):
                data = X[i : i + 1, :]
                if self.keep_index:
                    data = convert_to_instance_with_index(data, column_name, index_value[i : i + 1], index_name)
                explanations.append(self.explain(data, **kwargs))
                if kwargs.get("gc_collect", False):
                    gc.collect()

            # vector-output
            s = explanations[0].shape
            if len(s) == 2:
                outs = [np.zeros((X.shape[0], s[0])) for j in range(s[1])]
                for i in range(X.shape[0]):
                    for j in range(s[1]):
                        outs[j][i] = explanations[i][:, j]
                outs = np.stack(outs, axis=-1)
                return outs

            # single-output
            else:
                out = np.zeros((X.shape[0], s[0]))
                for i in range(X.shape[0]):
                    out[i] = explanations[i]
                return out

        else:
            emsg = "Instance must have 1 or 2 dimensions!"
            raise DimensionError(emsg)

    def explain(self, incoming_instance, **kwargs):
        # convert incoming input to a standardized iml object
        instance = convert_to_instance(incoming_instance)
        match_instance_to_data(instance, self.data)

        # find the feature groups we will test. If a feature does not change from its
        # current value then we know it doesn't impact the model
        self.varyingInds = self.varying_groups(instance.x)
        if self.data.groups is None:
            self.varyingFeatureGroups = np.array([i for i in self.varyingInds])
            self.M = self.varyingFeatureGroups.shape[0]
        else:
            self.varyingFeatureGroups = [self.data.groups[i] for i in self.varyingInds]
            self.M = len(self.varyingFeatureGroups)
            groups = self.data.groups
            # convert to numpy array as it is much faster if not jagged array (all groups of same length)
            if self.varyingFeatureGroups and all(len(groups[i]) == len(groups[0]) for i in self.varyingInds):
                self.varyingFeatureGroups = np.array(self.varyingFeatureGroups)
                # further performance optimization in case each group has a single value
                if self.varyingFeatureGroups.shape[1] == 1:
                    self.varyingFeatureGroups = self.varyingFeatureGroups.flatten()

        # find f(x)
        if self.keep_index:
            model_out = self.model.f(instance.convert_to_df())
        else:
            model_out = self.model.f(instance.x)
        if isinstance(model_out, (pd.DataFrame, pd.Series)):
            model_out = model_out.values
        elif safe_isinstance(model_out, "tensorflow.python.framework.ops.SymbolicTensor"):
            model_out = self._convert_symbolic_tensor(model_out)
        self.fx = model_out[0]

        if not self.vector_out:
            self.fx = np.array([self.fx])

        # if no features vary then no feature has an effect
        if self.M == 0:
            phi = np.zeros((self.data.groups_size, self.D))
            phi_var = np.zeros((self.data.groups_size, self.D))

        # if only one feature varies then it has all the effect
        elif self.M == 1:
            phi = np.zeros((self.data.groups_size, self.D))
            phi_var = np.zeros((self.data.groups_size, self.D))
            diff = self.link.f(self.fx) - self.link.f(self.fnull)
            for d in range(self.D):
                phi[self.varyingInds[0], d] = diff[d]

        # if more than one feature varies then we have to do real work
        else:
            self.l1_reg = kwargs.get("l1_reg", "num_features(10)")

            # pick a reasonable number of samples if the user didn't specify how many they wanted
            self.nsamples = kwargs.get("nsamples", "auto")
            if self.nsamples == "auto":
                self.nsamples = 2 * self.M + 2**11

            # if we have enough samples to enumerate all subsets then ignore the unneeded samples
            self.max_samples = 2**30
            if self.M <= 30:
                self.max_samples = 2**self.M - 2
                if self.nsamples > self.max_samples:
                    self.nsamples = self.max_samples

            # reserve space for some of our computations
            self.allocate()

            # weight the different subset sizes
            num_subset_sizes = int(np.ceil((self.M - 1) / 2.0))
            num_paired_subset_sizes = int(np.floor((self.M - 1) / 2.0))
            weight_vector = np.array([(self.M - 1.0) / (i * (self.M - i)) for i in range(1, num_subset_sizes + 1)])
            weight_vector[:num_paired_subset_sizes] *= 2
            weight_vector /= np.sum(weight_vector)
            log.debug(f"{weight_vector = }")
            log.debug(f"{num_subset_sizes = }")
            log.debug(f"{num_paired_subset_sizes = }")
            log.debug(f"{self.M = }")

            # fill out all the subset sizes we can completely enumerate
            # given nsamples*remaining_weight_vector[subset_size]
            num_full_subsets = 0
            num_samples_left = self.nsamples
            group_inds = np.arange(self.M, dtype="int64")
            mask = np.zeros(self.M)
            remaining_weight_vector = copy.copy(weight_vector)
            for subset_size in range(1, num_subset_sizes + 1):
                # determine how many subsets (and their complements) are of the current size
                nsubsets = binom(self.M, subset_size)
                if subset_size <= num_paired_subset_sizes:
                    nsubsets *= 2
                log.debug(f"{subset_size = }")
                log.debug(f"{nsubsets = }")
                log.debug(
                    "self.nsamples*weight_vector[subset_size-1] = "
                    f"{num_samples_left * remaining_weight_vector[subset_size - 1]}"
                )
                log.debug(
                    "self.nsamples*weight_vector[subset_size-1]/nsubsets = "
                    f"{num_samples_left * remaining_weight_vector[subset_size - 1] / nsubsets}"
                )

                # see if we have enough samples to enumerate all subsets of this size
                if num_samples_left * remaining_weight_vector[subset_size - 1] / nsubsets >= 1.0 - 1e-8:
                    num_full_subsets += 1
                    num_samples_left -= nsubsets

                    # rescale what's left of the remaining weight vector to sum to 1
                    if remaining_weight_vector[subset_size - 1] < 1.0:
                        remaining_weight_vector /= 1 - remaining_weight_vector[subset_size - 1]

                    # add all the samples of the current subset size
                    w = weight_vector[subset_size - 1] / binom(self.M, subset_size)
                    if subset_size <= num_paired_subset_sizes:
                        w /= 2.0
                    for inds in itertools.combinations(group_inds, subset_size):
                        mask[:] = 0.0
                        mask[np.array(inds, dtype="int64")] = 1.0
                        self.addsample(instance.x, mask, w)
                        if subset_size <= num_paired_subset_sizes:
                            mask[:] = np.abs(mask - 1)
                            self.addsample(instance.x, mask, w)
                else:
                    break
            log.info(f"{num_full_subsets = }")

            # add random samples from what is left of the subset space
            nfixed_samples = self.nsamplesAdded
            samples_left = self.nsamples - self.nsamplesAdded
            log.debug(f"{samples_left = }")
            if num_full_subsets != num_subset_sizes:
                remaining_weight_vector = copy.copy(weight_vector)
                remaining_weight_vector[:num_paired_subset_sizes] /= 2  # because we draw two samples each below
                remaining_weight_vector = remaining_weight_vector[num_full_subsets:]
                remaining_weight_vector /= np.sum(remaining_weight_vector)
                log.info(f"{remaining_weight_vector = }")
                log.info(f"{num_paired_subset_sizes = }")
                ind_set = np.random.choice(len(remaining_weight_vector), 4 * samples_left, p=remaining_weight_vector)
                ind_set_pos = 0
                used_masks = {}
                while samples_left > 0 and ind_set_pos < len(ind_set):
                    mask.fill(0.0)
                    ind = ind_set[ind_set_pos]  # we call np.random.choice once to save time and then just read it here
                    ind_set_pos += 1
                    subset_size = ind + num_full_subsets + 1
                    mask[np.random.permutation(self.M)[:subset_size]] = 1.0

                    # only add the sample if we have not seen it before, otherwise just
                    # increment a previous sample's weight
                    mask_tuple = tuple(mask)
                    new_sample = False
                    if mask_tuple not in used_masks:
                        new_sample = True
                        used_masks[mask_tuple] = self.nsamplesAdded
                        samples_left -= 1
                        self.addsample(instance.x, mask, 1.0)
                    else:
                        self.kernelWeights[used_masks[mask_tuple]] += 1.0

                    # add the compliment sample
                    if samples_left > 0 and subset_size <= num_paired_subset_sizes:
                        mask[:] = np.abs(mask - 1)

                        # only add the sample if we have not seen it before, otherwise just
                        # increment a previous sample's weight
                        if new_sample:
                            samples_left -= 1
                            self.addsample(instance.x, mask, 1.0)
                        else:
                            # we know the compliment sample is the next one after the original sample, so + 1
                            self.kernelWeights[used_masks[mask_tuple] + 1] += 1.0

                # normalize the kernel weights for the random samples to equal the weight left after
                # the fixed enumerated samples have been already counted
                weight_left = np.sum(weight_vector[num_full_subsets:])
                log.info(f"{weight_left = }")
                self.kernelWeights[nfixed_samples:] *= weight_left / self.kernelWeights[nfixed_samples:].sum()

            # execute the model on the synthetic samples we have created
            self.run()

            # solve then expand the feature importance (Shapley value) vector to contain the non-varying features
            phi = np.zeros((self.data.groups_size, self.D))
            phi_var = np.zeros((self.data.groups_size, self.D))
            for d in range(self.D):
                vphi, vphi_var = self.solve(self.nsamples / self.max_samples, d)
                phi[self.varyingInds, d] = vphi
                phi_var[self.varyingInds, d] = vphi_var

        if not self.vector_out:
            phi = np.squeeze(phi, axis=1)
            phi_var = np.squeeze(phi_var, axis=1)

        return phi

    @staticmethod
    def not_equal(i, j):
        number_types = (int, float, np.number)
        if isinstance(i, number_types) and isinstance(j, number_types):
            return 0 if np.allclose(i, j, equal_nan=True) else 1
        elif hasattr(i, "dtype") and hasattr(j, "dtype"):
            if np.issubdtype(i.dtype, np.number) and np.issubdtype(j.dtype, np.number):
                return 0 if np.allclose(i, j, equal_nan=True) else 1
            if np.issubdtype(i.dtype, np.bool_) and np.issubdtype(j.dtype, np.bool_):
                return 0 if np.allclose(i, j, equal_nan=True) else 1
            return 0 if all(i == j) else 1
        else:
            return 0 if i == j else 1

    def varying_groups(self, x):
        if not scipy.sparse.issparse(x):
            varying = np.zeros(self.data.groups_size)
            for i in range(self.data.groups_size):
                inds = self.data.groups[i]
                x_group = x[0, inds]
                if scipy.sparse.issparse(x_group):
                    if all(j not in x.nonzero()[1] for j in inds):
                        varying[i] = False
                        continue
                    x_group = x_group.todense()
                varying[i] = self.not_equal(x_group, self.data.data[:, inds])
            varying_indices = np.nonzero(varying)[0]
            return varying_indices
        else:
            varying_indices = []
            # go over all nonzero columns in background and evaluation data
            # if both background and evaluation are zero, the column does not vary
            varying_indices = np.unique(np.union1d(self.data.data.nonzero()[1], x.nonzero()[1]))
            remove_unvarying_indices = []
            for i in range(len(varying_indices)):
                varying_index = varying_indices[i]
                # now verify the nonzero values do vary
                data_rows = self.data.data[:, [varying_index]]
                nonzero_rows = data_rows.nonzero()[0]

                if nonzero_rows.size > 0:
                    background_data_rows = data_rows[nonzero_rows]
                    if scipy.sparse.issparse(background_data_rows):
                        background_data_rows = background_data_rows.toarray()
                    num_mismatches = np.sum(np.abs(background_data_rows - x[0, varying_index]) > 1e-7)
                    # Note: If feature column non-zero but some background zero, can't remove index
                    if num_mismatches == 0 and not (
                        np.abs(x[0, [varying_index]][0, 0]) > 1e-7 and len(nonzero_rows) < data_rows.shape[0]
                    ):
                        remove_unvarying_indices.append(i)
            mask = np.ones(len(varying_indices), dtype=bool)
            mask[remove_unvarying_indices] = False
            varying_indices = varying_indices[mask]
            return varying_indices

    def allocate(self):
        if scipy.sparse.issparse(self.data.data):
            # We tile the sparse matrix in csr format but convert it to lil
            # for performance when adding samples
            shape = self.data.data.shape
            nnz = self.data.data.nnz
            data_rows, data_cols = shape
            rows = data_rows * self.nsamples
            shape = rows, data_cols
            if nnz == 0:
                self.synth_data = scipy.sparse.csr_matrix(shape, dtype=self.data.data.dtype).tolil()
            else:
                data = self.data.data.data
                indices = self.data.data.indices
                indptr = self.data.data.indptr
                last_indptr_idx = indptr[len(indptr) - 1]
                indptr_wo_last = indptr[:-1]
                new_indptrs = []
                for i in range(self.nsamples - 1):
                    new_indptrs.append(indptr_wo_last + (i * last_indptr_idx))
                new_indptrs.append(indptr + ((self.nsamples - 1) * last_indptr_idx))
                new_indptr = np.concatenate(new_indptrs)
                new_data = np.tile(data, self.nsamples)
                new_indices = np.tile(indices, self.nsamples)
                self.synth_data = scipy.sparse.csr_matrix((new_data, new_indices, new_indptr), shape=shape).tolil()
        else:
            self.synth_data = np.tile(self.data.data, (self.nsamples, 1))

        self.maskMatrix = np.zeros((self.nsamples, self.M))
        self.kernelWeights = np.zeros(self.nsamples)
        self.y = np.zeros((self.nsamples * self.N, self.D))
        self.ey = np.zeros((self.nsamples, self.D))
        self.lastMask = np.zeros(self.nsamples)
        self.nsamplesAdded = 0
        self.nsamplesRun = 0
        if self.keep_index:
            self.synth_data_index = np.tile(self.data.index_value, self.nsamples)

    def addsample(self, x, m, w):
        offset = self.nsamplesAdded * self.N
        if isinstance(self.varyingFeatureGroups, (list,)):
            for j in range(self.M):
                for k in self.varyingFeatureGroups[j]:
                    if m[j] == 1.0:
                        self.synth_data[offset : offset + self.N, k] = x[0, k]
        else:
            # for non-jagged numpy array we can significantly boost performance
            mask = m == 1.0
            groups = self.varyingFeatureGroups[mask]
            if len(groups.shape) == 2:
                for group in groups:
                    self.synth_data[offset : offset + self.N, group] = x[0, group]
            else:
                # further performance optimization in case each group has a single feature
                evaluation_data = x[0, groups]
                # In edge case where background is all dense but evaluation data
                # is all sparse, make evaluation data dense
                if scipy.sparse.issparse(x) and not scipy.sparse.issparse(self.synth_data):
                    evaluation_data = evaluation_data.toarray()
                self.synth_data[offset : offset + self.N, groups] = evaluation_data
        self.maskMatrix[self.nsamplesAdded, :] = m
        self.kernelWeights[self.nsamplesAdded] = w
        self.nsamplesAdded += 1

    def run(self):
        num_to_run = self.nsamplesAdded * self.N - self.nsamplesRun * self.N
        data = self.synth_data[self.nsamplesRun * self.N : self.nsamplesAdded * self.N, :]
        if self.keep_index:
            index = self.synth_data_index[self.nsamplesRun * self.N : self.nsamplesAdded * self.N]
            index = pd.DataFrame(index, columns=[self.data.index_name])
            data = pd.DataFrame(data, columns=self.data.group_names)
            data = pd.concat([index, data], axis=1).set_index(self.data.index_name)
            if self.keep_index_ordered:
                data = data.sort_index()
        modelOut = self.model.f(data)
        if isinstance(modelOut, (pd.DataFrame, pd.Series)):
            modelOut = modelOut.values
        elif safe_isinstance(modelOut, "tensorflow.python.framework.ops.SymbolicTensor"):
            modelOut = self._convert_symbolic_tensor(modelOut)

        self.y[self.nsamplesRun * self.N : self.nsamplesAdded * self.N, :] = np.reshape(modelOut, (num_to_run, self.D))

        # find the expected value of each output
        self.ey, self.nsamplesRun = _exp_val(
            self.nsamplesRun, self.nsamplesAdded, self.D, self.N, self.data.weights, self.y, self.ey
        )

    def solve(self, fraction_evaluated, dim):
        eyAdj = self.linkfv(self.ey[:, dim]) - self.link.f(self.fnull[dim])
        s = np.sum(self.maskMatrix, 1)

        # do feature selection if we have not well enumerated the space
        nonzero_inds = np.arange(self.M)
        log.debug(f"{fraction_evaluated = }")
        if self.l1_reg == "auto":
            warnings.warn("l1_reg='auto' is deprecated and will be removed in a future version.", DeprecationWarning)
        if (self.l1_reg not in ["auto", False, 0]) or (fraction_evaluated < 0.2 and self.l1_reg == "auto"):
            w_aug = np.hstack((self.kernelWeights * (self.M - s), self.kernelWeights * s))
            log.info(f"{np.sum(w_aug) = }")
            log.info(f"{np.sum(self.kernelWeights) = }")
            w_sqrt_aug = np.sqrt(w_aug)
            eyAdj_aug = np.hstack((eyAdj, eyAdj - (self.link.f(self.fx[dim]) - self.link.f(self.fnull[dim]))))
            eyAdj_aug *= w_sqrt_aug
            mask_aug = np.transpose(w_sqrt_aug * np.transpose(np.vstack((self.maskMatrix, self.maskMatrix - 1))))
            # var_norms = np.array([np.linalg.norm(mask_aug[:, i]) for i in range(mask_aug.shape[1])])

            # select a fixed number of top features
            if isinstance(self.l1_reg, str) and self.l1_reg.startswith("num_features("):
                r = int(self.l1_reg[len("num_features(") : -1])
                nonzero_inds = lars_path(mask_aug, eyAdj_aug, max_iter=r)[1]

            # use an adaptive regularization method
            elif self.l1_reg in ("auto", "bic", "aic"):
                c = "aic" if self.l1_reg == "auto" else self.l1_reg

                # "Normalize" parameter of LassoLarsIC was deprecated in sklearn version 1.2
                if version.parse(sklearn.__version__) < version.parse("1.2.0"):
                    kwg = dict(normalize=False)
                else:
                    kwg = {}
                model = make_pipeline(StandardScaler(with_mean=False), LassoLarsIC(criterion=c, **kwg))
                nonzero_inds = np.nonzero(model.fit(mask_aug, eyAdj_aug)[1].coef_)[0]

            # use a fixed regularization coefficient
            else:
                nonzero_inds = np.nonzero(Lasso(alpha=self.l1_reg).fit(mask_aug, eyAdj_aug).coef_)[0]

        if len(nonzero_inds) == 0:
            return np.zeros(self.M), np.ones(self.M)

        # eliminate one variable with the constraint that all features sum to the output
        eyAdj2 = eyAdj - self.maskMatrix[:, nonzero_inds[-1]] * (
            self.link.f(self.fx[dim]) - self.link.f(self.fnull[dim])
        )
        etmp = np.transpose(np.transpose(self.maskMatrix[:, nonzero_inds[:-1]]) - self.maskMatrix[:, nonzero_inds[-1]])
        log.debug(f"{etmp[:4, :] = }")

        # solve a weighted least squares equation to estimate phi
        # least squares:
        #     phi = min_w ||W^(1/2) (y - X w)||^2
        # the corresponding normal equation:
        #     (X' W X) phi = X' W y
        # with
        #     X = etmp
        #     W = np.diag(self.kernelWeights)
        #     y = eyAdj2
        #
        # We could just rely on sciki-learn
        #     from sklearn.linear_model import LinearRegression
        #     lm = LinearRegression(fit_intercept=False).fit(etmp, eyAdj2, sample_weight=self.kernelWeights)
        # Under the hood, as of scikit-learn version 1.3, LinearRegression still uses np.linalg.lstsq and
        # there are more performant options. See https://github.com/scikit-learn/scikit-learn/issues/22855.
        y = np.asarray(eyAdj2)
        X = etmp
        WX = self.kernelWeights[:, None] * X
        try:
            w = np.linalg.solve(X.T @ WX, WX.T @ y)
        except np.linalg.LinAlgError:
            warnings.warn(
                "Linear regression equation is singular, a least squares solutions is used instead.\n"
                "To avoid this situation and get a regular matrix do one of the following:\n"
                "1) turn up the number of samples,\n"
                "2) turn up the L1 regularization with num_features(N) where N is less than the number of samples,\n"
                "3) group features together to reduce the number of inputs that need to be explained."
            )
            # XWX = np.linalg.pinv(X.T @ WX)
            # w = np.dot(XWX, np.dot(np.transpose(WX), y))
            sqrt_W = np.sqrt(self.kernelWeights)
            w = np.linalg.lstsq(sqrt_W[:, None] * X, sqrt_W * y, rcond=None)[0]
        log.debug(f"{np.sum(w) = }")
        log.debug(
            f"self.link(self.fx) - self.link(self.fnull) = {self.link.f(self.fx[dim]) - self.link.f(self.fnull[dim])}"
        )
        log.debug(f"self.fx = {self.fx[dim]}")
        log.debug(f"self.link(self.fx) = {self.link.f(self.fx[dim])}")
        log.debug(f"self.fnull = {self.fnull[dim]}")
        log.debug(f"self.link(self.fnull) = {self.link.f(self.fnull[dim])}")
        phi = np.zeros(self.M)
        phi[nonzero_inds[:-1]] = w
        phi[nonzero_inds[-1]] = (self.link.f(self.fx[dim]) - self.link.f(self.fnull[dim])) - sum(w)
        log.info(f"{phi = }")

        # clean up any rounding errors
        for i in range(self.M):
            if np.abs(phi[i]) < 1e-10:
                phi[i] = 0

        return phi, np.ones(len(phi))

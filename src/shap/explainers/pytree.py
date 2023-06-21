"""
This module is a pure python implementation of Tree SHAP.
It is primarily for illustration since it is slower than the 'tree'
module which uses a compiled C++ implmentation.
"""
import numpy as np

#import numba
from ..utils._exceptions import ExplainerError

# class TreeExplainer(Explainer):
#     def __init__(self, model, **kwargs):
#         self.model_type = "internal"

#         if str(type(model)).endswith("sklearn.ensemble.forest.RandomForestRegressor'>"):
#             self.trees = [Tree(e.tree_) for e in model.estimators_]
#         elif str(type(model)).endswith("sklearn.ensemble.forest.RandomForestClassifier'>"):
#             self.trees = [Tree(e.tree_, normalize=True) for e in model.estimators_]
#         elif str(type(model)).endswith("xgboost.core.Booster'>"):
#             self.model_type = "xgboost"
#             self.trees = model
#         elif str(type(model)).endswith("lightgbm.basic.Booster'>"):
#             self.model_type = "lightgbm"
#             self.trees = model
#         else:
#             raise Exception("Model type not supported by TreeExplainer: " + str(type(model)))

#     def shap_values(self, X, tree_limit=-1, **kwargs):

#         # shortcut using the C++ version of Tree SHAP in XGBoost and LightGBM
#         # these are about 10x faster than the numba jit'd implementation below...
#         if self.model_type == "xgboost":
#             if not str(type(X)).endswith("xgboost.core.DMatrix'>"):
#                 X = xgboost.DMatrix(X)
#             if tree_limit==-1:
#                 tree_limit=0
#             return self.trees.predict(X, ntree_limit=tree_limit, pred_contribs=True)
#         elif self.model_type == "lightgbm":
#             return self.trees.predict(X, num_iteration=tree_limit, pred_contrib=True)

#         # convert dataframes
#         if str(type(X)).endswith("pandas.core.series.Series'>"):
#             X = X.values
#         elif str(type(X)).endswith("pandas.core.frame.DataFrame'>"):
#             X = X.values

#         assert str(type(X)).endswith("'numpy.ndarray'>"), "Unknown instance type: " + str(type(X))
#         assert len(X.shape) == 1 or len(X.shape) == 2, "Instance must have 1 or 2 dimensions!"

#         n_outputs = self.trees[0].values.shape[1]

#         # single instance
#         if len(X.shape) == 1:

#             phi = np.zeros((X.shape[0] + 1, n_outputs))
#             x_missing = np.zeros(X.shape[0], dtype=bool)
#             for t in self.trees:
#                 self.tree_shap(t, X, x_missing, phi)
#             phi /= len(self.trees)

#             if n_outputs == 1:
#                 return phi[:, 0]
#             else:
#                 return [phi[:, i] for i in range(n_outputs)]

#         elif len(X.shape) == 2:
#             phi = np.zeros((X.shape[0], X.shape[1] + 1, n_outputs))
#             x_missing = np.zeros(X.shape[1], dtype=bool)
#             for i in range(X.shape[0]):
#                 for t in self.trees:
#                     self.tree_shap(t, X[i,:], x_missing, phi[i,:,:])
#             phi /= len(self.trees)

#             if n_outputs == 1:
#                 return phi[:, :, 0]
#             else:
#                 return [phi[:, :, i] for i in range(n_outputs)]

#     def shap_interaction_values(self, X, tree_limit=-1, **kwargs):

#         # shortcut using the C++ version of Tree SHAP in XGBoost and LightGBM
#         if self.model_type == "xgboost":
#             if not str(type(X)).endswith("xgboost.core.DMatrix'>"):
#                 X = xgboost.DMatrix(X)
#             if tree_limit==-1:
#                 tree_limit=0
#             return self.trees.predict(X, ntree_limit=tree_limit, pred_interactions=True)
#         else:
#             raise Exception("Interaction values not yet supported for model type: " + str(type(X)))

#     def tree_shap(self, tree, x, x_missing, phi, condition=0, condition_feature=0):

#         # start the recursive algorithm
#         shap._cext.tree_shap(
#             tree.max_depth, tree.children_left, tree.children_right, tree.children_default, tree.features,
#             tree.thresholds, tree.values, tree.node_sample_weight,
#             x, x_missing, phi, condition, condition_feature
#         )


# class Tree:
#     def __init__(self, children_left, children_right, children_default, feature, threshold, value, node_sample_weight):
#         self.children_left = children_left.astype(np.int32)
#         self.children_right = children_right.astype(np.int32)
#         self.children_default = children_default.astype(np.int32)
#         self.features = feature.astype(np.int32)
#         self.thresholds = threshold
#         self.values = value
#         self.node_sample_weight = node_sample_weight

#         # we compute the expectations to make sure they follow the SHAP logic
#         self.max_depth = shap._cext.compute_expectations(
#             self.children_left, self.children_right, self.node_sample_weight,
#             self.values
#         )

#     def __init__(self, tree, normalize=False):
#         if str(type(tree)).endswith("'sklearn.tree._tree.Tree'>"):
#             self.children_left = tree.children_left.astype(np.int32)
#             self.children_right = tree.children_right.astype(np.int32)
#             self.children_default = self.children_left # missing values not supported in sklearn
#             self.features = tree.feature.astype(np.int32)
#             self.thresholds = tree.threshold.astype(np.float64)
#             if normalize:
#                 self.values = (tree.value[:,0,:].T / tree.value[:,0,:].sum(1)).T
#             else:
#                 self.values = tree.value[:,0,:]


#             self.node_sample_weight = tree.weighted_n_node_samples.astype(np.float64)

#             # we compute the expectations to make sure they follow the SHAP logic
#             self.max_depth = shap._cext.compute_expectations(
#                 self.children_left, self.children_right, self.node_sample_weight,
#                 self.values
#             )


class TreeExplainer:
    """ A pure Python (slow) implementation of Tree SHAP.
    """

    def __init__(self, model, **kwargs):
        self.model_type = "internal"

        if str(type(model)).endswith("sklearn.ensemble.forest.RandomForestRegressor'>"):
            self.trees = [Tree(e.tree_) for e in model.estimators_]
        elif str(type(model)).endswith("sklearn.ensemble.forest.RandomForestClassifier'>"):
            self.trees = [Tree(e.tree_, normalize=True) for e in model.estimators_]
        elif str(type(model)).endswith("xgboost.core.Booster'>"):
            self.model_type = "xgboost"
            self.trees = model
        elif str(type(model)).endswith("lightgbm.basic.Booster'>"):
            self.model_type = "lightgbm"
            self.trees = model
        else:
            raise ExplainerError("Model type not supported by TreeExplainer: " + str(type(model)))

        if self.model_type == "internal":
            # Preallocate space for the unique path data
            maxd = np.max([t.max_depth for t in self.trees]) + 2
            s = (maxd * (maxd + 1)) // 2
            self.feature_indexes = np.zeros(s, dtype=np.int32)
            self.zero_fractions = np.zeros(s, dtype=np.float64)
            self.one_fractions = np.zeros(s, dtype=np.float64)
            self.pweights = np.zeros(s, dtype=np.float64)

    def shap_values(self, X, tree_limit=-1, **kwargs):

        # shortcut using the C++ version of Tree SHAP in XGBoost and LightGBM
        # these are about 10x faster than the numba jit'd implementation below...
        if self.model_type == "xgboost":
            import xgboost
            if not str(type(X)).endswith("xgboost.core.DMatrix'>"):
                X = xgboost.DMatrix(X)
            if tree_limit==-1:
                tree_limit=0
            return self.trees.predict(X, ntree_limit=tree_limit, pred_contribs=True)
        elif self.model_type == "lightgbm":
            return self.trees.predict(X, num_iteration=tree_limit, pred_contrib=True)

        # convert dataframes
        if str(type(X)).endswith("pandas.core.series.Series'>"):
            X = X.values
        elif str(type(X)).endswith("pandas.core.frame.DataFrame'>"):
            X = X.values

        assert str(type(X)).endswith("'numpy.ndarray'>"), "Unknown instance type: " + str(type(X))
        assert len(X.shape) == 1 or len(X.shape) == 2, "Instance must have 1 or 2 dimensions!"

        n_outputs = self.trees[0].values.shape[1]

        # single instance
        if len(X.shape) == 1:

            phi = np.zeros(X.shape[0] + 1, n_outputs)
            x_missing = np.zeros(X.shape[0], dtype=bool)
            for t in self.trees:
                self.tree_shap(t, X, x_missing, phi)
            phi /= len(self.trees)

            if n_outputs == 1:
                return phi[:, 0]
            else:
                return [phi[:, i] for i in range(n_outputs)]

        elif len(X.shape) == 2:
            phi = np.zeros((X.shape[0], X.shape[1] + 1, n_outputs))
            x_missing = np.zeros(X.shape[1], dtype=bool)
            for i in range(X.shape[0]):
                for t in self.trees:
                    self.tree_shap(t, X[i,:], x_missing, phi[i,:,:])
            phi /= len(self.trees)

            if n_outputs == 1:
                return phi[:, :, 0]
            else:
                return [phi[:, :, i] for i in range(n_outputs)]

    def shap_interaction_values(self, X, tree_limit=-1, **kwargs):

        # shortcut using the C++ version of Tree SHAP in XGBoost and LightGBM
        if self.model_type == "xgboost":
            import xgboost
            if not str(type(X)).endswith("xgboost.core.DMatrix'>"):
                X = xgboost.DMatrix(X)
            if tree_limit==-1:
                tree_limit=0
            return self.trees.predict(X, ntree_limit=tree_limit, pred_interactions=True)
        else:
            raise NotImplementedError("Interaction values not yet supported for model type: " + str(type(X)))

    def tree_shap(self, tree, x, x_missing, phi, condition=0, condition_feature=0):

        # update the bias term, which is the last index in phi
        # (note the paper has this as phi_0 instead of phi_M)
        if condition == 0:
            phi[-1,:] += tree.values[0,:]

        # start the recursive algorithm
        tree_shap_recursive(
            tree.children_left, tree.children_right, tree.children_default, tree.features,
            tree.thresholds, tree.values, tree.node_sample_weight,
            x, x_missing, phi, 0, 0, self.feature_indexes, self.zero_fractions, self.one_fractions, self.pweights,
            1, 1, -1, condition, condition_feature, 1
        )


# extend our decision path with a fraction of one and zero extensions
#@numba.jit(nopython=True, nogil=True)
def extend_path(feature_indexes, zero_fractions, one_fractions, pweights,
                unique_depth, zero_fraction, one_fraction, feature_index):
    feature_indexes[unique_depth] = feature_index
    zero_fractions[unique_depth] = zero_fraction
    one_fractions[unique_depth] = one_fraction
    if unique_depth == 0:
        pweights[unique_depth] = 1.
    else:
        pweights[unique_depth] = 0.

    for i in range(unique_depth - 1, -1, -1):
        pweights[i + 1] += one_fraction * pweights[i] * (i + 1.) / (unique_depth + 1.)
        pweights[i] = zero_fraction * pweights[i] * (unique_depth - i) / (unique_depth + 1.)

# undo a previous extension of the decision path
#@numba.jit(nopython=True, nogil=True)
def unwind_path(feature_indexes, zero_fractions, one_fractions, pweights,
                unique_depth, path_index):
    one_fraction = one_fractions[path_index]
    zero_fraction = zero_fractions[path_index]
    next_one_portion = pweights[unique_depth]

    for i in range(unique_depth - 1, -1, -1):
        if one_fraction != 0.:
            tmp = pweights[i]
            pweights[i] = next_one_portion * (unique_depth + 1.) / ((i + 1.) * one_fraction)
            next_one_portion = tmp - pweights[i] * zero_fraction * (unique_depth - i) / (unique_depth + 1.)
        else:
            pweights[i] = (pweights[i] * (unique_depth + 1)) / (zero_fraction * (unique_depth - i))

    for i in range(path_index, unique_depth):
        feature_indexes[i] = feature_indexes[i + 1]
        zero_fractions[i] = zero_fractions[i + 1]
        one_fractions[i] = one_fractions[i + 1]

# determine what the total permuation weight would be if
# we unwound a previous extension in the decision path
#@numba.jit(nopython=True, nogil=True)
def unwound_path_sum(feature_indexes, zero_fractions, one_fractions, pweights, unique_depth, path_index):
    one_fraction = one_fractions[path_index]
    zero_fraction = zero_fractions[path_index]
    next_one_portion = pweights[unique_depth]
    total = 0

    for i in range(unique_depth - 1, -1, -1):
        if one_fraction != 0.:
            tmp = next_one_portion * (unique_depth + 1.) / ((i + 1.) * one_fraction)
            total += tmp
            next_one_portion = pweights[i] - tmp * zero_fraction * ((unique_depth - i) / (unique_depth + 1.))
        else:
            total += (pweights[i] / zero_fraction) / ((unique_depth - i) / (unique_depth + 1.))

    return total


class Tree:
    # def __init__(self, children_left, children_right, children_default, feature, threshold, value, node_sample_weight):
    #     self.children_left = children_left.astype(np.int32)
    #     self.children_right = children_right.astype(np.int32)
    #     self.children_default = children_default.astype(np.int32)
    #     self.features = feature.astype(np.int32)
    #     self.thresholds = threshold
    #     self.values = value
    #     self.node_sample_weight = node_sample_weight

    #     self.max_depth = compute_expectations(
    #         self.children_left, self.children_right, self.node_sample_weight,
    #         self.values, 0
    #     )

    def __init__(self, tree, normalize=False):
        if str(type(tree)).endswith("'sklearn.tree._tree.Tree'>"):
            self.children_left = tree.children_left.astype(np.int32)
            self.children_right = tree.children_right.astype(np.int32)
            self.children_default = self.children_left # missing values not supported in sklearn
            self.features = tree.feature.astype(np.int32)
            self.thresholds = tree.threshold.astype(np.float64)
            if normalize:
                self.values = (tree.value[:,0,:].T / tree.value[:,0,:].sum(1)).T
            else:
                self.values = tree.value[:,0,:]


            self.node_sample_weight = tree.weighted_n_node_samples.astype(np.float64)

            # we recompute the expectations to make sure they follow the SHAP logic
            self.max_depth = compute_expectations(
                self.children_left, self.children_right, self.node_sample_weight,
                self.values, 0
            )

#@numba.jit(nopython=True)
def compute_expectations(children_left, children_right, node_sample_weight, values, i, depth=0):
    if children_right[i] == -1:
        values[i,:] = values[i,:]
        return 0
    else:
        li = children_left[i]
        ri = children_right[i]
        depth_left = compute_expectations(children_left, children_right, node_sample_weight, values, li, depth + 1)
        depth_right = compute_expectations(children_left, children_right, node_sample_weight, values, ri, depth + 1)
        left_weight = node_sample_weight[li]
        right_weight = node_sample_weight[ri]
        v = (left_weight * values[li,:] + right_weight * values[ri,:]) / (left_weight + right_weight)
        values[i,:] = v
        return max(depth_left, depth_right) + 1

# recursive computation of SHAP values for a decision tree
#@numba.jit(nopython=True, nogil=True)
def tree_shap_recursive(children_left, children_right, children_default, features, thresholds, values, node_sample_weight,
                        x, x_missing, phi, node_index, unique_depth, parent_feature_indexes,
                        parent_zero_fractions, parent_one_fractions, parent_pweights, parent_zero_fraction,
                        parent_one_fraction, parent_feature_index, condition, condition_feature, condition_fraction):

    # stop if we have no weight coming down to us
    if condition_fraction == 0.:
        return

    # extend the unique path
    feature_indexes = parent_feature_indexes[unique_depth + 1:]
    feature_indexes[:unique_depth + 1] = parent_feature_indexes[:unique_depth + 1]
    zero_fractions = parent_zero_fractions[unique_depth + 1:]
    zero_fractions[:unique_depth + 1] = parent_zero_fractions[:unique_depth + 1]
    one_fractions = parent_one_fractions[unique_depth + 1:]
    one_fractions[:unique_depth + 1] = parent_one_fractions[:unique_depth + 1]
    pweights = parent_pweights[unique_depth + 1:]
    pweights[:unique_depth + 1] = parent_pweights[:unique_depth + 1]

    if condition == 0 or condition_feature != parent_feature_index:
        extend_path(
            feature_indexes, zero_fractions, one_fractions, pweights,
            unique_depth, parent_zero_fraction, parent_one_fraction, parent_feature_index
        )

    split_index = features[node_index]

    # leaf node
    if children_right[node_index] == -1:
        for i in range(1, unique_depth+1):
            w = unwound_path_sum(feature_indexes, zero_fractions, one_fractions, pweights, unique_depth, i)
            phi[feature_indexes[i],:] += w * (one_fractions[i] - zero_fractions[i]) * values[node_index,:] * condition_fraction

    # internal node
    else:
        # find which branch is "hot" (meaning x would follow it)
        hot_index = 0
        cleft = children_left[node_index]
        cright = children_right[node_index]
        if x_missing[split_index] == 1:
            hot_index = children_default[node_index]
        elif x[split_index] < thresholds[node_index]:
            hot_index = cleft
        else:
            hot_index = cright
        cold_index = (cright if hot_index == cleft else cleft)
        w = node_sample_weight[node_index]
        hot_zero_fraction = node_sample_weight[hot_index] / w
        cold_zero_fraction = node_sample_weight[cold_index] / w
        incoming_zero_fraction = 1.
        incoming_one_fraction = 1.

        # see if we have already split on this feature,
        # if so we undo that split so we can redo it for this node
        path_index = 0
        while (path_index <= unique_depth):
            if feature_indexes[path_index] == split_index:
                break
            path_index += 1

        if path_index != unique_depth + 1:
            incoming_zero_fraction = zero_fractions[path_index]
            incoming_one_fraction = one_fractions[path_index]
            unwind_path(feature_indexes, zero_fractions, one_fractions, pweights, unique_depth, path_index)
            unique_depth -= 1

        # divide up the condition_fraction among the recursive calls
        hot_condition_fraction = condition_fraction
        cold_condition_fraction = condition_fraction
        if condition > 0 and split_index == condition_feature:
            cold_condition_fraction = 0.
            unique_depth -= 1
        elif condition < 0 and split_index == condition_feature:
            hot_condition_fraction *= hot_zero_fraction
            cold_condition_fraction *= cold_zero_fraction
            unique_depth -= 1

        tree_shap_recursive(
            children_left, children_right, children_default, features, thresholds, values, node_sample_weight,
            x, x_missing, phi, hot_index, unique_depth + 1,
            feature_indexes, zero_fractions, one_fractions, pweights,
            hot_zero_fraction * incoming_zero_fraction, incoming_one_fraction,
            split_index, condition, condition_feature, hot_condition_fraction
        )

        tree_shap_recursive(
            children_left, children_right, children_default, features, thresholds, values, node_sample_weight,
            x, x_missing, phi, cold_index, unique_depth + 1,
            feature_indexes, zero_fractions, one_fractions, pweights,
            cold_zero_fraction * incoming_zero_fraction, 0,
            split_index, condition, condition_feature, cold_condition_fraction
        )

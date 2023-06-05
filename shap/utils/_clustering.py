import warnings

import numpy as np
import scipy.cluster
import scipy.spatial
import sklearn
from numba import njit

from ._general import safe_isinstance
from ._show_progress import show_progress


def partition_tree(X, metric="correlation"):
    X_full_rank = X + np.random.randn(*X.shape) * 1e-8
    D = scipy.spatial.distance.pdist(X_full_rank.fillna(X_full_rank.mean()).T, metric=metric)
    return scipy.cluster.hierarchy.complete(D)


def partition_tree_shuffle(indexes, index_mask, partition_tree):
    """ Randomly shuffle the indexes in a way that is consistent with the given partition tree.

    Parameters
    ----------
    indexes: np.array
        The output location of the indexes we want shuffled. Note that len(indexes) should equal index_mask.sum().

    index_mask: np.array
        A bool mask of which indexes we want to include in the shuffled list.

    partition_tree: np.array
        The partition tree we should follow.
    """
    M = len(index_mask)
    #switch = np.random.randn(M) < 0
    _pt_shuffle_rec(partition_tree.shape[0]-1, indexes, index_mask, partition_tree, M, 0)
@njit
def _pt_shuffle_rec(i, indexes, index_mask, partition_tree, M, pos):
    if i < 0:
        # see if we should include this index in the ordering
        if index_mask[i + M]:
            indexes[pos] = i + M
            return pos + 1
        else:
            return pos
    left = int(partition_tree[i,0] - M)
    right = int(partition_tree[i,1] - M)
    if np.random.randn() < 0:
        pos = _pt_shuffle_rec(left, indexes, index_mask, partition_tree, M, pos)
        pos = _pt_shuffle_rec(right, indexes, index_mask, partition_tree, M, pos)
    else:
        pos = _pt_shuffle_rec(right, indexes, index_mask, partition_tree, M, pos)
        pos = _pt_shuffle_rec(left, indexes, index_mask, partition_tree, M, pos)
    return pos

@njit
def delta_minimization_order(all_masks, max_swap_size=100, num_passes=2):
    order = np.arange(len(all_masks))
    for _ in range(num_passes):
        for length in list(range(2, max_swap_size)):
            for i in range(1, len(order)-length):
                if _reverse_window_score_gain(all_masks, order, i, length) > 0:
                    _reverse_window(order, i, length)
    return order
@njit
def _reverse_window(order, start, length):
    for i in range(length // 2):
        tmp = order[start + i]
        order[start + i] = order[start + length - i - 1]
        order[start + length - i - 1] = tmp
@njit
def _reverse_window_score_gain(masks, order, start, length):
    forward_score = _mask_delta_score(masks[order[start - 1]], masks[order[start]]) + \
                    _mask_delta_score(masks[order[start + length-1]], masks[order[start + length]])
    reverse_score = _mask_delta_score(masks[order[start - 1]], masks[order[start + length-1]]) + \
                    _mask_delta_score(masks[order[start]], masks[order[start + length]])

    return forward_score - reverse_score
@njit
def _mask_delta_score(m1, m2):
    return (m1 ^ m2).sum()


def hclust_ordering(X, metric="sqeuclidean", anchor_first=False):
    """ A leaf ordering is under-defined, this picks the ordering that keeps nearby samples similar.
    """

    # compute a hierarchical clustering and return the optimal leaf ordering
    D = scipy.spatial.distance.pdist(X, metric)
    cluster_matrix = scipy.cluster.hierarchy.complete(D)
    return scipy.cluster.hierarchy.leaves_list(scipy.cluster.hierarchy.optimal_leaf_ordering(cluster_matrix, D))

def xgboost_distances_r2(X, y, learning_rate=0.6, early_stopping_rounds=2, subsample=1, max_estimators=10000, random_state=0):
    """ Compute reducancy distances scaled from 0-1 amoung all the feature in X relative to the label y.

    Distances are measured by training univariate XGBoost models of y for all the features, and then
    predicting the output of these models using univariate XGBoost models of other features. If one
    feature can effectively predict the output of another feature's univariate XGBoost model of y,
    then the second feature is redundant with the first with respect to y. A distance of 1 corresponds
    to no redundancy while a distance of 0 corresponds to perfect redundancy (measured using the
    proportion of variance explained). Note these distances are not symmetric.
    """

    import xgboost

    # pick our train/text split
    X_train,X_test,y_train,y_test = sklearn.model_selection.train_test_split(X, y, random_state=random_state)

    # fit an XGBoost model on each of the features
    test_preds = []
    train_preds = []
    for i in range(X.shape[1]):
        model = xgboost.XGBRegressor(subsample=subsample, n_estimators=max_estimators, learning_rate=learning_rate, max_depth=1)
        model.fit(X_train[:,i:i+1], y_train, eval_set=[(X_test[:,i:i+1], y_test)], early_stopping_rounds=early_stopping_rounds, verbose=False)
        train_preds.append(model.predict(X_train[:,i:i+1]))
        test_preds.append(model.predict(X_test[:,i:i+1]))
    train_preds = np.vstack(train_preds).T
    test_preds = np.vstack(test_preds).T

    # fit XGBoost models to predict the outputs of other XGBoost models to see how redundant features are
    dist = np.zeros((X.shape[1], X.shape[1]))
    for i in show_progress(range(X.shape[1]), total=X.shape[1]):
        for j in range(X.shape[1]):
            if i == j:
                dist[i,j] = 0
                continue

            # skip features that have not variance in their predictions (likely because the feature is a constant)
            preds_var = np.var(test_preds[:,i])
            if preds_var < 1e-4:
                warnings.warn(f"No/low signal found from feature {i} (this is typically caused by constant or near-constant features)! Cluster distances can't be computed for it (so setting all distances to 1).")
                r2 = 0

            # fit the model
            else:
                model = xgboost.XGBRegressor(subsample=subsample, n_estimators=max_estimators, learning_rate=learning_rate, max_depth=1)
                model.fit(X_train[:,j:j+1], train_preds[:,i], eval_set=[(X_test[:,j:j+1], test_preds[:,i])], early_stopping_rounds=early_stopping_rounds, verbose=False)
                r2 = max(0, 1 - np.mean((test_preds[:,i] - model.predict(X_test[:,j:j+1]))**2) / preds_var)
            dist[i,j] = 1 - r2

    return dist

def hclust(X, y=None, linkage="single", metric="auto", random_state=0):
    if safe_isinstance(X, "pandas.core.frame.DataFrame"):
        X = X.values

    if metric == "auto":
        if y is not None:
            metric = "xgboost_distances_r2"

    # build the distance matrix
    if metric == "xgboost_distances_r2":
        dist_full = xgboost_distances_r2(X, y, random_state=random_state)

        # build a condensed upper triangular version by taking the max distance from either direction
        dist = []
        for i in range(dist_full.shape[0]):
            for j in range(i+1, dist_full.shape[1]):
                if i != j:
                    if linkage == "single":
                        dist.append(min(dist_full[i,j], dist_full[j,i]))
                    elif linkage == "complete":
                        dist.append(max(dist_full[i,j], dist_full[j,i]))
                    elif linkage == "average":
                        dist.append((dist_full[i,j] + dist_full[j,i]) / 2)
                    else:
                        raise Exception("Unsupported linkage type!")
        dist = np.array(dist)

    else:
        if y is not None:
            warnings.warn("Ignoring the y argument passed to shap.utils.hclust since the given clustering metric is not based on label fitting!")
        if safe_isinstance(X, "pandas.core.frame.DataFrame"):
            bg_no_nan = X.values.copy()
        else:
            bg_no_nan = X.copy()
        for i in range(bg_no_nan.shape[1]):
            np.nan_to_num(bg_no_nan[:,i], nan=np.nanmean(bg_no_nan[:,i]), copy=False)
        dist = scipy.spatial.distance.pdist(bg_no_nan.T + np.random.randn(*bg_no_nan.T.shape)*1e-8, metric=metric)
    # else:
    #     raise Exception("Unknown metric: " + str(metric))

    # build linkage
    if linkage == "single":
        return scipy.cluster.hierarchy.single(dist)
    elif linkage == "complete":
        return scipy.cluster.hierarchy.complete(dist)
    elif linkage == "average":
        return scipy.cluster.hierarchy.average(dist)
    else:
        raise Exception("Unknown linkage: " + str(linkage))

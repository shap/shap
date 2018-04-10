import numpy as np
#import numba
from .. import _cext

try:
    import xgboost
except ImportError:
    pass

try:
    import lightgbm
except ImportError:
    pass

class TreeExplainer:
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
            raise Exception("Model type not supported by TreeExplainer: " + str(type(model)))

    def shap_values(self, X, **kwargs):

        # shortcut using the C++ version of Tree SHAP in XGBoost and LightGBM
        # these are about 10x faster than the numba jit'd implementation below...
        if self.model_type == "xgboost":
            if not str(type(X)).endswith("xgboost.core.DMatrix'>"):
                X = xgboost.DMatrix(X)
            return self.trees.predict(X, pred_contribs=True)
        elif self.model_type == "lightgbm":
            return self.trees.predict(X, pred_contrib=True)

        # convert dataframes
        if str(type(X)).endswith("pandas.core.series.Series'>"):
            X = X.as_matrix()
        elif str(type(X)).endswith("pandas.core.frame.DataFrame'>"):
            X = X.as_matrix()

        assert str(type(X)).endswith("'numpy.ndarray'>"), "Unknown instance type: " + str(type(X))
        assert len(X.shape) == 1 or len(X.shape) == 2, "Instance must have 1 or 2 dimensions!"

        n_outputs = self.trees[0].values.shape[1]

        # single instance
        if len(X.shape) == 1:

            phi = np.zeros((X.shape[0] + 1, n_outputs))
            x_missing = np.zeros(X.shape[0], dtype=np.bool)
            for t in self.trees:
                self.tree_shap(t, X, x_missing, phi)
            phi /= len(self.trees)

            if n_outputs == 1:
                return phi[:, 0]
            else:
                return [phi[:, i] for i in range(n_outputs)]

        elif len(X.shape) == 2:
            phi = np.zeros((X.shape[0], X.shape[1] + 1, n_outputs))
            x_missing = np.zeros(X.shape[1], dtype=np.bool)
            for i in range(X.shape[0]):
                for t in self.trees:
                    self.tree_shap(t, X[i,:], x_missing, phi[i,:,:])
            phi /= len(self.trees)

            if n_outputs == 1:
                return phi[:, :, 0]
            else:
                return [phi[:, :, i] for i in range(n_outputs)]

    def shap_interaction_values(self, X, **kwargs):

        # shortcut using the C++ version of Tree SHAP in XGBoost and LightGBM
        if self.model_type == "xgboost":
            if not str(type(X)).endswith("xgboost.core.DMatrix'>"):
                X = xgboost.DMatrix(X)
            return self.trees.predict(X, pred_interactions=True)
        else:
            raise Exception("Interaction values not yet supported for model type: " + str(type(X)))

    def tree_shap(self, tree, x, x_missing, phi, condition=0, condition_feature=0):

        # start the recursive algorithm
        _cext.tree_shap(
            tree.max_depth, tree.children_left, tree.children_right, tree.children_default, tree.features,
            tree.thresholds, tree.values, tree.node_sample_weight,
            x, x_missing, phi, condition, condition_feature
        )


class Tree:
    def __init__(self, children_left, children_right, children_default, feature, threshold, value, node_sample_weight):
        self.children_left = children_left.astype(np.int32)
        self.children_right = children_right.astype(np.int32)
        self.children_default = children_default.astype(np.int32)
        self.features = feature.astype(np.int32)
        self.thresholds = threshold
        self.values = value
        self.node_sample_weight = node_sample_weight

        # we compute the expectations to make sure they follow the SHAP logic
        self.max_depth = _cext.compute_expectations(
            self.children_left, self.children_right, self.node_sample_weight,
            self.values
        )

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

            # we compute the expectations to make sure they follow the SHAP logic
            self.max_depth = _cext.compute_expectations(
                self.children_left, self.children_right, self.node_sample_weight,
                self.values
            )

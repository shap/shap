import numpy as np
import multiprocessing
import sys
from .explainer import Explainer

have_cext = False
try:
    from .. import _cext
    have_cext = True
except ImportError:
    pass
except:
    print("the C extension is installed...but failed to load!")
    pass

try:
    import xgboost
except ImportError:
    pass
except:
    print("xgboost is installed...but failed to load!")
    pass

try:
    import lightgbm
except ImportError:
    pass
except:
    print("lightgbm is installed...but failed to load!")
    pass

try:
    import catboost
except ImportError:
    pass
except:
    print("catboost is installed...but failed to load!")
    pass


class TreeExplainer(Explainer):
    """Uses the Tree SHAP method to explain the output of ensemble tree models.

    Tree SHAP is a fast and exact method to estimate SHAP values for tree models and ensembles
    of trees. It depends on fast C++ implementations either inside the package or in the
    compiled C extention.
    """

    def __init__(self, model, **kwargs):
        self.model_type = "internal"
        self.less_than_or_equal = False # are threshold comparisons < or <= for this model
        self.base_offset = 0.0
        self.expected_value = None

        if str(type(model)).endswith("sklearn.ensemble.forest.RandomForestRegressor'>"):
            self.trees = [Tree(e.tree_) for e in model.estimators_]
            self.less_than_or_equal = True
        elif str(type(model)).endswith("sklearn.tree.tree.DecisionTreeRegressor'>"):
            self.trees = [Tree(model.tree_)]
            self.less_than_or_equal = True
        elif str(type(model)).endswith("sklearn.tree.tree.DecisionTreeClassifier'>"):
            self.trees = [Tree(model.tree_)]
            self.less_than_or_equal = True
        elif str(type(model)).endswith("sklearn.ensemble.forest.RandomForestClassifier'>"):
            self.trees = [Tree(e.tree_, normalize=True) for e in model.estimators_]
            self.less_than_or_equal = True
        elif str(type(model)).endswith("sklearn.ensemble.forest.ExtraTreesClassifier'>"): # TODO: add unit test for this case
            self.trees = [Tree(e.tree_, normalize=True) for e in model.estimators_]
            self.less_than_or_equal = True
        elif str(type(model)).endswith("sklearn.ensemble.gradient_boosting.GradientBoostingRegressor'>"): # TODO: add unit test for this case

            # currently we only support the mean estimator
            if str(type(model.init_)).endswith("ensemble.gradient_boosting.MeanEstimator'>"):
                self.base_offset = model.init_.mean
            else:
                assert False, "Unsupported init model type: " + str(type(gb_model.init_))

            scale = len(model.estimators_) * model.learning_rate
            self.trees = [Tree(e.tree_, scaling=scale) for e in model.estimators_[:,0]]
            self.less_than_or_equal = True
        elif str(type(model)).endswith("xgboost.core.Booster'>"):
            self.model_type = "xgboost"
            self.trees = model
        elif str(type(model)).endswith("xgboost.sklearn.XGBClassifier'>"):
            self.model_type = "xgboost"
            self.trees = model.get_booster()
        elif str(type(model)).endswith("xgboost.sklearn.XGBRegressor'>"):
            self.model_type = "xgboost"
            self.trees = model.get_booster()
        elif str(type(model)).endswith("lightgbm.basic.Booster'>"):
            self.model_type = "lightgbm"
            self.model = model
            tree_info = self.model.dump_model()["tree_info"]
            self.trees = [Tree(e, scaling=len(tree_info)) for e in tree_info]
        elif str(type(model)).endswith("lightgbm.sklearn.LGBMRegressor'>"):
            self.model_type = "lightgbm"
            self.model = model.booster_
            tree_info = self.model.dump_model()["tree_info"]
            self.trees = [Tree(e, scaling=len(tree_info)) for e in tree_info]
        elif str(type(model)).endswith("lightgbm.sklearn.LGBMClassifier'>"):
            self.model_type = "lightgbm"
            self.model = model.booster_
            tree_info = self.model.dump_model()["tree_info"]
            self.trees = [Tree(e, scaling=len(tree_info)) for e in tree_info]
        elif str(type(model)).endswith("catboost.core.CatBoostRegressor'>"):
            self.model_type = "catboost"
            self.trees = model
        elif str(type(model)).endswith("catboost.core.CatBoostClassifier'>"):
            self.model_type = "catboost"
            self.trees = model
        else:
            raise Exception("Model type not yet supported by TreeExplainer: " + str(type(model)))

    def shap_values(self, X, tree_limit=-1, **kwargs):
        """ Estimate the SHAP values for a set of samples.

        Parameters
        ----------
        X : numpy.array or pandas.DataFrame
            A matrix of samples (# samples x # features) on which to explain the model's output.

        Returns
        -------
        For models with a single output this returns a matrix of SHAP values
        (# samples x # features). Each row sums to the difference between the model output for that
        sample and the expected value of the model output (which is stored as expected_value
        attribute of the explainer). For models with vector outputs this returns a list
        of such matrices, one for each output.
        """

        # shortcut using the C++ version of Tree SHAP in XGBoost, LightGBM, and CatBoost
        phi = None
        if self.model_type == "xgboost":
            if not str(type(X)).endswith("xgboost.core.DMatrix'>"):
                X = xgboost.DMatrix(X)
            if tree_limit == -1:
                tree_limit = 0
            phi = self.trees.predict(X, ntree_limit=tree_limit, pred_contribs=True)
        elif self.model_type == "lightgbm":
            phi = self.model.predict(X, num_iteration=tree_limit, pred_contrib=True)
            if phi.shape[1] != X.shape[1] + 1:
                phi = phi.reshape(X.shape[0], phi.shape[1]//(X.shape[1]+1), X.shape[1]+1)
        elif self.model_type == "catboost": # thanks to the CatBoost team for implementing this...
            assert tree_limit == -1, "tree_limit is not yet supported for CatBoost models!"
            phi = self.trees.get_feature_importance(data=catboost.Pool(X), fstr_type='ShapValues')

        # note we pull off the last column and keep it as our expected_value
        if phi is not None:
            if len(phi.shape) == 3:
                self.expected_value = [phi[0, i, -1] for i in range(phi.shape[1])]
                return [phi[:, i, :-1] for i in range(phi.shape[1])]
            else:
                self.expected_value = phi[0, -1]
                return phi[:, :-1]

        # convert dataframes
        if str(type(X)).endswith("pandas.core.series.Series'>"):
            X = X.values
        elif str(type(X)).endswith("pandas.core.frame.DataFrame'>"):
            X = X.values

        assert str(type(X)).endswith("'numpy.ndarray'>"), "Unknown instance type: " + str(type(X))
        assert len(X.shape) == 1 or len(X.shape) == 2, "Instance must have 1 or 2 dimensions!"

        if tree_limit<0 or tree_limit>len(self.trees):
            self.tree_limit = len(self.trees)
        else:
            self.tree_limit = tree_limit

        self.n_outputs = self.trees[0].values.shape[1]
        # single instance
        if len(X.shape) == 1:
            self._current_X = X.reshape(1,X.shape[0])
            self._current_x_missing = np.zeros(X.shape[0], dtype=np.bool)
            phi = self._tree_shap_ind(0)

            # note we pull off the last column and keep it as our expected_value
            if self.n_outputs == 1:
                self.expected_value = phi[-1, 0]
                return phi[:-1, 0]
            else:
                self.expected_value = [phi[-1, i] for i in range(phi.shape[1])]
                return [phi[:-1, i] for i in range(self.n_outputs)]

        elif len(X.shape) == 2:
            x_missing = np.zeros(X.shape[1], dtype=np.bool)
            self._current_X = X
            self._current_x_missing = x_missing

            # Only python 3 can serialize a method to send to another process
            if sys.version_info[0] >= 3:
                pool = multiprocessing.Pool()
                phi = np.stack(pool.map(self._tree_shap_ind, range(X.shape[0])), 0)
                pool.close()
                pool.join()
            else:
                phi = np.stack(map(self._tree_shap_ind, range(X.shape[0])), 0)

            # note we pull off the last column and keep it as our expected_value
            if self.n_outputs == 1:
                self.expected_value = phi[0, -1, 0]
                return phi[:, :-1, 0]
            else:
                self.expected_value = [phi[0, -1, i] for i in range(phi.shape[2])]
                return [phi[:, :-1, i] for i in range(self.n_outputs)]

    def shap_interaction_values(self, X, tree_limit=-1, **kwargs):

        # shortcut using the C++ version of Tree SHAP in XGBoost and LightGBM
        if self.model_type == "xgboost":
            if not str(type(X)).endswith("xgboost.core.DMatrix'>"):
                X = xgboost.DMatrix(X)
            if tree_limit==-1:
                tree_limit=0
            phi = self.trees.predict(X, ntree_limit=tree_limit, pred_interactions=True)

            # note we pull off the last column and keep it as our expected_value
            if len(phi.shape) == 4:
                self.expected_value = [phi[0, i, -1, -1] for i in range(phi.shape[1])]
                return [phi[:, i, :-1, :-1] for i in range(phi.shape[1])]
            else:
                self.expected_value = phi[0, -1, -1]
                return phi[:, :-1, :-1]
        else:

            if str(type(X)).endswith("pandas.core.series.Series'>"):
                X = X.values
            elif str(type(X)).endswith("pandas.core.frame.DataFrame'>"):
                X = X.values

            assert str(type(X)).endswith("'numpy.ndarray'>"), "Unknown instance type: " + str(type(X))
            assert len(X.shape) == 1 or len(X.shape) == 2, "Instance must have 1 or 2 dimensions!"

            self.n_outputs = self.trees[0].values.shape[1]

            if tree_limit < 0 or tree_limit > len(self.trees):
                self.tree_limit = len(self.trees)
            else:
                self.tree_limit = tree_limit

            self.n_outputs = self.trees[0].values.shape[1]
            # single instance
            if len(X.shape) == 1:
                self._current_X = X.reshape(1,X.shape[0])
                self._current_x_missing = np.zeros(X.shape[0], dtype=np.bool)
                phi = self._tree_shap_ind_interactions(0)

                # note we pull off the last column and keep it as our expected_value
                if self.n_outputs == 1:
                    self.expected_value = phi[-1, -1, 0]
                    return phi[:-1, :-1, 0]
                else:
                    self.expected_value = [phi[-1, -1, i] for i in range(phi.shape[2])]
                    return [phi[:-1, :-1, i] for i in range(self.n_outputs)]

            elif len(X.shape) == 2:
                x_missing = np.zeros(X.shape[1], dtype=np.bool)
                self._current_X = X
                self._current_x_missing = x_missing

                # Only python 3 can serialize a method to send to another process
                # TODO: LightGBM models are attached to this object and this seems to cause pool.map to hang
                if sys.version_info[0] >= 3 and self.model_type != "lightgbm":
                    pool = multiprocessing.Pool()
                    phi = np.stack(pool.map(self._tree_shap_ind_interactions, range(X.shape[0])), 0)
                    pool.close()
                    pool.join()
                else:
                    phi = np.stack(map(self._tree_shap_ind_interactions, range(X.shape[0])), 0)

                # note we pull off the last column and keep it as our expected_value
                if self.n_outputs == 1:
                    self.expected_value = phi[0, -1, -1, 0]
                    return phi[:, :-1, :-1, 0]
                else:
                    self.expected_value = [phi[0, -1, -1, i] for i in range(phi.shape[3])]
                    return [phi[:, :-1, :-1, i] for i in range(self.n_outputs)]

    def _tree_shap_ind(self, i):
        phi = np.zeros((self._current_X.shape[1] + 1, self.n_outputs))
        phi[-1, :] = self.base_offset * self.tree_limit
        for t in range(self.tree_limit):
            self.tree_shap(self.trees[t], self._current_X[i,:], self._current_x_missing, phi)
        phi /= self.tree_limit
        return phi

    def _tree_shap_ind_interactions(self, i):
        phi = np.zeros((self._current_X.shape[1] + 1, self._current_X.shape[1] + 1, self.n_outputs))
        phi_diag = np.zeros((self._current_X.shape[1] + 1, self.n_outputs))
        for t in range(self.tree_limit):
            self.tree_shap(self.trees[t], self._current_X[i,:], self._current_x_missing, phi_diag)
            for j in self.trees[t].unique_features:
                phi_on = np.zeros((self._current_X.shape[1] + 1, self.n_outputs))
                phi_off = np.zeros((self._current_X.shape[1] + 1, self.n_outputs))
                self.tree_shap(self.trees[t], self._current_X[i,:], self._current_x_missing, phi_on, 1, j)
                self.tree_shap(self.trees[t], self._current_X[i,:], self._current_x_missing, phi_off, -1, j)
                phi[j] += np.true_divide(np.subtract(phi_on,phi_off),2.0)
                phi_diag[j] -= np.sum(np.true_divide(np.subtract(phi_on,phi_off),2.0))
        for j in range(self._current_X.shape[1]+1):
            phi[j][j] = phi_diag[j]
        phi /= self.tree_limit
        return phi

    def tree_shap(self, tree, x, x_missing, phi, condition=0, condition_feature=0):
        # start the recursive algorithm
        assert have_cext, "C extension was not built during install!"
        _cext.tree_shap(
            tree.max_depth, tree.children_left, tree.children_right, tree.children_default, tree.features,
            tree.thresholds, tree.values, tree.node_sample_weight,
            x, x_missing, phi, condition, condition_feature, self.less_than_or_equal
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
        self.unique_features = np.unique(self.features)
        self.unique_features = np.delete(self.unique_features, np.where(self.unique_features==-1))
        # we compute the expectations to make sure they follow the SHAP logic
        assert have_cext, "C extension was not built during install!"
        self.max_depth = _cext.compute_expectations(
            self.children_left, self.children_right, self.node_sample_weight,
            self.values
        )

    def __init__(self, tree, normalize=False, scaling=1.0):
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
            self.values = self.values * scaling

            self.node_sample_weight = tree.weighted_n_node_samples.astype(np.float64)
            self.unique_features = np.unique(self.features)
            self.unique_features = np.delete(self.unique_features, np.where(self.unique_features < 0))

            # we compute the expectations to make sure they follow the SHAP logic
            self.max_depth = _cext.compute_expectations(
                self.children_left, self.children_right, self.node_sample_weight,
                self.values
            )

        elif type(tree) == dict:
            start = tree['tree_structure']
            num_parents = tree['num_leaves']-1
            self.children_left = np.empty((2*num_parents+1), dtype=np.int32)
            self.children_right = np.empty((2*num_parents+1), dtype=np.int32)
            self.children_default = np.empty((2*num_parents+1), dtype=np.int32)
            self.features = np.empty((2*num_parents+1), dtype=np.int32)
            self.thresholds = np.empty((2*num_parents+1), dtype=np.float64)
            self.values = [-2]*(2*num_parents+1)
            self.node_sample_weight = np.empty((2*num_parents+1), dtype=np.float64)
            visited, queue = [], [start]
            while queue:
                vertex = queue.pop(0)
                if 'split_index' in vertex.keys():
                    if vertex['split_index'] not in visited:
                        if 'split_index' in vertex['left_child'].keys():
                            self.children_left[vertex['split_index']] = vertex['left_child']['split_index']
                        else:
                            self.children_left[vertex['split_index']] = vertex['left_child']['leaf_index']+num_parents
                        if 'split_index' in vertex['right_child'].keys():
                            self.children_right[vertex['split_index']] = vertex['right_child']['split_index']
                        else:
                            self.children_right[vertex['split_index']] = vertex['right_child']['leaf_index']+num_parents
                        if vertex['default_left']:
                            self.children_default[vertex['split_index']] = self.children_left[vertex['split_index']]
                        else:
                            self.children_default[vertex['split_index']] = self.children_right[vertex['split_index']]
                        self.features[vertex['split_index']] = vertex['split_feature']
                        self.thresholds[vertex['split_index']] = vertex['threshold']
                        self.values[vertex['split_index']] = [vertex['internal_value']]
                        self.node_sample_weight[vertex['split_index']] = vertex['internal_count']
                        visited.append(vertex['split_index'])
                        queue.append(vertex['left_child'])
                        queue.append(vertex['right_child'])
                else:
                    self.children_left[vertex['leaf_index']+num_parents] = -1
                    self.children_right[vertex['leaf_index']+num_parents] = -1
                    self.children_default[vertex['leaf_index']+num_parents] = -1
                    self.features[vertex['leaf_index']+num_parents] = -1
                    self.children_left[vertex['leaf_index']+num_parents] = -1
                    self.children_right[vertex['leaf_index']+num_parents] = -1
                    self.children_default[vertex['leaf_index']+num_parents] = -1
                    self.features[vertex['leaf_index']+num_parents] = -1
                    self.thresholds[vertex['leaf_index']+num_parents] = -1
                    self.values[vertex['leaf_index']+num_parents] = [vertex['leaf_value']]
                    self.node_sample_weight[vertex['leaf_index']+num_parents] = vertex['leaf_count']
            self.values = np.asarray(self.values)
            self.values = np.multiply(self.values, scaling)
            self.unique_features = np.unique(self.features)
            self.unique_features = np.delete(self.unique_features, np.where(self.unique_features < 0))

            assert have_cext, "C extension was not built during install!" + str(have_cext)
            self.max_depth = _cext.compute_expectations(
                self.children_left, self.children_right, self.node_sample_weight,
                self.values
            )

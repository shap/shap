import numpy as np
import multiprocessing
import sys
import json
import os
import struct
from distutils.version import LooseVersion
from .explainer import Explainer
from ..common import assert_import, record_import_error, DenseData

try:
    from .. import _cext
except ImportError as e:
    record_import_error("cext", "C extension was not built during install!", e)

try:
    import xgboost
except ImportError as e:
    record_import_error("xgboost", "XGBoost could not be imported!", e)

try:
    import lightgbm
except ImportError as e:
    record_import_error("lightgbm", "LightGBM could not be imported!", e)

try:
    import catboost
except ImportError as e:
    record_import_error("catboost", "CatBoost could not be imported!", e)

output_transform_codes = {
    "identity": 0,
    "logistic": 1,
    "logistic_nlogloss": 2,
    "squared_loss": 3
}

feature_dependence_codes = {
    "independent": 0,
    "tree_path_dependent": 1,
    "global_path_dependent": 2
}

class TreeExplainer(Explainer):
    """Uses Tree SHAP algorithms to explain the output of ensemble tree models.

    Tree SHAP is a fast and exact method to estimate SHAP values for tree models and ensembles of trees,
    under several different possible assumptions about feature dependence. It depends on fast C++
    implementations either inside an externel model package or in the local compiled C extention.

    Parameters
    ----------
    model : model object
        The tree based machine learning model that we want to explain. XGBoost, LightGBM, CatBoost,
        and most tree-based scikit-learn models are supported.

    data : numpy.array or pandas.DataFrame
        The background dataset to use for integrating out features. This argument is optional when
        feature_dependence="tree_path_dependent", since in that case we can use the number of training
        samples that went down each tree path as our background dataset (this is recorded in the model object).

    feature_dependence : "tree_path_dependent" (default) or "independent"
        Since SHAP values rely on conditional expectations we need to decide how to handle correlated
        (or otherwise dependent) input features. The default "tree_path_dependent" approach is to just
        follow the trees and use the number of training examples that went down each leaf to represent
        the background distribution. This approach repects feature dependecies along paths in the trees.
        However, for non-linear marginal transforms (like explaining the model loss)  we don't yet
        have fast algorithms that respect the tree path dependence, so instead we offer an "independent"
        approach that breaks the dependencies between features, but allows us to explain non-linear
        transforms of the model's output. Note that the "independent" option requires a background
        dataset and its runtime scales linearly with the size of the background dataset you use. Anywhere
        from 100 to 1000 random background samples are good sizes to use.
    
    model_output : "margin", "probability", or "log_loss"
        What output of the model should be explained. If "margin" then we explain the raw output of the
        trees, which varies by model (for binary classification in XGBoost this is the log odds ratio).
        If "probability" then we explain the output of the model transformed into probability space
        (note that this means the SHAP values now sum to the probability output of the model). If "log_loss"
        then we explain the log base e of the model loss function, so that the SHAP values sum up to the
        log loss of the model for each sample. This is helpful for breaking down model performance by feature.
        Currently the probability and log_loss options are only supported when feature_dependence="independent".
    """

    def __init__(self, model, data = None, model_output = "margin", feature_dependence = "tree_path_dependent"):
        if str(type(data)).endswith("pandas.core.frame.DataFrame'>"):
            self.data = data.values
        elif isinstance(data, DenseData):
            self.data = data.data
        else:
            self.data = data
        self.data_missing = None if self.data is None else np.isnan(self.data)
        self.model_output = model_output
        self.feature_dependence = feature_dependence
        self.expected_value = None
        self.model = TreeEnsemble(model, self.data, self.data_missing)

        assert feature_dependence in feature_dependence_codes, "Invalid feature_dependence option!"

        # check for unsupported combinations of feature_dependence and model_outputs
        if feature_dependence == "tree_path_dependent":
            assert model_output == "margin", "Only margin model_output is supported for feature_dependence=\"tree_path_dependent\""
        else:   
            assert data is not None, "A background dataset must be provided unless you are using feature_dependence=\"tree_path_dependent\"!"

        if model_output != "margin":
            if self.model.objective is None and self.model.tree_output is None:
                raise Exception("Model does have a known objective or output type! When model_output is " \
                                "not \"margin\" then we need to know the model's objective or link function.")

        # A bug in XGBoost fixed in v0.81 makes XGBClassifier fail to give margin outputs
        if str(type(model)).endswith("xgboost.sklearn.XGBClassifier'>") and model_output != "margin":
            assert_import("xgboost")
            assert LooseVersion(xgboost.__version__) >= LooseVersion('0.81'), \
                "A bug in XGBoost fixed in v0.81 makes XGBClassifier fail to give margin outputs! Please upgrade to XGBoost >= v0.81!"

        # compute the expected value if we have a parsed tree for the cext
        if self.model_output == "logloss":
            self.expected_value = self.__dynamic_expected_value
        elif data is not None:
            self.expected_value = self.model.predict(self.data, output=model_output).mean(0)
        elif hasattr(self.model, "node_sample_weight"):
            self.expected_value = self.model.values[:,0].sum(0)
            self.expected_value += self.model.base_offset

    def __dynamic_expected_value(self, y):
        """ This computes the expected value conditioned on the given label value.
        """

        return self.model.predict(self.data, np.ones(self.data.shape[0]) * y, output=self.model_output).mean(0)
        
    def shap_values(self, X, y=None, tree_limit=None, approximate=False, model_stack = False):
        """ Estimate the SHAP values for a set of samples.

        Parameters
        ----------
        X : numpy.array, pandas.DataFrame or catboost.Pool (for catboost)
            A matrix of samples (# samples x # features) on which to explain the model's output.

        y : numpy.array
            An array of label values for each sample. Used when explaining loss functions.

        tree_limit : None (default) or int 
            Limit the number of trees used by the model. By default None means no use the limit of the
            original model, and -1 means no limit.

        approximate : bool
            Run fast, but only roughly approximate the Tree SHAP values. This runs a method
            previously proposed by Saabas which only considers a single feature ordering. Take care
            since this does not have the consistency guarantees of Shapley values and places too
            much weight on lower splits in the tree.

        Returns
        -------
        For models with a single output this returns a matrix of SHAP values
        (# samples x # features). Each row sums to the difference between the model output for that
        sample and the expected value of the model output (which is stored in the expected_value
        attribute of the explainer when it is constant). For models with vector outputs this returns
        a list of such matrices, one for each output.
        """
        # see if we have a default tree_limit in place.
        if tree_limit is None:
            tree_limit = -1 if self.model.tree_limit is None else self.model.tree_limit

        # shortcut using the C++ version of Tree SHAP in XGBoost, LightGBM, and CatBoost
        if self.feature_dependence == "tree_path_dependent" and self.model.model_type != "internal" and self.data is None:
            phi = None
            if self.model.model_type == "xgboost":
                assert_import("xgboost")
                if not str(type(X)).endswith("xgboost.core.DMatrix'>"):
                    X = xgboost.DMatrix(X)
                if tree_limit == -1:
                    tree_limit = 0
                phi = self.model.original_model.predict(
                    X, ntree_limit=tree_limit, pred_contribs=True,
                    approx_contribs=approximate, validate_features=False
                )
            
            elif self.model.model_type == "lightgbm":
                assert not approximate, "approximate=True is not supported for LightGBM models!"
                phi = self.model.original_model.predict(X, num_iteration=tree_limit, pred_contrib=True)
                if phi.shape[1] != X.shape[1] + 1:
                    phi = phi.reshape(X.shape[0], phi.shape[1]//(X.shape[1]+1), X.shape[1]+1)
            
            elif self.model.model_type == "catboost": # thanks to the CatBoost team for implementing this...
                assert not approximate, "approximate=True is not supported for CatBoost models!"
                assert tree_limit == -1, "tree_limit is not yet supported for CatBoost models!"
                if type(X) != catboost.Pool:
                    X = catboost.Pool(X)
                phi = self.model.original_model.get_feature_importance(data=X, fstr_type='ShapValues')

            # note we pull off the last column and keep it as our expected_value
            if phi is not None:
                if len(phi.shape) == 3:
                    self.expected_value = [phi[0, i, -1] for i in range(phi.shape[1])]
                    return [phi[:, i, :-1] for i in range(phi.shape[1])]
                else:
                    self.expected_value = phi[0, -1]
                    return phi[:, :-1]

        # convert dataframes
        orig_X = X
        if str(type(X)).endswith("pandas.core.series.Series'>"):
            X = X.values
        elif str(type(X)).endswith("pandas.core.frame.DataFrame'>"):
            X = X.values
        flat_output = False
        if len(X.shape) == 1:
            flat_output = True
            X = X.reshape(1, X.shape[0])
        if X.dtype != self.model.dtype:
            X = X.astype(self.model.dtype)
        X_missing = np.isnan(X, dtype=np.bool)
        assert str(type(X)).endswith("'numpy.ndarray'>"), "Unknown instance type: " + str(type(X))
        assert len(X.shape) == 2, "Passed input data matrix X must have 1 or 2 dimensions!"

        if tree_limit < 0 or tree_limit > self.model.values.shape[0]:
            tree_limit = self.model.values.shape[0]
        
        if self.model_output == "logloss":
            assert y is not None, "Both samples and labels must be provided when explaining the loss (i.e. `explainer.shap_values(X, y)`)!"
            assert X.shape[0] == len(y), "The number of labels (%d) does not match the number of samples to explain (%d)!" % (len(y), X.shape[0])
        transform = self.model.get_transform(self.model_output)

        if self.feature_dependence == "tree_path_dependent":
            assert self.model.fully_defined_weighting, "The background dataset you provided does not cover all the leaves in the model, " \
                                                       "so TreeExplainer cannot run with the feature_dependence=\"tree_path_dependent\" option! " \
                                                       "Try providing a larger background dataset, or using feature_dependence=\"independent\"."
 
        # run the core algorithm using the C extension
        assert_import("cext")
        if not model_stack:
            phi = np.zeros((X.shape[0], X.shape[1]+1, self.model.n_outputs))
        else:
            # In this case, we are doing independent tree shap
            phi = np.zeros((X.shape[0], X.shape[1]+1, self.model.n_outputs, self.data.shape[0]))
        if not approximate:
            _cext.dense_tree_shap(
                self.model.children_left, self.model.children_right, self.model.children_default,
                self.model.features, self.model.thresholds, self.model.values, self.model.node_sample_weight,
                self.model.max_depth, X, X_missing, y, self.data, self.data_missing, tree_limit,
                self.model.base_offset, phi, feature_dependence_codes[self.feature_dependence],
                output_transform_codes[transform], False, model_stack
            )
        else:
            _cext.dense_tree_saabas(
                self.model.children_left, self.model.children_right, self.model.children_default,
                self.model.features, self.model.thresholds, self.model.values,
                self.model.max_depth, tree_limit, self.model.base_offset, output_transform_codes[transform], 
                X, X_missing, y, phi
            )

        # note we pull off the last column and keep it as our expected_value
        if self.model.n_outputs == 1:
            if self.model_output != "logloss":
                self.expected_value = phi[0, -1, 0]
            if flat_output:
                return phi[0, :-1, 0]
            else:
                return phi[:, :-1, 0]
        else:
            if self.model_output != "logloss":
                self.expected_value = [phi[0, -1, i] for i in range(phi.shape[2])]
            if flat_output:
                return [phi[0, :-1, i] for i in range(self.model.n_outputs)]
            else:
                return [phi[:, :-1, i] for i in range(self.model.n_outputs)]

    def shap_interaction_values(self, X, y=None, tree_limit=None, model_stack = False):
        """ Estimate the SHAP interaction values for a set of samples.

        Parameters
        ----------
        X : numpy.array, pandas.DataFrame or catboost.Pool (for catboost)
            A matrix of samples (# samples x # features) on which to explain the model's output.

        y : numpy.array
            An array of label values for each sample. Used when explaining loss functions (not yet supported).

        tree_limit : None (default) or int 
            Limit the number of trees used by the model. By default None means no use the limit of the
            original model, and -1 means no limit.

        Returns
        -------
        For models with a single output this returns a tensor of SHAP values
        (# samples x # features x # features). The matrix (# features x # features) for each sample sums
        to the difference between the model output for that sample and the expected value of the model output
        (which is stored in the expected_value attribute of the explainer). Each row of this matrix sums to the
        SHAP value for that feature for that sample. The diagonal entries of the matrix represent the
        "main effect" of that feature on the prediction and the symmetric off-diagonal entries represent the
        interaction effects between all pairs of features for that sample. For models with vector outputs
        this returns a list of tensors, one for each output.
        """

        assert self.model_output == "margin", "Only model_output = \"margin\" is supported for SHAP interaction values right now!"
        assert self.feature_dependence == "tree_path_dependent", "Only feature_dependence = \"tree_path_dependent\" is supported for SHAP interaction values right now!"
        transform = "identity"

        # see if we have a default tree_limit in place.
        if tree_limit is None:
            tree_limit = -1 if self.model.tree_limit is None else self.model.tree_limit

        # shortcut using the C++ version of Tree SHAP in XGBoost
        if self.model.model_type == "xgboost":
            assert_import("xgboost")
            if not str(type(X)).endswith("xgboost.core.DMatrix'>"):
                X = xgboost.DMatrix(X)
            if tree_limit == -1:
                tree_limit = 0
            phi = self.model.original_model.predict(X, ntree_limit=tree_limit, pred_interactions=True)

            # note we pull off the last column and keep it as our expected_value
            if len(phi.shape) == 4:
                self.expected_value = [phi[0, i, -1, -1] for i in range(phi.shape[1])]
                return [phi[:, i, :-1, :-1] for i in range(phi.shape[1])]
            else:
                self.expected_value = phi[0, -1, -1]
                return phi[:, :-1, :-1]

        # convert dataframes
        if str(type(X)).endswith("pandas.core.series.Series'>"):
            X = X.values
        elif str(type(X)).endswith("pandas.core.frame.DataFrame'>"):
            X = X.values
        flat_output = False
        if len(X.shape) == 1:
            flat_output = True
            X = X.reshape(1, X.shape[0])
        if X.dtype != self.model.dtype:
            X = X.astype(self.model.dtype)
        X_missing = np.isnan(X, dtype=np.bool)
        assert str(type(X)).endswith("'numpy.ndarray'>"), "Unknown instance type: " + str(type(X))
        assert len(X.shape) == 2, "Passed input data matrix X must have 1 or 2 dimensions!"

        if tree_limit < 0 or tree_limit > self.model.values.shape[0]:
            tree_limit = self.model.values.shape[0]

        # run the core algorithm using the C extension
        assert_import("cext")
        phi = np.zeros((X.shape[0], X.shape[1]+1, X.shape[1]+1, self.model.n_outputs))
        _cext.dense_tree_shap(
            self.model.children_left, self.model.children_right, self.model.children_default,
            self.model.features, self.model.thresholds, self.model.values, self.model.node_sample_weight,
            self.model.max_depth, X, X_missing, y, self.data, self.data_missing, tree_limit,
            self.model.base_offset, phi, feature_dependence_codes[self.feature_dependence],
            output_transform_codes[transform], True, model_stack
        )

        # note we pull off the last column and keep it as our expected_value
        if self.model.n_outputs == 1:
            self.expected_value = phi[0, -1, -1, 0]
            if flat_output:
                return phi[0, :-1, :-1, 0]
            else:
                return phi[:, :-1, :-1, 0]
        else:
            self.expected_value = [phi[0, -1, -1, i] for i in range(phi.shape[3])]
            if flat_output:
                return [phi[0, :-1, :-1, i] for i in range(self.model.n_outputs)]
            else:
                return [phi[:, :-1, :-1, i] for i in range(self.model.n_outputs)]


class TreeEnsemble:
    """ An ensemble of decision trees.

    This object provides a common interface to many different types of models.
    """

    def __init__(self, model, data=None, data_missing=None):
        self.model_type = "internal"
        self.trees = None
        less_than_or_equal = True
        self.base_offset = 0
        self.objective = None # what we explain when explaining the loss of the model
        self.tree_output = None # what are the units of the values in the leaves of the trees
        self.dtype = np.float64 # for sklearn we need to use np.float32 to always get exact matches to their predictions
        self.data = data
        self.data_missing = data_missing
        self.fully_defined_weighting = True # does the background dataset land in every leaf (making it valid for the tree_path_dependent method)
        self.tree_limit = None # used for limiting the number of trees we use by default (like from early stopping) 

        # we use names like keras
        objective_name_map = {
            "mse": "squared_error",
            "friedman_mse": "squared_error",
            "reg:linear": "squared_error",
            "regression": "squared_error",
            "regression_l2": "squared_error",
            "mae": "absolute_error",
            "gini": "binary_crossentropy",
            "entropy": "binary_crossentropy",
            "binary:logistic": "binary_crossentropy",
            "binary_logloss": "binary_crossentropy",
            "binary": "binary_crossentropy"
        }

        tree_output_name_map = {
            "regression": "raw_value",
            "regression_l2": "squared_error",
            "reg:linear": "raw_value",
            "binary:logistic": "log_odds",
            "binary_logloss": "log_odds",
            "binary": "log_odds"
        }

        if type(model) == list and type(model[0]) == Tree:
            self.trees = model
        elif str(type(model)).endswith("sklearn.ensemble.forest.RandomForestRegressor'>"):
            self.dtype = np.float32
            scaling = 1.0 / len(model.estimators_) # output is average of trees
            self.trees = [Tree(e.tree_, scaling=scaling, data=data, data_missing=data_missing) for e in model.estimators_]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "raw_value"
        elif str(type(model)).endswith("skopt.learning.forest.RandomForestRegressor'>"):
            self.dtype = np.float32
            scaling = 1.0 / len(model.estimators_) # output is average of trees
            self.trees = [Tree(e.tree_, scaling=scaling, data=data, data_missing=data_missing) for e in model.estimators_]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "raw_value"
        elif str(type(model)).endswith("sklearn.ensemble.forest.ExtraTreesRegressor'>"):
            self.dtype = np.float32
            scaling = 1.0 / len(model.estimators_) # output is average of trees
            self.trees = [Tree(e.tree_, scaling=scaling, data=data, data_missing=data_missing) for e in model.estimators_]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "raw_value"
        elif str(type(model)).endswith("skopt.learning.forest.ExtraTreesRegressor'>"):
            self.dtype = np.float32
            scaling = 1.0 / len(model.estimators_) # output is average of trees
            self.trees = [Tree(e.tree_, scaling=scaling, data=data, data_missing=data_missing) for e in model.estimators_]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "raw_value"
        elif str(type(model)).endswith("sklearn.tree.tree.DecisionTreeRegressor'>"):
            self.dtype = np.float32
            self.trees = [Tree(model.tree_, data=data, data_missing=data_missing)]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "raw_value"
        elif str(type(model)).endswith("sklearn.tree.tree.DecisionTreeClassifier'>"):
            self.dtype = np.float32
            self.trees = [Tree(model.tree_, normalize=True, data=data, data_missing=data_missing)]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "probability"
        elif str(type(model)).endswith("sklearn.ensemble.forest.RandomForestClassifier'>"):
            self.dtype = np.float32
            scaling = 1.0 / len(model.estimators_) # output is average of trees
            self.trees = [Tree(e.tree_, normalize=True, scaling=scaling, data=data, data_missing=data_missing) for e in model.estimators_]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "probability"
        elif str(type(model)).endswith("sklearn.ensemble.forest.ExtraTreesClassifier'>"): # TODO: add unit test for this case
            self.dtype = np.float32
            scaling = 1.0 / len(model.estimators_) # output is average of trees
            self.trees = [Tree(e.tree_, normalize=True, scaling=scaling, data=data, data_missing=data_missing) for e in model.estimators_]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "probability"
        elif str(type(model)).endswith("sklearn.ensemble.gradient_boosting.GradientBoostingRegressor'>"):
            self.dtype = np.float32

            # currently we only support the mean estimator
            if str(type(model.init_)).endswith("ensemble.gradient_boosting.MeanEstimator'>"):
                self.base_offset = model.init_.mean
            else:
                assert False, "Unsupported init model type: " + str(type(model.init_))

            self.trees = [Tree(e.tree_, scaling=model.learning_rate, data=data, data_missing=data_missing) for e in model.estimators_[:,0]]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "raw_value"
        elif str(type(model)).endswith("sklearn.ensemble.gradient_boosting.GradientBoostingClassifier'>"):
            self.dtype = np.float32
            
            # currently we only support the logs odds estimator
            if str(type(model.init_)).endswith("ensemble.gradient_boosting.LogOddsEstimator'>"):
                self.base_offset = model.init_.prior
                self.tree_output = "log_odds"
            else:
                assert False, "Unsupported init model type: " + str(type(model.init_))

            self.trees = [Tree(e.tree_, scaling=model.learning_rate, data=data, data_missing=data_missing) for e in model.estimators_[:,0]]
            self.objective = objective_name_map.get(model.criterion, None)
        elif str(type(model)).endswith("xgboost.core.Booster'>"):
            assert_import("xgboost")
            self.original_model = model
            self.model_type = "xgboost"
            xgb_loader = XGBTreeModelLoader(self.original_model)
            self.trees = xgb_loader.get_trees(data=data, data_missing=data_missing)
            self.base_offset = xgb_loader.base_score
            less_than_or_equal = False
            self.objective = objective_name_map.get(xgb_loader.name_obj, None)
            self.tree_output = tree_output_name_map.get(xgb_loader.name_obj, None)
        elif str(type(model)).endswith("xgboost.sklearn.XGBClassifier'>"):
            assert_import("xgboost")
            self.dtype = np.float32
            self.model_type = "xgboost"
            self.original_model = model.get_booster()
            xgb_loader = XGBTreeModelLoader(self.original_model)
            self.trees = xgb_loader.get_trees(data=data, data_missing=data_missing)
            self.base_offset = xgb_loader.base_score
            less_than_or_equal = False
            self.objective = objective_name_map.get(xgb_loader.name_obj, None)
            self.tree_output = tree_output_name_map.get(xgb_loader.name_obj, None)
            self.tree_limit = getattr(model, "best_ntree_limit", None)
        elif str(type(model)).endswith("xgboost.sklearn.XGBRegressor'>"):
            assert_import("xgboost")
            self.original_model = model.get_booster()
            self.model_type = "xgboost"
            xgb_loader = XGBTreeModelLoader(self.original_model)
            self.trees = xgb_loader.get_trees(data=data, data_missing=data_missing)
            self.base_offset = xgb_loader.base_score
            less_than_or_equal = False
            self.objective = objective_name_map.get(xgb_loader.name_obj, None)
            self.tree_output = tree_output_name_map.get(xgb_loader.name_obj, None)
            self.tree_limit = getattr(model, "best_ntree_limit", None)
        elif str(type(model)).endswith("xgboost.sklearn.XGBRanker'>"):
            assert_import("xgboost")
            self.original_model = model.get_booster()
            self.model_type = "xgboost"
            xgb_loader = XGBTreeModelLoader(self.original_model)
            self.trees = xgb_loader.get_trees(data=data, data_missing=data_missing)
            self.base_offset = xgb_loader.base_score
            less_than_or_equal = False
            # Note: for ranker, leaving tree_output and objective as None as they
            # are not implemented in native code yet
            self.tree_limit = getattr(model, "best_ntree_limit", None)
        elif str(type(model)).endswith("lightgbm.basic.Booster'>"):
            assert_import("lightgbm")
            self.model_type = "lightgbm"
            self.original_model = model
            tree_info = self.original_model.dump_model()["tree_info"]
            try:
                self.trees = [Tree(e, data=data, data_missing=data_missing) for e in tree_info]
            except:
                self.trees = None # we get here because the cext can't handle categorical splits yet
            
            self.objective = objective_name_map.get(model.params.get("objective", "regression"), None)
            self.tree_output = tree_output_name_map.get(model.params.get("objective", "regression"), None)
            
        elif str(type(model)).endswith("lightgbm.sklearn.LGBMRegressor'>"):
            assert_import("lightgbm")
            self.model_type = "lightgbm"
            self.original_model = model.booster_
            tree_info = self.original_model.dump_model()["tree_info"]
            try:
                self.trees = [Tree(e, data=data, data_missing=data_missing) for e in tree_info]
            except:
                self.trees = None # we get here because the cext can't handle categorical splits yet
            self.objective = objective_name_map.get(model.objective, None)
            self.tree_output = tree_output_name_map.get(model.objective, None)
            if model.objective is None:
                self.objective = "squared_error"
                self.tree_output = "raw_value"
        elif str(type(model)).endswith("lightgbm.sklearn.LGBMRanker'>"):
            assert_import("lightgbm")
            self.model_type = "lightgbm"
            self.original_model = model.booster_
            tree_info = self.original_model.dump_model()["tree_info"]
            try:
                self.trees = [Tree(e, data=data, data_missing=data_missing) for e in tree_info]
            except:
                self.trees = None # we get here because the cext can't handle categorical splits yet
            # Note: for ranker, leaving tree_output and objective as None as they
            # are not implemented in native code yet
        elif str(type(model)).endswith("lightgbm.sklearn.LGBMClassifier'>"):
            assert_import("lightgbm")
            self.model_type = "lightgbm"
            self.original_model = model.booster_
            tree_info = self.original_model.dump_model()["tree_info"]
            try:
                self.trees = [Tree(e, data=data, data_missing=data_missing) for e in tree_info]
            except:
                self.trees = None # we get here because the cext can't handle categorical splits yet
            self.objective = objective_name_map.get(model.objective, None)
            self.tree_output = tree_output_name_map.get(model.objective, None)
            if model.objective is None:
                self.objective = "binary_crossentropy"
                self.tree_output = "log_odds"
        elif str(type(model)).endswith("catboost.core.CatBoostRegressor'>"):
            assert_import("catboost")
            self.model_type = "catboost"
            self.original_model = model
        elif str(type(model)).endswith("catboost.core.CatBoostClassifier'>"):
            assert_import("catboost")
            self.model_type = "catboost"
            self.original_model = model
        elif str(type(model)).endswith("catboost.core.CatBoost'>"):
            assert_import("catboost")
            self.model_type = "catboost"
            self.original_model = model
        elif str(type(model)).endswith("imblearn.ensemble._forest.BalancedRandomForestClassifier'>"):
            self.dtype = np.float32
            scaling = 1.0 / len(model.estimators_) # output is average of trees
            self.trees = [Tree(e.tree_, normalize=True, scaling=scaling, data=data, data_missing=data_missing) for e in model.estimators_]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "probability"
        else:
            raise Exception("Model type not yet supported by TreeExplainer: " + str(type(model)))
        
        # build a dense numpy version of all the tree objects
        if self.trees is not None:
            max_nodes = np.max([len(t.values) for t in self.trees])
            assert len(np.unique([t.values.shape[1] for t in self.trees])) == 1, "All trees in the ensemble must have the same output dimension!"
            ntrees = len(self.trees)
            self.n_outputs = self.trees[0].values.shape[1]

            # important to be -1 in unused sections!! This way we can tell which entries are valid.
            self.children_left = -np.ones((ntrees, max_nodes), dtype=np.int32)
            self.children_right = -np.ones((ntrees, max_nodes), dtype=np.int32)
            self.children_default = -np.ones((ntrees, max_nodes), dtype=np.int32)
            self.features = -np.ones((ntrees, max_nodes), dtype=np.int32)

            self.thresholds = np.zeros((ntrees, max_nodes), dtype=self.dtype)
            self.values = np.zeros((ntrees, max_nodes, self.trees[0].values.shape[1]), dtype=self.dtype)
            self.node_sample_weight = np.zeros((ntrees, max_nodes), dtype=self.dtype)
            
            for i in range(ntrees):
                l = len(self.trees[i].features)
                self.children_left[i,:l] = self.trees[i].children_left
                self.children_right[i,:l] = self.trees[i].children_right
                self.children_default[i,:l] = self.trees[i].children_default
                self.features[i,:l] = self.trees[i].features
                self.thresholds[i,:l] = self.trees[i].thresholds
                self.values[i,:l,:] = self.trees[i].values
                self.node_sample_weight[i,:l] = self.trees[i].node_sample_weight

                # ensure that the passed background dataset lands in every leaf
                if np.min(self.trees[i].node_sample_weight) <= 0:
                    self.fully_defined_weighting = False
            
            # If we should do <= then we nudge the thresholds to make our <= work like <
            if not less_than_or_equal:
                self.thresholds = np.nextafter(self.thresholds, -np.inf)
            
            self.num_nodes = np.array([len(t.values) for t in self.trees], dtype=np.int32)
            self.max_depth = np.max([t.max_depth for t in self.trees])

    def get_transform(self, model_output):
        """ A consistent interface to make predictions from this model.
        """
        if model_output == "margin":
            transform = "identity"
        elif model_output == "probability":
            if self.tree_output == "log_odds":
                transform = "logistic"
            elif self.tree_output == "probability":
                transform = "identity"
            else:
                raise Exception("model_output = \"probability\" is not yet supported when model.tree_output = \"" + self.tree_output + "\"!")
        elif model_output == "logloss":

            if self.objective == "squared_error":
                transform = "squared_loss"
            elif self.objective == "binary_crossentropy":
                transform = "logistic_nlogloss"
            else:
                raise Exception("model_output = \"logloss\" is not yet supported when model.objective = \"" + self.objective + "\"!")
        return transform

    def predict(self, X, y=None, output="margin", tree_limit=None):
        """ A consistent interface to make predictions from this model.

        Parameters
        ----------
        tree_limit : None (default) or int 
            Limit the number of trees used by the model. By default None means no use the limit of the
            original model, and -1 means no limit.
        """

        # see if we have a default tree_limit in place.
        if tree_limit is None:
            tree_limit = -1 if self.tree_limit is None else self.tree_limit

        # convert dataframes
        if str(type(X)).endswith("pandas.core.series.Series'>"):
            X = X.values
        elif str(type(X)).endswith("pandas.core.frame.DataFrame'>"):
            X = X.values
        flat_output = False
        if len(X.shape) == 1:
            flat_output = True
            X = X.reshape(1, X.shape[0])
        if X.dtype != self.dtype:
            X = X.astype(self.dtype)
        X_missing = np.isnan(X, dtype=np.bool)
        assert str(type(X)).endswith("'numpy.ndarray'>"), "Unknown instance type: " + str(type(X))
        assert len(X.shape) == 2, "Passed input data matrix X must have 1 or 2 dimensions!"

        if tree_limit < 0 or tree_limit > self.values.shape[0]:
            tree_limit = self.values.shape[0]

        if output == "logloss":
            assert y is not None, "Both samples and labels must be provided when explaining the loss (i.e. `explainer.shap_values(X, y)`)!"
            assert X.shape[0] == len(y), "The number of labels (%d) does not match the number of samples to explain (%d)!" % (len(y), X.shape[0])
        transform = self.get_transform(output)
        
        if True or self.model_type == "internal":
            output = np.zeros((X.shape[0], self.n_outputs))
            assert_import("cext")
            _cext.dense_tree_predict(
                self.children_left, self.children_right, self.children_default,
                self.features, self.thresholds, self.values,
                self.max_depth, tree_limit, self.base_offset, output_transform_codes[transform], 
                X, X_missing, y, output
            )

        elif self.model_type == "xgboost":
            assert_import("xgboost")
            output = self.original_model.predict(X, output_margin=True, tree_limit=tree_limit)

        # drop dimensions we don't need
        if flat_output:
            if self.n_outputs == 1:
                return output.flatten()[0]
            else:
                return output.reshape(-1, self.n_outputs)
        else:
            if self.n_outputs == 1:
                return output.flatten()
            else:
                return output


class Tree:
    """ A single decision tree.

    The primary point of this object is to parse many different tree types into a common format.
    """
    def __init__(self, tree, normalize=False, scaling=1.0, data=None, data_missing=None):
        assert_import("cext")

        if str(type(tree)).endswith("'sklearn.tree._tree.Tree'>"):
            self.children_left = tree.children_left.astype(np.int32)
            self.children_right = tree.children_right.astype(np.int32)
            self.children_default = self.children_left # missing values not supported in sklearn
            self.features = tree.feature.astype(np.int32)
            self.thresholds = tree.threshold.astype(np.float64)
            self.values = tree.value.reshape(tree.value.shape[0], tree.value.shape[1] * tree.value.shape[2])
            if normalize:
                self.values = (self.values.T / self.values.sum(1)).T
            self.values = self.values * scaling
            self.node_sample_weight = tree.weighted_n_node_samples.astype(np.float64)

        elif type(tree) == dict and 'children_left' in tree:
            self.children_left = tree["children_left"].astype(np.int32)
            self.children_right = tree["children_right"].astype(np.int32)
            self.children_default = tree["children_default"].astype(np.int32)
            self.features = tree["feature"].astype(np.int32)
            self.thresholds = tree["threshold"]
            self.values = tree["value"] * scaling
            self.node_sample_weight = tree["node_sample_weight"]

        elif type(tree) == dict and 'tree_structure' in tree:
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
        
        elif type(tree) == dict and 'nodeid' in tree:
            """ Directly create tree given the JSON dump (with stats) of a XGBoost model.
            """

            def max_id(node):
                if "children" in node:
                    return max(node["nodeid"], *[max_id(n) for n in node["children"]])
                else:
                    return node["nodeid"]
            
            m = max_id(tree) + 1
            self.children_left = -np.ones(m, dtype=np.int32)
            self.children_right = -np.ones(m, dtype=np.int32)
            self.children_default = -np.ones(m, dtype=np.int32)
            self.features = -np.ones(m, dtype=np.int32)
            self.thresholds = np.zeros(m, dtype=np.float64)
            self.values = np.zeros((m, 1), dtype=np.float64)
            self.node_sample_weight = np.empty(m, dtype=np.float64)

            def extract_data(node, tree):
                i = node["nodeid"]
                tree.node_sample_weight[i] = node["cover"]

                if "children" in node:
                    tree.children_left[i] = node["yes"]
                    tree.children_right[i] = node["no"]
                    tree.children_default[i] = node["missing"]
                    tree.features[i] = node["split"]
                    tree.thresholds[i] = node["split_condition"]

                    for n in node["children"]:
                        extract_data(n, tree)
                elif "leaf" in node:
                    tree.values[i] = node["leaf"] * scaling

            extract_data(tree, self)
    
        elif type(tree) == str:
            """ Build a tree from a text dump (with stats) of xgboost.
            """

            nodes = [t.lstrip() for t in tree[:-1].split("\n")]
            nodes_dict = {}
            for n in nodes: nodes_dict[int(n.split(":")[0])] = n.split(":")[1]
            m = max(nodes_dict.keys())+1
            children_left = -1*np.ones(m,dtype="int32")
            children_right = -1*np.ones(m,dtype="int32")
            children_default = -1*np.ones(m,dtype="int32")
            features = -2*np.ones(m,dtype="int32")
            thresholds = -1*np.ones(m,dtype="float64")
            values = 1*np.ones(m,dtype="float64")
            node_sample_weight = np.zeros(m,dtype="float64")
            values_lst = list(nodes_dict.values())
            keys_lst = list(nodes_dict.keys())
            for i in range(0,len(keys_lst)):
                value = values_lst[i]
                key = keys_lst[i]
                if ("leaf" in value):
                    # Extract values
                    val = float(value.split("leaf=")[1].split(",")[0])
                    node_sample_weight_val = float(value.split("cover=")[1])
                    # Append to lists
                    values[key] = val
                    node_sample_weight[key] = node_sample_weight_val
                else:
                    c_left = int(value.split("yes=")[1].split(",")[0])
                    c_right = int(value.split("no=")[1].split(",")[0])
                    c_default = int(value.split("missing=")[1].split(",")[0])
                    feat_thres = value.split(" ")[0]
                    if ("<" in feat_thres):
                        feature = int(feat_thres.split("<")[0][2:])
                        threshold = float(feat_thres.split("<")[1][:-1])
                    if ("=" in feat_thres):
                        feature = int(feat_thres.split("=")[0][2:])
                        threshold = float(feat_thres.split("=")[1][:-1])
                    node_sample_weight_val = float(value.split("cover=")[1].split(",")[0])
                    children_left[key] = c_left
                    children_right[key] = c_right
                    children_default[key] = c_default
                    features[key] = feature
                    thresholds[key] = threshold
                    node_sample_weight[key] = node_sample_weight_val
            
            self.children_left = children_left
            self.children_right = children_right
            self.children_default = children_default
            self.features = features
            self.thresholds = thresholds
            self.values = values[:,np.newaxis] * scaling
            self.node_sample_weight = node_sample_weight
        else:
            raise Exception("Unknown input to Tree constructor!")
        
        # Re-compute the number of samples that pass through each node if we are given data
        if data is not None and data_missing is not None:
            self.node_sample_weight[:] = 0.0
            _cext.dense_tree_update_weights(
                self.children_left, self.children_right, self.children_default, self.features,
                self.thresholds, self.values, 1, self.node_sample_weight, data, data_missing
            )
        
        # we compute the expectations to make sure they follow the SHAP logic
        self.max_depth = _cext.compute_expectations(
            self.children_left, self.children_right, self.node_sample_weight,
            self.values
        )




def get_xgboost_json(model):
    """ This gets a JSON dump of an XGBoost model while ensuring the features names are their indexes.
    """
    fnames = model.feature_names
    model.feature_names = None
    json_trees = model.get_dump(with_stats=True, dump_format="json")
    model.feature_names = fnames

    # this fixes a bug where XGBoost can return invalid JSON
    json_trees = [t.replace(": inf,", ": 1000000000000.0,") for t in json_trees]
    json_trees = [t.replace(": -inf,", ": -1000000000000.0,") for t in json_trees]

    return json_trees


class XGBTreeModelLoader(object):
    """ This loads an XGBoost model directly from a raw memory dump.

    We can't use the JSON dump because due to numerical precision issues those
    tree can actually be wrong when feature values land almost on a threshold.
    """
    def __init__(self, xgb_model):
        self.buf = xgb_model.save_raw()
        self.pos = 0
        
        # load the model parameters
        self.base_score = self.read('f')
        self.num_feature = self.read('I')
        self.num_class = self.read('i')
        self.contain_extra_attrs = self.read('i')
        self.contain_eval_metrics = self.read('i')
        self.read_arr('i', 29) # reserved
        self.name_obj_len = self.read('Q')
        self.name_obj = self.read_str(self.name_obj_len)
        self.name_gbm_len = self.read('Q')
        self.name_gbm = self.read_str(self.name_gbm_len)
        
        assert self.name_gbm == "gbtree", "Only the 'gbtree' model type is supported, not '%s'!" % self.name_gbm
        
        # load the gbtree specific parameters
        self.num_trees = self.read('i')
        self.num_roots = self.read('i')
        self.num_feature = self.read('i')
        self.pad_32bit = self.read('i')
        self.num_pbuffer_deprecated = self.read('Q')
        self.num_output_group = self.read('i')
        self.size_leaf_vector = self.read('i')
        self.read_arr('i', 32) # reserved
        
        # load each tree
        self.num_roots = np.zeros(self.num_trees, dtype=np.int32)
        self.num_nodes = np.zeros(self.num_trees, dtype=np.int32)
        self.num_deleted = np.zeros(self.num_trees, dtype=np.int32)
        self.max_depth = np.zeros(self.num_trees, dtype=np.int32)
        self.num_feature = np.zeros(self.num_trees, dtype=np.int32)
        self.size_leaf_vector = np.zeros(self.num_trees, dtype=np.int32)
        self.node_parents = []
        self.node_cleft = []
        self.node_cright = []
        self.node_sindex = []
        self.node_info = []
        self.loss_chg = []
        self.sum_hess = []
        self.base_weight = []
        self.leaf_child_cnt = []
        for i in range(self.num_trees):
            
            # load the per-tree params
            self.num_roots[i] = self.read('i')
            self.num_nodes[i] = self.read('i')
            self.num_deleted[i] = self.read('i')
            self.max_depth[i] = self.read('i')
            self.num_feature[i] = self.read('i')
            self.size_leaf_vector[i] = self.read('i')
            
            # load the nodes
            self.read_arr('i', 31) # reserved
            self.node_parents.append(np.zeros(self.num_nodes[i], dtype=np.int32))
            self.node_cleft.append(np.zeros(self.num_nodes[i], dtype=np.int32))
            self.node_cright.append(np.zeros(self.num_nodes[i], dtype=np.int32))
            self.node_sindex.append(np.zeros(self.num_nodes[i], dtype=np.uint32))
            self.node_info.append(np.zeros(self.num_nodes[i], dtype=np.float32))
            for j in range(self.num_nodes[i]):
                self.node_parents[-1][j] = self.read('i')
                self.node_cleft[-1][j] = self.read('i')
                self.node_cright[-1][j] = self.read('i')
                self.node_sindex[-1][j] = self.read('I')
                self.node_info[-1][j] = self.read('f')
#                 print("self.node_cleft[-1][%d]" % j, self.node_cleft[-1][j])
#                 print("self.node_cright[-1][%d]" % j, self.node_cright[-1][j])
#                 print("self.node_sindex[-1][%d]" % j, self.node_sindex[-1][j])
#                 print("self.node_info[-1][%d]" % j, self.node_info[-1][j])
#                 print()
            
            # load the stat nodes
            self.loss_chg.append(np.zeros(self.num_nodes[i], dtype=np.float32))
            self.sum_hess.append(np.zeros(self.num_nodes[i], dtype=np.float32))
            self.base_weight.append(np.zeros(self.num_nodes[i], dtype=np.float32))
            self.leaf_child_cnt.append(np.zeros(self.num_nodes[i], dtype=np.int))
            for j in range(self.num_nodes[i]):
                self.loss_chg[-1][j] = self.read('f')
                self.sum_hess[-1][j] = self.read('f')
                self.base_weight[-1][j] = self.read('f')
                self.leaf_child_cnt[-1][j] = self.read('i')
#                 print("self.loss_chg[-1][%d]" % j, self.loss_chg[-1][j])
#                 print("self.sum_hess[-1][%d]" % j, self.sum_hess[-1][j])
#                 print("self.base_weight[-1][%d]" % j, self.base_weight[-1][j])
#                 print("self.leaf_child_cnt[-1][%d]" % j, self.leaf_child_cnt[-1][j])
#                 print()

    def get_trees(self, data=None, data_missing=None):
        shape = (self.num_trees, self.num_nodes.max())
        self.children_default = np.zeros(shape, dtype=np.int)
        self.features = np.zeros(shape, dtype=np.int)
        self.thresholds = np.zeros(shape, dtype=np.float32)
        self.values = np.zeros((shape[0], shape[1], 1), dtype=np.float32)
        trees = []
        for i in range(self.num_trees):
            for j in range(self.num_nodes[i]):
                if np.right_shift(self.node_sindex[i][j], np.uint32(31)) != 0:
                    self.children_default[i,j] = self.node_cleft[i][j]
                else:
                    self.children_default[i,j] = self.node_cright[i][j]
                self.features[i,j] = self.node_sindex[i][j] & ((np.uint32(1) << np.uint32(31)) - np.uint32(1))
                if self.node_cleft[i][j] >= 0:
                    self.thresholds[i,j] = self.node_info[i][j]
                else:
                    self.values[i,j] = self.node_info[i][j]

            l = len(self.node_cleft[i])
            trees.append(Tree({
                "children_left": self.node_cleft[i],
                "children_right": self.node_cright[i],
                "children_default": self.children_default[i,:l],
                "feature": self.features[i,:l],
                "threshold": self.thresholds[i,:l],
                "value": self.values[i,:l],
                "node_sample_weight": self.sum_hess[i]
            }, data=data, data_missing=data_missing))
        return trees
            
    
    def read(self, dtype):
        size = struct.calcsize(dtype)
        val = struct.unpack(dtype, self.buf[self.pos:self.pos+size])[0]
        self.pos += size
        return val
    
    def read_arr(self, dtype, n_items):
        format = "%d%s" % (n_items, dtype)
        size = struct.calcsize(format)
        val = struct.unpack(format, self.buf[self.pos:self.pos+size])[0]
        self.pos += size
        return val
    
    def read_str(self, size):
        val = self.buf[self.pos:self.pos+size].decode('utf-8')
        self.pos += size
        return val
    
    def print_info(self):
        
        print("--- global parmeters ---")
        print("base_score =", self.base_score)
        print("num_feature =", self.num_feature)
        print("num_class =", self.num_class)
        print("contain_extra_attrs =", self.contain_extra_attrs)
        print("contain_eval_metrics =", self.contain_eval_metrics)
        print("name_obj_len =", self.name_obj_len)
        print("name_obj =", self.name_obj)
        print("name_gbm_len =", self.name_gbm_len)
        print("name_gbm =", self.name_gbm)
        print()
        print("--- gbtree specific parameters ---")
        print("num_trees =", self.num_trees)
        print("num_roots =", self.num_roots)
        print("num_feature =", self.num_feature)
        print("pad_32bit =", self.pad_32bit)
        print("num_pbuffer_deprecated =", self.num_pbuffer_deprecated)
        print("num_output_group =", self.num_output_group)
        print("size_leaf_vector =", self.size_leaf_vector)

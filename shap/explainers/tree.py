import numpy as np
import multiprocessing
import sys
import json
import os
from distutils.version import LooseVersion
from .explainer import Explainer

import_errors = {}

try:
    from .. import _cext
except Exception as e:
    import_errors["cext"] = "C extension was not built during install!", e

try:
    import xgboost
except Exception as e:
    import_errors["xgboost"] = "XGBoost could not be imported!", e

try:
    import lightgbm
except Exception as e:
    import_errors["lightgbm"] = "LightGBM could not be imported!", e

try:
    import catboost
except Exception as e:
    import_errors["catboost"] = "CatBoost could not be imported!", e

def assert_import(package_name):
    if package_name in import_errors:
        msg,e = import_errors[package_name]
        print(msg)
        raise e

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
        self.model = TreeEnsemble(model)
        if str(type(data)).endswith("pandas.core.frame.DataFrame'>"):
            self.data = data.values
        else:
            self.data = data
        self.data_missing = None if data is None else np.isnan(data)
        self.model_output = model_output
        self.feature_dependence = feature_dependence
        self.expected_value = None

        assert feature_dependence in feature_dependence_codes, "Invalid feature_dependence option!"

        # check for unsupported combinations of feature_dependence and model_outputs
        if feature_dependence == "tree_path_dependent":
            assert model_output == "margin", "Only margin model_output is supported for feature_dependence=\"tree_path_dependent\""
        else:   
            assert data is not None, "A background dataset must be provided unless you are using feature_dependence=\"tree_path_dependent\"!"

        if model_output != "margin":
            if self.model.model_type == "xgboost" and self.model.objective is None:
                msg = "When model_output is not \"margin\" then we need to know the model's objective. Unfortuneately " + \
                      "raw XGBoost Booster objects don't expose this information. Consider using the XGBRegressor/" + \
                      "XGBClassifier objects, or annotate the Booster object with the objective " + \
                      "you are using, for example: xgb_model.set_attr(objective=\"binary:logistic\")."
                raise Exception(msg)
            elif self.model.objective is None:
                raise Exception("Model does have a known objective! When model_output is not \"margin\" then we need to know the model's objective")

        # A bug in XGBoost fixed in v0.81 makes XGBClassifier fail to give margin outputs
        if str(type(model)).endswith("xgboost.sklearn.XGBClassifier'>") and model_output != "margin":
            assert_import("xgboost")
            assert LooseVersion(xgboost.__version__) >= LooseVersion('0.81'), \
                "A bug in XGBoost fixed in v0.81 makes XGBClassifier fail to give margin outputs! Please upgrade to XGBoost >= v0.81!"

    def shap_values(self, X, y=None, tree_limit=-1, approximate=False):
        """ Estimate the SHAP values for a set of samples.

        Parameters
        ----------
        X : numpy.array, pandas.DataFrame or catboost.Pool (for catboost)
            A matrix of samples (# samples x # features) on which to explain the model's output.

        y : numpy.array
            An array of label values for each sample. Used when explaining loss functions.

        tree_limit : int
            Limit the number of trees used by the model. By default -1 means no limit.

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

        # shortcut using the C++ version of Tree SHAP in XGBoost, LightGBM, and CatBoost
        if self.feature_dependence == "tree_path_dependent" and self.model.model_type != "internal":
            phi = None
            if self.model.model_type == "xgboost":
                assert_import("xgboost")
                if not str(type(X)).endswith("xgboost.core.DMatrix'>"):
                    X = xgboost.DMatrix(X)
                if tree_limit == -1:
                    tree_limit = 0
                phi = self.model.original_model.predict(X, ntree_limit=tree_limit, pred_contribs=True, approx_contribs=approximate)
            
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
        if X.dtype != np.float64 and X.dtype != np.float32:
            X = X.astype(np.float64)
        X_missing = np.isnan(X, dtype=np.bool)
        assert str(type(X)).endswith("'numpy.ndarray'>"), "Unknown instance type: " + str(type(X))
        assert len(X.shape) == 2, "Passed input data matrix X must have 1 or 2 dimensions!"

        if tree_limit < 0 or tree_limit > len(self.model.values.shape[0]):
            tree_limit = self.model.values.shape[0]

        if self.model_output == "margin":
            transform = "identity"
        elif self.model_output == "probability":
            if self.model.tree_output == "log_odds":
                transform = "logistic"
            elif self.model.tree_output == "probability":
                transform = "identity"
            else:
                raise Exception("model_output = \"probability\" is not supported when model.tree_output = \"" + self.model.tree_output + "\"!")
        elif self.model_output == "logloss":
            assert X.shape[0] == len(y), "Labels must be provided for all samples when explaining the loss!"

            if self.model.objective == "squared_error":
                transform = "squared_loss"
            elif self.model.objective == "binary_crossentropy":
                transform = "logistic_nlogloss"
            else:
                raise Exception("model_output = \"logloss\" is not supported when model.objective = \"" + self.model.objective + "\"!")

        # in case we couldn't get the base_offset when building the model
        # (like from xgboost raw Booster objects)
        if self.model.base_offset is None:
            if self.model.model_type == "xgboost":
                assert_import("xgboost")
                self.model.base_offset = 0
                orig_pred = self.model.original_model.predict(xgboost.DMatrix(orig_X), output_margin=True)
                self.model.base_offset = orig_pred[0] - self.model.predict(X)[0]
            else:
                raise Exception("Unable to determine the base offset of " + self.model.model_type + " models!")

        # run the core algorithm using the C extension
        assert_import("cext")
        phi = np.zeros((X.shape[0], X.shape[1]+1, self.model.n_outputs))
        if not approximate:
            _cext.dense_tree_shap(
                self.model.children_left, self.model.children_right, self.model.children_default,
                self.model.features, self.model.thresholds, self.model.values, self.model.node_sample_weight,
                self.model.max_depth, X, X_missing, y, self.data, self.data_missing, tree_limit,
                self.model.base_offset, phi, feature_dependence_codes[self.feature_dependence],
                output_transform_codes[transform], False
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
            self.expected_value = phi[0, -1, 0]
            if flat_output:
                return phi[0, :-1, 0]
            else:
                return phi[:, :-1, 0]
        else:
            self.expected_value = [phi[0, -1, i] for i in range(phi.shape[2])]
            if flat_output:
                return [phi[0, :-1, i] for i in range(self.model.n_outputs)]
            else:
                return [phi[:, :-1, i] for i in range(self.model.n_outputs)]

    def shap_interaction_values(self, X, y=None, tree_limit=-1):
        """ Estimate the SHAP interaction values for a set of samples.

        Parameters
        ----------
        X : numpy.array, pandas.DataFrame or catboost.Pool (for catboost)
            A matrix of samples (# samples x # features) on which to explain the model's output.

        y : numpy.array
            An array of label values for each sample. Used when explaining loss functions (not yet supported).

        tree_limit : int
            Limit the number of trees used by the model. By default -1 means no limit.

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
        if X.dtype != np.float64 and X.dtype != np.float32:
            X = X.astype(np.float64)
        X_missing = np.isnan(X, dtype=np.bool)
        assert str(type(X)).endswith("'numpy.ndarray'>"), "Unknown instance type: " + str(type(X))
        assert len(X.shape) == 2, "Passed input data matrix X must have 1 or 2 dimensions!"

        if tree_limit < 0 or tree_limit > len(self.model.values.shape[0]):
            tree_limit = self.model.values.shape[0]

        # run the core algorithm using the C extension
        assert_import("cext")
        phi = np.zeros((X.shape[0], X.shape[1]+1, X.shape[1]+1, self.model.n_outputs))
        _cext.dense_tree_shap(
            self.model.children_left, self.model.children_right, self.model.children_default,
            self.model.features, self.model.thresholds, self.model.values, self.model.node_sample_weight,
            self.model.max_depth, X, X_missing, y, self.data, self.data_missing, tree_limit,
            self.model.base_offset, phi, feature_dependence_codes[self.feature_dependence],
            output_transform_codes[transform], True
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

    def __init__(self, model):
        self.model_type = "internel"
        self.trees = None
        less_than_or_equal = True
        self.base_offset = 0
        self.objective = None # what we explain when explaining the loss of the model
        self.tree_output = None # what are the units of the values in the leaves of the trees

        # we use names like keras
        objective_name_map = {
            "mse": "squared_error",
            "friedman_mse": "squared_error",
            "reg:linear": "squared_error",
            "mae": "absolute_error",
            "gini": "binary_crossentropy",
            "entropy": "binary_crossentropy",
            "binary:logistic": "binary_crossentropy",
            "binary_logloss": "binary_crossentropy"
        }

        tree_output_name_map = {
            "reg:linear": "raw_value",
            "binary:logistic": "log_odds",
            "binary_logloss": "log_odds"
        }

        if type(model) == list and type(model[0]) == Tree:
            self.trees = model
        elif str(type(model)).endswith("sklearn.ensemble.forest.RandomForestRegressor'>"):
            scaling = 1.0 / len(model.estimators_) # output is average of trees
            self.trees = [Tree(e.tree_, scaling=scaling) for e in model.estimators_]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "raw_value"
        elif str(type(model)).endswith("sklearn.ensemble.forest.ExtraTreesRegressor'>"):
            scaling = 1.0 / len(model.estimators_) # output is average of trees
            self.trees = [Tree(e.tree_, scaling=scaling) for e in model.estimators_]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "raw_value"
        elif str(type(model)).endswith("sklearn.tree.tree.DecisionTreeRegressor'>"):
            self.trees = [Tree(model.tree_)]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "raw_value"
        elif str(type(model)).endswith("sklearn.tree.tree.DecisionTreeClassifier'>"):
            self.trees = [Tree(model.tree_)]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "probability"
        elif str(type(model)).endswith("sklearn.ensemble.forest.RandomForestClassifier'>"):
            scaling = 1.0 / len(model.estimators_) # output is average of trees
            self.trees = [Tree(e.tree_, normalize=True, scaling=scaling) for e in model.estimators_]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "probability"
        elif str(type(model)).endswith("sklearn.ensemble.forest.ExtraTreesClassifier'>"): # TODO: add unit test for this case
            scaling = 1.0 / len(model.estimators_) # output is average of trees
            self.trees = [Tree(e.tree_, normalize=True, scaling=scaling) for e in model.estimators_]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "probability"
        elif str(type(model)).endswith("sklearn.ensemble.gradient_boosting.GradientBoostingRegressor'>"):

            # currently we only support the mean and quantile estimators
            if str(type(model.init_)).endswith("ensemble.gradient_boosting.MeanEstimator'>"):
                self.base_offset = model.init_.mean
            elif str(type(model.init_)).endswith("ensemble.gradient_boosting.QuantileEstimator'>"):
                self.base_offset = model.init_.quantile
            else:
                assert False, "Unsupported init model type: " + str(type(model.init_))

            self.trees = [Tree(e.tree_, scaling=model.learning_rate) for e in model.estimators_[:,0]]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "raw_value"
        elif str(type(model)).endswith("sklearn.ensemble.gradient_boosting.GradientBoostingClassifier'>"):
            
            # currently we only support the logs odds estimator
            if str(type(model.init_)).endswith("ensemble.gradient_boosting.LogOddsEstimator'>"):
                self.base_offset = model.init_.prior
                self.tree_output = "log_odds"
            else:
                assert False, "Unsupported init model type: " + str(type(model.init_))

            self.trees = [Tree(e.tree_, scaling=model.learning_rate) for e in model.estimators_[:,0]]
            self.objective = objective_name_map.get(model.criterion, None)
        elif str(type(model)).endswith("xgboost.core.Booster'>"):
            assert_import("xgboost")
            self.original_model = model
            self.base_offset = None
            self.model_type = "xgboost"
            json_trees = get_xgboost_json(self.original_model)
            self.trees = [Tree(json.loads(t)) for t in json_trees]
            less_than_or_equal = False
            if model.attr("objective") is not None:
                self.objective = objective_name_map.get(model.attr("objective"), None)
                self.tree_output = tree_output_name_map.get(model.attr("objective"), None)
        elif str(type(model)).endswith("xgboost.sklearn.XGBClassifier'>"):
            assert_import("xgboost")
            self.model_type = "xgboost"
            self.original_model = model.get_booster()
            self.base_offset = None
            json_trees = get_xgboost_json(self.original_model)
            self.trees = [Tree(json.loads(t)) for t in json_trees]
            less_than_or_equal = False
            self.objective = objective_name_map.get(model.objective, None)
            self.tree_output = tree_output_name_map.get(model.objective, None)
        elif str(type(model)).endswith("xgboost.sklearn.XGBRegressor'>"):
            assert_import("xgboost")
            self.original_model = model.get_booster()
            self.model_type = "xgboost"
            self.base_offset = None
            json_trees = get_xgboost_json(self.original_model)
            self.trees = [Tree(json.loads(t)) for t in json_trees]
            less_than_or_equal = False
            self.objective = objective_name_map.get(model.objective, None)
            self.tree_output = tree_output_name_map.get(model.objective, None)
        elif str(type(model)).endswith("lightgbm.basic.Booster'>"):
            assert_import("lightgbm")
            self.model_type = "lightgbm"
            self.original_model = model
            tree_info = self.original_model.dump_model()["tree_info"]
            try:
                self.trees = [Tree(e) for e in tree_info]
            except:
                self.trees = None # we get here because the cext can't handle categorical splits yet
            self.objective = objective_name_map.get(model._Booster__name_inner_eval[0], None)
            self.tree_output = tree_output_name_map.get(model._Booster__name_inner_eval[0], None)
        elif str(type(model)).endswith("lightgbm.sklearn.LGBMRegressor'>"):
            assert_import("lightgbm")
            self.model_type = "lightgbm"
            self.original_model = model.booster_
            tree_info = self.original_model.dump_model()["tree_info"]
            try:
                self.trees = [Tree(e) for e in tree_info]
            except:
                self.trees = None # we get here because the cext can't handle categorical splits yet
            self.objective = objective_name_map.get(model.objective, None)
            self.tree_output = tree_output_name_map.get(model.objective, None)
            if self.objective is None:
                self.objective = "squared_error"
                self.tree_output = "raw_value"
        elif str(type(model)).endswith("lightgbm.sklearn.LGBMClassifier'>"):
            assert_import("lightgbm")
            self.model_type = "lightgbm"
            self.original_model = model.booster_
            tree_info = self.original_model.dump_model()["tree_info"]
            try:
                self.trees = [Tree(e) for e in tree_info]
            except:
                self.trees = None # we get here because the cext can't handle categorical splits yet
            self.objective = objective_name_map.get(model.objective, None)
            self.tree_output = tree_output_name_map.get(model.objective, None)
            if self.objective is None:
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

            self.thresholds = np.zeros((ntrees, max_nodes), dtype=np.float64)
            self.values = np.zeros((ntrees, max_nodes, self.trees[0].values.shape[1]), dtype=np.float64)
            self.node_sample_weight = np.zeros((ntrees, max_nodes), dtype=np.float64)
            
            for i in range(ntrees):
                l = len(self.trees[i].features)
                self.children_left[i,:l] = self.trees[i].children_left
                self.children_right[i,:l] = self.trees[i].children_right
                self.children_default[i,:l] = self.trees[i].children_default
                self.features[i,:l] = self.trees[i].features
                self.thresholds[i,:l] = self.trees[i].thresholds
                self.values[i,:l,:] = self.trees[i].values
                self.node_sample_weight[i,:l] = self.trees[i].node_sample_weight
            
            # If we should do <= then we nudge the thresholds to make our <= work like <
            if not less_than_or_equal:
                self.thresholds -= 1e-8
            
            self.num_nodes = np.array([len(t.values) for t in self.trees], dtype=np.int32)
            self.max_depth = np.max([t.max_depth for t in self.trees])

    def predict(self, X, y=None, output="margin", tree_limit=-1):
        """ A consistent interface to make predictions from this model.
        """

        # convert dataframes
        if str(type(X)).endswith("pandas.core.series.Series'>"):
            X = X.values
        elif str(type(X)).endswith("pandas.core.frame.DataFrame'>"):
            X = X.values
        flat_output = False
        if len(X.shape) == 1:
            flat_output = True
            X = X.reshape(1, X.shape[0])
        if X.dtype != np.float64 and X.dtype != np.float32:
            X = X.astype(np.float64)
        X_missing = np.isnan(X, dtype=np.bool)
        assert str(type(X)).endswith("'numpy.ndarray'>"), "Unknown instance type: " + str(type(X))
        assert len(X.shape) == 2, "Passed input data matrix X must have 1 or 2 dimensions!"

        if tree_limit < 0 or tree_limit > len(self.values.shape[0]):
            tree_limit = self.values.shape[0]

        transform = "identity"

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
    def __init__(self, tree, normalize=False, scaling=1.0):
        assert_import("cext")

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

            # we compute the expectations to make sure they follow the SHAP logic
            self.max_depth = _cext.compute_expectations(
                self.children_left, self.children_right, self.node_sample_weight,
                self.values
            )

        elif type(tree) == dict and 'children_left' in tree:
            self.children_left = tree["children_left"].astype(np.int32)
            self.children_right = tree["children_right"].astype(np.int32)
            self.children_default = tree["children_default"].astype(np.int32)
            self.features = tree["feature"].astype(np.int32)
            self.thresholds = tree["threshold"]
            self.values = tree["value"] * scaling
            self.node_sample_weight = tree["node_sample_weight"]

            # we compute the expectations to make sure they follow the SHAP logic
            self.max_depth = _cext.compute_expectations(
                self.children_left, self.children_right, self.node_sample_weight,
                self.values
            )

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

            self.max_depth = _cext.compute_expectations(
                self.children_left, self.children_right, self.node_sample_weight,
                self.values
            )
        
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

            # we compute the expectations to make sure they follow the SHAP logic
            self.max_depth = _cext.compute_expectations(
                self.children_left, self.children_right, self.node_sample_weight,
                self.values
            )
    
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

            self.max_depth = _cext.compute_expectations(
                self.children_left, self.children_right, self.node_sample_weight,
                self.values
            )
        else:
            raise Exception("Unknown input to Tree constructor!")


def get_xgboost_json(model):
    """ This gets a JSON dump of an XGBoost model while ensuring the features names are their indexes.
    """
    fnames = model.feature_names
    model.feature_names = None
    json_trees = model.get_dump(with_stats=True, dump_format="json")
    model.feature_names = fnames

    # this fixes a bug where XGBoost can return invalid JSON
    json_trees = [t.replace(": inf,", ": 1000000000000.0,") for t in json_trees]

    return json_trees

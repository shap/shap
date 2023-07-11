import json
import struct
import time
import warnings

import numpy as np
import pandas as pd
import scipy.special
from packaging import version

from .. import maskers
from .._explanation import Explanation
from ..utils import assert_import, record_import_error, safe_isinstance
from ..utils._exceptions import (
    ExplainerError,
    InvalidFeaturePerturbationError,
    InvalidMaskerError,
    InvalidModelError,
)
from ..utils._legacy import DenseData
from ._explainer import Explainer

warnings.formatwarning = lambda msg, *args, **kwargs: str(msg) + '\n' # ignore everything except the message

try:
    from .. import _cext
except ImportError as e:
    record_import_error("cext", "C extension was not built during install!", e)

try:
    import pyspark  # noqa
except ImportError as e:
    record_import_error("pyspark", "PySpark could not be imported!", e)

output_transform_codes = {
    "identity": 0,
    "logistic": 1,
    "logistic_nlogloss": 2,
    "squared_loss": 3,
}

feature_perturbation_codes = {
    "interventional": 0,
    "tree_path_dependent": 1,
    "global_path_dependent": 2,
}


class Tree(Explainer):
    """ Uses Tree SHAP algorithms to explain the output of ensemble tree models.

    Tree SHAP is a fast and exact method to estimate SHAP values for tree models and ensembles of trees,
    under several different possible assumptions about feature dependence. It depends on fast C++
    implementations either inside an external model package or in the local compiled C extension.
    """

    def __init__(
        self,
        model,
        data=None,
        model_output="raw",
        feature_perturbation="interventional",
        feature_names=None,
        approximate=False,
        **deprecated_options,
    ):
        """ Build a new Tree explainer for the passed model.

        Parameters
        ----------
        model : model object
            The tree based machine learning model that we want to explain. XGBoost, LightGBM, CatBoost, Pyspark
            and most tree-based scikit-learn models are supported.

        data : numpy.array or pandas.DataFrame
            The background dataset to use for integrating out features. This argument is optional when
            ``feature_perturbation="tree_path_dependent"``, since in that case we can use the number of training
            samples that went down each tree path as our background dataset (this is recorded in the ``model``
            object).

        feature_perturbation : "interventional" (default) or "tree_path_dependent" (default when data=None)
            Since SHAP values rely on conditional expectations, we need to decide how to handle correlated
            (or otherwise dependent) input features.
            The "interventional" approach breaks the dependencies between features according to the rules
            dictated by causal inference (Janzing et al. 2019). Note that the "interventional" option
            requires a background dataset ``data``, and its runtime scales linearly with the size of the
            background dataset you use. Anywhere from 100 to 1000 random background samples are good
            sizes to use.
            The "tree_path_dependent" approach is to just follow the trees and use the number of training
            examples that went down each leaf to represent the background distribution. This approach
            does not require a background dataset, and so is used by default when no background dataset
            is provided.

        model_output : "raw", "probability", "log_loss", or model method name
            What output of the model should be explained.
            If "raw", then we explain the raw output of the trees, which varies by model. For regression models,
            "raw" is the standard output. For binary classification in XGBoost, this is the log odds ratio.
            If ``model_output`` is the name of a supported prediction method on the ``model`` object, then we
            explain the output of that model method name. For example, ``model_output="predict_proba"``
            explains the result of calling ``model.predict_proba``.
            If "probability", then we explain the output of the model transformed into probability space
            (note that this means the SHAP values now sum to the probability output of the model).
            If "log_loss", then we explain the log base e of the model loss function, so that the SHAP values
            sum up to the log loss of the model for each sample. This is helpful for breaking down model
            performance by feature.
            Currently the "probability" and "log_loss" options are only supported when
            ``feature_perturbation="interventional"``.

        Examples
        --------
        See `Tree explainer examples <https://shap.readthedocs.io/en/latest/api_examples/explainers/Tree.html>`_
        """
        if feature_names is not None:
            self.data_feature_names = feature_names
        elif safe_isinstance(data, "pandas.core.frame.DataFrame"):
            self.data_feature_names = list(data.columns)

        masker = data
        super().__init__(model, masker, feature_names=feature_names)

        if type(self.masker) is maskers.Independent:
            data = self.masker.data
        elif masker is not None:
            raise InvalidMaskerError("Unsupported masker type: %s!" % str(type(self.masker)))

        if getattr(self.masker, "clustering", None) is not None:
            raise ExplainerError("TreeExplainer does not support clustered data inputs! Please use shap.Explainer or pass an unclustered masker!")

        # check for deprecated options
        if model_output == "margin":
            warnings.warn("model_output = \"margin\" has been renamed to model_output = \"raw\"")
            model_output = "raw"
        if model_output == "logloss":
            warnings.warn("model_output = \"logloss\" has been renamed to model_output = \"log_loss\"")
            model_output = "log_loss"
        if "feature_dependence" in deprecated_options:
            dep_val = deprecated_options["feature_dependence"]
            if dep_val == "independent" and feature_perturbation == "interventional":
                warnings.warn("feature_dependence = \"independent\" has been renamed to feature_perturbation" \
                    " = \"interventional\"! See GitHub issue #882.")
            elif feature_perturbation != "interventional":
                warnings.warn("feature_dependence = \"independent\" has been renamed to feature_perturbation" \
                    " = \"interventional\", you can't supply both options! See GitHub issue #882.")
            if dep_val == "tree_path_dependent" and feature_perturbation == "interventional":
                raise ValueError("The feature_dependence option has been renamed to feature_perturbation! " \
                    "Please update the option name before calling TreeExplainer. See GitHub issue #882.")
        if feature_perturbation == "independent":
            raise InvalidFeaturePerturbationError("feature_perturbation = \"independent\" is not a valid option value, please use " \
                "feature_perturbation = \"interventional\" instead. See GitHub issue #882.")

        if safe_isinstance(data, "pandas.core.frame.DataFrame"):
            self.data = data.values
        elif isinstance(data, DenseData):
            self.data = data.data
        else:
            self.data = data
        if self.data is None:
            feature_perturbation = "tree_path_dependent"
            #warnings.warn("Setting feature_perturbation = \"tree_path_dependent\" because no background data was given.")
        elif feature_perturbation == "interventional" and self.data.shape[0] > 1000:
                warnings.warn("Passing "+str(self.data.shape[0]) + " background samples may lead to slow runtimes. Consider "
                    "using shap.sample(data, 100) to create a smaller background data set.")
        self.data_missing = None if self.data is None else pd.isna(self.data)
        self.feature_perturbation = feature_perturbation
        self.expected_value = None
        self.model = TreeEnsemble(model, self.data, self.data_missing, model_output)
        self.model_output = model_output
        #self.model_output = self.model.model_output # this allows the TreeEnsemble to translate model outputs types by how it loads the model

        self.approximate = approximate

        if feature_perturbation not in feature_perturbation_codes:
            raise InvalidFeaturePerturbationError("Invalid feature_perturbation option!")

        # check for unsupported combinations of feature_perturbation and model_outputs
        if feature_perturbation == "tree_path_dependent":
            if self.model.model_output != "raw":
                raise ValueError("Only model_output=\"raw\" is supported for feature_perturbation=\"tree_path_dependent\"")
        elif data is None:
            raise ValueError("A background dataset must be provided unless you are using feature_perturbation=\"tree_path_dependent\"!")

        if self.model.model_output != "raw":
            if self.model.objective is None and self.model.tree_output is None:
                raise Exception("Model does not have a known objective or output type! When model_output is " \
                                "not \"raw\" then we need to know the model's objective or link function.")

        # A bug in XGBoost fixed in v0.81 makes XGBClassifier fail to give margin outputs
        if safe_isinstance(model, "xgboost.sklearn.XGBClassifier") and self.model.model_output != "raw":
            import xgboost
            if version.parse(xgboost.__version__) < version.parse('0.81'):
                raise RuntimeError("A bug in XGBoost fixed in v0.81 makes XGBClassifier fail to give margin outputs! Please upgrade to XGBoost >= v0.81!")

        # compute the expected value if we have a parsed tree for the cext
        if self.model.model_output == "log_loss":
            self.expected_value = self.__dynamic_expected_value
        elif data is not None:
            try:
                self.expected_value = self.model.predict(self.data).mean(0)
            except ValueError:
                raise ExplainerError("Currently TreeExplainer can only handle models with categorical splits when " \
                                "feature_perturbation=\"tree_path_dependent\" and no background data is passed. Please try again using " \
                                "shap.TreeExplainer(model, feature_perturbation=\"tree_path_dependent\").")
            if hasattr(self.expected_value, '__len__') and len(self.expected_value) == 1:
                self.expected_value = self.expected_value[0]
        elif hasattr(self.model, "node_sample_weight"):
            self.expected_value = self.model.values[:,0].sum(0)
            if self.expected_value.size == 1:
                self.expected_value = self.expected_value[0]
            self.expected_value += self.model.base_offset
            if self.model.model_output != "raw":
                self.expected_value = None # we don't handle transforms in this case right now...

        # if our output format requires binary classification to be represented as two outputs then we do that here
        if self.model.model_output == "probability_doubled" and self.expected_value is not None:
            self.expected_value = [1 - self.expected_value, self.expected_value]

    def __dynamic_expected_value(self, y):
        """ This computes the expected value conditioned on the given label value.
        """

        return self.model.predict(self.data, np.ones(self.data.shape[0]) * y).mean(0)

    def __call__(self, X, y=None, interactions=False, check_additivity=True):

        start_time = time.time()

        if safe_isinstance(X, "pandas.core.frame.DataFrame"):
            feature_names = list(X.columns)
        else:
            feature_names = getattr(self, "data_feature_names", None)

        if not interactions:
            v = self.shap_values(X, y=y, from_call=True, check_additivity=check_additivity, approximate=self.approximate)
            if type(v) is list:
                v = np.stack(v, axis=-1)  # put outputs at the end
        else:
            assert not self.approximate, "Approximate computation not yet supported for interaction effects!"
            v = self.shap_interaction_values(X)

        # the Explanation object expects an expected value for each row
        if hasattr(self.expected_value, "__len__"):
            ev_tiled = np.tile(self.expected_value, (v.shape[0], 1))
        else:
            ev_tiled = np.tile(self.expected_value, v.shape[0])

        # cf. GH issue dsgibbons#66, this conversion to numpy array should be done AFTER
        # calculation of shap values
        if safe_isinstance(X, "pandas.core.frame.DataFrame"):
            X = X.values

        return Explanation(
            v,
            base_values=ev_tiled,
            data=X,
            feature_names=feature_names,
            compute_time=time.time() - start_time,
        )

    def _validate_inputs(self, X, y, tree_limit, check_additivity):
        # see if we have a default tree_limit in place.
        if tree_limit is None:
            tree_limit = -1 if self.model.tree_limit is None else self.model.tree_limit

        if tree_limit < 0 or tree_limit > self.model.values.shape[0]:
            tree_limit = self.model.values.shape[0]
        # convert dataframes
        if safe_isinstance(X, "pandas.core.series.Series"):
            X = X.values
        elif safe_isinstance(X, "pandas.core.frame.DataFrame"):
            X = X.values
        flat_output = False
        if len(X.shape) == 1:
            flat_output = True
            X = X.reshape(1, X.shape[0])
        if X.dtype != self.model.input_dtype:
            X = X.astype(self.model.input_dtype)
        X_missing = np.isnan(X, dtype=bool)
        assert isinstance(X, np.ndarray), "Unknown instance type: " + str(type(X))
        assert len(X.shape) == 2, "Passed input data matrix X must have 1 or 2 dimensions!"

        if self.model.model_output == "log_loss":
            assert y is not None, "Both samples and labels must be provided when model_output = " \
                                  "\"log_loss\" (i.e. `explainer.shap_values(X, y)`)!"
            assert X.shape[0] == len(
                y), "The number of labels (%d) does not match the number of samples to explain (" \
                    "%d)!" % (
                        len(y), X.shape[0])

        if self.feature_perturbation == "tree_path_dependent":
            assert self.model.fully_defined_weighting, "The background dataset you provided does " \
                                                       "not cover all the leaves in the model, " \
                                                       "so TreeExplainer cannot run with the " \
                                                       "feature_perturbation=\"tree_path_dependent\" option! " \
                                                       "Try providing a larger background " \
                                                       "dataset, no background dataset, or using " \
                                                       "feature_perturbation=\"interventional\"."

        if check_additivity and self.model.model_type == "pyspark":
            warnings.warn(
                "check_additivity requires us to run predictions which is not supported with "
                "spark, "
                "ignoring."
                " Set check_additivity=False to remove this warning")
            check_additivity = False

        return X, y, X_missing, flat_output, tree_limit, check_additivity

    def shap_values(self, X, y=None, tree_limit=None, approximate=False, check_additivity=True, from_call=False):
        """ Estimate the SHAP values for a set of samples.

        Parameters
        ----------
        X : numpy.array, pandas.DataFrame or catboost.Pool (for catboost)
            A matrix of samples (# samples x # features) on which to explain the model's output.

        y : numpy.array
            An array of label values for each sample. Used when explaining loss functions.

        tree_limit : None (default) or int
            Limit the number of trees used by the model. By default, the limit of the original model
            is used (``None``). ``-1`` means no limit.

        approximate : bool
            Run fast, but only roughly approximate the Tree SHAP values. This runs a method
            previously proposed by Saabas which only considers a single feature ordering. Take care
            since this does not have the consistency guarantees of Shapley values and places too
            much weight on lower splits in the tree.

        check_additivity : bool
            Run a validation check that the sum of the SHAP values equals the output of the model. This
            check takes only a small amount of time, and will catch potential unforeseen errors.
            Note that this check only runs right now when explaining the margin of the model.

        Returns
        -------
        array or list
            For models with a single output, this returns a matrix of SHAP values
            (# samples x # features). Each row sums to the difference between the model output for that
            sample and the expected value of the model output (which is stored in the ``expected_value``
            attribute of the explainer when it is constant). For models with vector outputs, this returns
            a list of such matrices, one for each output.
        """
        # see if we have a default tree_limit in place.
        if tree_limit is None:
            tree_limit = -1 if self.model.tree_limit is None else self.model.tree_limit

        # shortcut using the C++ version of Tree SHAP in XGBoost, LightGBM, and CatBoost
        if self.feature_perturbation == "tree_path_dependent" and self.model.model_type != "internal" and self.data is None:
            model_output_vals = None
            phi = None
            if self.model.model_type == "xgboost":
                import xgboost
                if not isinstance(X, xgboost.core.DMatrix):
                    X = xgboost.DMatrix(X)
                if tree_limit == -1:
                    tree_limit = 0
                try:
                    phi = self.model.original_model.predict(
                        X, iteration_range=(0, tree_limit), pred_contribs=True,
                        approx_contribs=approximate, validate_features=False
                    )
                except ValueError as e:
                    emsg = (
                        "This reshape error is often caused by passing a bad data matrix to SHAP. "
                        "See https://github.com/slundberg/shap/issues/580."
                    )
                    raise ValueError(emsg) from e

                if check_additivity and self.model.model_output == "raw":
                    xgb_tree_limit = tree_limit // self.model.num_stacked_models
                    model_output_vals = self.model.original_model.predict(
                        X, iteration_range=(0, xgb_tree_limit), output_margin=True,
                        validate_features=False
                    )

            elif self.model.model_type == "lightgbm":
                assert not approximate, "approximate=True is not supported for LightGBM models!"
                phi = self.model.original_model.predict(X, num_iteration=tree_limit, pred_contrib=True)
                # Note: the data must be joined on the last axis
                if self.model.original_model.params['objective'] == 'binary':
                    if not from_call:
                        warnings.warn('LightGBM binary classifier with TreeExplainer shap values output has changed to a list of ndarray')
                    phi = np.concatenate((0-phi, phi), axis=-1)
                if phi.shape[1] != X.shape[1] + 1:
                    try:
                        phi = phi.reshape(X.shape[0], phi.shape[1]//(X.shape[1]+1), X.shape[1]+1)
                    except ValueError as e:
                        emsg = (
                            "This reshape error is often caused by passing a bad data matrix to SHAP. "
                            "See https://github.com/slundberg/shap/issues/580."
                        )
                        raise ValueError(emsg) from e

            elif self.model.model_type == "catboost": # thanks to the CatBoost team for implementing this...
                assert not approximate, "approximate=True is not supported for CatBoost models!"
                assert tree_limit == -1, "tree_limit is not yet supported for CatBoost models!"
                import catboost
                if type(X) != catboost.Pool:
                    X = catboost.Pool(X, cat_features=self.model.cat_feature_indices)
                phi = self.model.original_model.get_feature_importance(data=X, fstr_type='ShapValues')

            # note we pull off the last column and keep it as our expected_value
            if phi is not None:
                if len(phi.shape) == 3:
                    self.expected_value = [phi[0, i, -1] for i in range(phi.shape[1])]
                    out = [phi[:, i, :-1] for i in range(phi.shape[1])]
                else:
                    self.expected_value = phi[0, -1]
                    out = phi[:, :-1]

                if check_additivity and model_output_vals is not None:
                    self.assert_additivity(out, model_output_vals)

                return out

        X, y, X_missing, flat_output, tree_limit, check_additivity = self._validate_inputs(
            X, y, tree_limit, check_additivity
        )
        transform = self.model.get_transform()

        # run the core algorithm using the C extension
        assert_import("cext")
        phi = np.zeros((X.shape[0], X.shape[1]+1, self.model.num_outputs))
        if not approximate:
            _cext.dense_tree_shap(
                self.model.children_left, self.model.children_right, self.model.children_default,
                self.model.features, self.model.thresholds, self.model.values, self.model.node_sample_weight,
                self.model.max_depth, X, X_missing, y, self.data, self.data_missing, tree_limit,
                self.model.base_offset, phi, feature_perturbation_codes[self.feature_perturbation],
                output_transform_codes[transform], False
            )
        else:
            _cext.dense_tree_saabas(
                self.model.children_left, self.model.children_right, self.model.children_default,
                self.model.features, self.model.thresholds, self.model.values,
                self.model.max_depth, tree_limit, self.model.base_offset, output_transform_codes[transform],
                X, X_missing, y, phi
            )

        out = self._get_shap_output(phi, flat_output)
        if check_additivity and self.model.model_output == "raw":
            self.assert_additivity(out, self.model.predict(X))

        return out

    def _get_shap_output(self, phi, flat_output):
        """Pull off the last column of ``phi`` and keep it as our expected_value."""
        if self.model.num_outputs == 1:
            if self.expected_value is None and self.model.model_output != "log_loss":
                self.expected_value = phi[0, -1, 0]
            if flat_output:
                out = phi[0, :-1, 0]
            else:
                out = phi[:, :-1, 0]
        else:
            if self.expected_value is None and self.model.model_output != "log_loss":
                self.expected_value = [phi[0, -1, i] for i in range(phi.shape[2])]
            if flat_output:
                out = [phi[0, :-1, i] for i in range(self.model.num_outputs)]
            else:
                out = [phi[:, :-1, i] for i in range(self.model.num_outputs)]

        # if our output format requires binary classification to be represented as two outputs then we do that here
        if self.model.model_output == "probability_doubled":
            out = [-out, out]
        return out

    def shap_interaction_values(self, X, y=None, tree_limit=None):
        """ Estimate the SHAP interaction values for a set of samples.

        Parameters
        ----------
        X : numpy.array, pandas.DataFrame or catboost.Pool (for catboost)
            A matrix of samples (# samples x # features) on which to explain the model's output.

        y : numpy.array
            An array of label values for each sample. Used when explaining loss functions (not yet supported).

        tree_limit : None (default) or int
            Limit the number of trees used by the model. By default, the limit of the original model
            is used (``None``). ``-1`` means no limit.

        Returns
        -------
        array or list
            For models with a single output, this returns a tensor of SHAP values
            (# samples x # features x # features). The matrix (# features x # features) for each sample sums
            to the difference between the model output for that sample and the expected value of the model output
            (which is stored in the ``expected_value`` attribute of the explainer). Each row of this matrix sums to the
            SHAP value for that feature for that sample. The diagonal entries of the matrix represent the
            "main effect" of that feature on the prediction. The symmetric off-diagonal entries represent the
            interaction effects between all pairs of features for that sample.
            For models with vector outputs, this returns a list of tensors, one for each output.
        """

        assert self.model.model_output == "raw", "Only model_output = \"raw\" is supported for SHAP interaction values right now!"
        #assert self.feature_perturbation == "tree_path_dependent", "Only feature_perturbation = \"tree_path_dependent\" is supported for SHAP interaction values right now!"
        transform = "identity"

        # see if we have a default tree_limit in place.
        if tree_limit is None:
            tree_limit = -1 if self.model.tree_limit is None else self.model.tree_limit

        # shortcut using the C++ version of Tree SHAP in XGBoost
        if (
            self.model.model_type == "xgboost"
            and self.feature_perturbation == "tree_path_dependent"
        ):
            import xgboost
            if not isinstance(X, xgboost.core.DMatrix):
                X = xgboost.DMatrix(X)
            if tree_limit == -1:
                tree_limit = 0
            xgb_tree_limit = tree_limit // self.model.num_stacked_models
            phi = self.model.original_model.predict(X, iteration_range=(0, xgb_tree_limit), pred_interactions=True, validate_features=False)

            # note we pull off the last column and keep it as our expected_value
            if len(phi.shape) == 4:
                self.expected_value = [phi[0, i, -1, -1] for i in range(phi.shape[1])]
                return [phi[:, i, :-1, :-1] for i in range(phi.shape[1])]
            else:
                self.expected_value = phi[0, -1, -1]
                return phi[:, :-1, :-1]

        X, y, X_missing, flat_output, tree_limit, _ = self._validate_inputs(X, y, tree_limit, False)
        # run the core algorithm using the C extension
        assert_import("cext")
        phi = np.zeros((X.shape[0], X.shape[1]+1, X.shape[1]+1, self.model.num_outputs))
        _cext.dense_tree_shap(
            self.model.children_left, self.model.children_right, self.model.children_default,
            self.model.features, self.model.thresholds, self.model.values, self.model.node_sample_weight,
            self.model.max_depth, X, X_missing, y, self.data, self.data_missing, tree_limit,
            self.model.base_offset, phi, feature_perturbation_codes[self.feature_perturbation],
            output_transform_codes[transform], True
        )

        return self._get_shap_interactions_output(phi,flat_output)

    def _get_shap_interactions_output(self, phi, flat_output):
        """Pull off the last column and keep it as our expected_value"""
        if self.model.num_outputs == 1:
            self.expected_value = phi[0, -1, -1, 0]
            if flat_output:
                out = phi[0, :-1, :-1, 0]
            else:
                out = phi[:, :-1, :-1, 0]
        else:
            self.expected_value = [phi[0, -1, -1, i] for i in range(phi.shape[3])]
            if flat_output:
                out = [phi[0, :-1, :-1, i] for i in range(self.model.num_outputs)]
            else:
                out = [phi[:, :-1, :-1, i] for i in range(self.model.num_outputs)]
        return out

    def assert_additivity(self, phi, model_output):

        def check_sum(sum_val, model_output):
            diff = np.abs(sum_val - model_output)
            if np.max(diff / (np.abs(sum_val) + 1e-2)) > 1e-2:
                ind = np.argmax(diff)
                err_msg = "Additivity check failed in TreeExplainer! Please ensure the data matrix you passed to the " \
                          "explainer is the same shape that the model was trained on. If your data shape is correct " \
                          "then please report this on GitHub."
                if self.feature_perturbation != "interventional":
                    err_msg += " Consider retrying with the feature_perturbation='interventional' option."
                err_msg += " This check failed because for one of the samples the sum of the SHAP values" \
                           " was {:f}, while the model output was {:f}. If this difference is acceptable" \
                           " you can set check_additivity=False to disable this check.".format(sum_val[ind], model_output[ind])
                raise ExplainerError(err_msg)

        if type(phi) is list:
            for i in range(len(phi)):
                check_sum(self.expected_value[i] + phi[i].sum(-1), model_output[:,i])
        else:
            check_sum(self.expected_value + phi.sum(-1), model_output)

    @staticmethod
    def supports_model_with_masker(model, masker):
        """ Determines if this explainer can handle the given model.

        This is an abstract static method meant to be implemented by each subclass.
        """

        if not isinstance(masker, (maskers.Independent)) and masker is not None:
            return False

        try:
            TreeEnsemble(model)
        except Exception:
            return False
        return True


class TreeEnsemble:
    """ An ensemble of decision trees.

    This object provides a common interface to many different types of models.
    """

    def __init__(self, model, data=None, data_missing=None, model_output=None):
        self.model_type = "internal"
        self.trees = None
        self.base_offset = 0
        self.model_output = model_output
        self.objective = None # what we explain when explaining the loss of the model
        self.tree_output = None # what are the units of the values in the leaves of the trees
        self.internal_dtype = np.float64
        self.input_dtype = np.float64 # for sklearn we need to use np.float32 to always get exact matches to their predictions
        self.data = data
        self.data_missing = data_missing
        self.fully_defined_weighting = True # does the background dataset land in every leaf (making it valid for the tree_path_dependent method)
        self.tree_limit = None # used for limiting the number of trees we use by default (like from early stopping)
        self.num_stacked_models = 1 # If this is greater than 1 it means we have multiple stacked models with the same number of trees in each model (XGBoost multi-output style)
        self.cat_feature_indices = None # If this is set it tells us which features are treated categorically

        # we use names like keras
        objective_name_map = {
            "mse": "squared_error",
            "variance": "squared_error",
            "friedman_mse": "squared_error",
            "reg:linear": "squared_error",
            "reg:squarederror": "squared_error",
            "regression": "squared_error",
            "regression_l2": "squared_error",
            "mae": "absolute_error",
            "gini": "binary_crossentropy",
            "entropy": "binary_crossentropy",
            "reg:logistic": "binary_crossentropy",
            "binary:logistic": "binary_crossentropy",
            "binary_logloss": "binary_crossentropy",
            "binary": "binary_crossentropy",
        }

        tree_output_name_map = {
            "regression": "raw_value",
            "regression_l2": "squared_error",
            "reg:linear": "raw_value",
            "reg:squarederror": "raw_value",
            "reg:logistic": "log_odds",
            "binary:logistic": "log_odds",
            "binary_logloss": "log_odds",
            "binary": "log_odds",
        }

        if type(model) is dict and "trees" in model:
            # This allows a dictionary to be passed that represents the model.
            # this dictionary has several numerical parameters and also a list of trees
            # where each tree is a dictionary describing that tree
            if "internal_dtype" in model:
                self.internal_dtype = model["internal_dtype"]
            if "input_dtype" in model:
                self.input_dtype = model["input_dtype"]
            if "objective" in model:
                self.objective = model["objective"]
            if "tree_output" in model:
                self.tree_output = model["tree_output"]
            if "base_offset" in model:
                self.base_offset = model["base_offset"]
            self.trees = [SingleTree(t, data=data, data_missing=data_missing) for t in model["trees"]]
        elif type(model) is list and type(model[0]) == SingleTree: # old-style direct-load format
            self.trees = model
        elif safe_isinstance(
            model,
            [
                "sklearn.ensemble.RandomForestRegressor",
                "sklearn.ensemble.forest.RandomForestRegressor",
                "econml.grf._base_grf.BaseGRF",
            ],
        ):
            assert hasattr(model, "estimators_"), "Model has no `estimators_`! Have you called `model.fit`?"
            self.internal_dtype = model.estimators_[0].tree_.value.dtype.type
            self.input_dtype = np.float32
            scaling = 1.0 / len(model.estimators_) # output is average of trees
            self.trees = [SingleTree(e.tree_, scaling=scaling, data=data, data_missing=data_missing) for e in model.estimators_]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "raw_value"
        elif safe_isinstance(
            model,
            [
                "sklearn.ensemble.IsolationForest",
                "sklearn.ensemble._iforest.IsolationForest",
            ],
        ):
            self.dtype = np.float32
            scaling = 1.0 / len(model.estimators_) # output is average of trees
            self.trees = [IsoTree(e.tree_, f, scaling=scaling, data=data, data_missing=data_missing) for e, f in zip(model.estimators_, model.estimators_features_)]
            self.tree_output = "raw_value"
        elif safe_isinstance(model, ["pyod.models.iforest.IForest"]):
            self.dtype = np.float32
            scaling = 1.0 / len(model.estimators_) # output is average of trees
            self.trees = [IsoTree(e.tree_, f, scaling=scaling, data=data, data_missing=data_missing) for e, f in zip(model.detector_.estimators_, model.detector_.estimators_features_)]
            self.tree_output = "raw_value"
        elif safe_isinstance(model, "skopt.learning.forest.RandomForestRegressor"):
            assert hasattr(model, "estimators_"), "Model has no `estimators_`! Have you called `model.fit`?"
            self.internal_dtype = model.estimators_[0].tree_.value.dtype.type
            self.input_dtype = np.float32
            scaling = 1.0 / len(model.estimators_) # output is average of trees
            self.trees = [SingleTree(e.tree_, scaling=scaling, data=data, data_missing=data_missing) for e in model.estimators_]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "raw_value"
        elif safe_isinstance(
            model,
            [
                "sklearn.ensemble.ExtraTreesRegressor",
                "sklearn.ensemble.forest.ExtraTreesRegressor",
            ],
        ):
            assert hasattr(model, "estimators_"), "Model has no `estimators_`! Have you called `model.fit`?"
            self.internal_dtype = model.estimators_[0].tree_.value.dtype.type
            self.input_dtype = np.float32
            scaling = 1.0 / len(model.estimators_) # output is average of trees
            self.trees = [SingleTree(e.tree_, scaling=scaling, data=data, data_missing=data_missing) for e in model.estimators_]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "raw_value"
        elif safe_isinstance(model, "skopt.learning.forest.ExtraTreesRegressor"):
            assert hasattr(model, "estimators_"), "Model has no `estimators_`! Have you called `model.fit`?"
            self.internal_dtype = model.estimators_[0].tree_.value.dtype.type
            self.input_dtype = np.float32
            scaling = 1.0 / len(model.estimators_) # output is average of trees
            self.trees = [SingleTree(e.tree_, scaling=scaling, data=data, data_missing=data_missing) for e in model.estimators_]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "raw_value"
        elif safe_isinstance(
            model,
            [
                "sklearn.tree.DecisionTreeRegressor",
                "sklearn.tree.tree.DecisionTreeRegressor",
                "econml.grf._base_grftree.GRFTree",
            ],
        ):
            self.internal_dtype = model.tree_.value.dtype.type
            self.input_dtype = np.float32
            self.trees = [SingleTree(model.tree_, data=data, data_missing=data_missing)]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "raw_value"
        elif safe_isinstance(
            model,
            [
                "sklearn.tree.DecisionTreeClassifier",
                "sklearn.tree.tree.DecisionTreeClassifier",
            ],
        ):
            self.internal_dtype = model.tree_.value.dtype.type
            self.input_dtype = np.float32
            self.trees = [SingleTree(model.tree_, normalize=True, data=data, data_missing=data_missing)]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "probability"
        elif safe_isinstance(
            model,
            [
                "sklearn.ensemble.RandomForestClassifier",
                "sklearn.ensemble.forest.RandomForestClassifier",
            ],
        ):
            assert hasattr(model, "estimators_"), "Model has no `estimators_`! Have you called `model.fit`?"
            self.internal_dtype = model.estimators_[0].tree_.value.dtype.type
            self.input_dtype = np.float32
            scaling = 1.0 / len(model.estimators_) # output is average of trees
            self.trees = [SingleTree(e.tree_, normalize=True, scaling=scaling, data=data, data_missing=data_missing) for e in model.estimators_]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "probability"
        elif safe_isinstance(
            model,
            [
                "sklearn.ensemble.ExtraTreesClassifier",
                "sklearn.ensemble.forest.ExtraTreesClassifier",
            ],
        ): # TODO: add unit test for this case
            assert hasattr(model, "estimators_"), "Model has no `estimators_`! Have you called `model.fit`?"
            self.internal_dtype = model.estimators_[0].tree_.value.dtype.type
            self.input_dtype = np.float32
            scaling = 1.0 / len(model.estimators_) # output is average of trees
            self.trees = [SingleTree(e.tree_, normalize=True, scaling=scaling, data=data, data_missing=data_missing) for e in model.estimators_]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "probability"
        elif safe_isinstance(
            model,
            [
                "sklearn.ensemble.GradientBoostingRegressor",
                "sklearn.ensemble.gradient_boosting.GradientBoostingRegressor",
            ],
        ):
            self.input_dtype = np.float32

            # currently we only support the mean and quantile estimators
            if safe_isinstance(
                model.init_,
                [
                    "sklearn.ensemble.MeanEstimator",
                    "sklearn.ensemble.gradient_boosting.MeanEstimator",
                ],
            ):
                self.base_offset = model.init_.mean
            elif safe_isinstance(
                model.init_,
                [
                    "sklearn.ensemble.QuantileEstimator",
                    "sklearn.ensemble.gradient_boosting.QuantileEstimator",
                ],
            ):
                self.base_offset = model.init_.quantile
            elif safe_isinstance(model.init_, "sklearn.dummy.DummyRegressor"):
                self.base_offset = model.init_.constant_[0]
            else:
                emsg = f"Unsupported init model type: {type(model.init_)}"
                raise InvalidModelError(emsg)

            self.trees = [SingleTree(e.tree_, scaling=model.learning_rate, data=data, data_missing=data_missing) for e in model.estimators_[:,0]]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "raw_value"
        elif safe_isinstance(model, ["sklearn.ensemble.HistGradientBoostingRegressor"]):
            # cf. GH #1028 for implementation notes
            import sklearn
            if self.model_output == "predict":
                self.model_output = "raw"
            self.input_dtype = sklearn.ensemble._hist_gradient_boosting.common.X_DTYPE
            self.base_offset = model._baseline_prediction
            self.trees = []
            for p in model._predictors:
                nodes = p[0].nodes
                # each node has values: ('value', 'count', 'feature_idx', 'threshold', 'missing_go_to_left', 'left', 'right', 'gain', 'depth', 'is_leaf', 'bin_threshold')
                tree = {
                    "children_left": np.array([-1 if n[9] else n[5] for n in nodes]),
                    "children_right": np.array([-1 if n[9] else n[6] for n in nodes]),
                    "children_default": np.array([-1 if n[9] else (n[5] if n[4] else n[6]) for n in nodes]),
                    "features": np.array([-2 if n[9] else n[2] for n in nodes]),
                    "thresholds": np.array([n[3] for n in nodes], dtype=np.float64),
                    "values": np.array([[n[0]] for n in nodes], dtype=np.float64),
                    "node_sample_weight": np.array([n[1] for n in nodes], dtype=np.float64),
                }
                self.trees.append(SingleTree(tree, data=data, data_missing=data_missing))
            self.objective = objective_name_map.get(model.loss, None)
            self.tree_output = "raw_value"
        elif safe_isinstance(model, ["sklearn.ensemble.HistGradientBoostingClassifier"]):
            # cf. GH #1028 for implementation notes
            import sklearn
            self.base_offset = model._baseline_prediction
            has_len = hasattr(self.base_offset, "__len__")
            # Note for newer sklearn versions, the base_offset is an array even for binary classification
            if has_len and self.base_offset.shape == (1, 1):
                self.base_offset = self.base_offset[0, 0]
                has_len = False
            if has_len and self.model_output != "raw":
                emsg = (
                    "Multi-output HistGradientBoostingClassifier models are not yet supported unless "
                    "model_output=\"raw\". See GitHub issue #1028."
                )
                raise NotImplementedError(emsg)
            self.input_dtype = sklearn.ensemble._hist_gradient_boosting.common.X_DTYPE
            self.num_stacked_models = len(model._predictors[0])
            if self.model_output == "predict_proba":
                if self.num_stacked_models == 1:
                    self.model_output = "probability_doubled"  # with predict_proba we need to double the outputs to match
                else:
                    self.model_output = "probability"
            self.trees = []
            for p in model._predictors:
                for i in range(self.num_stacked_models):
                    nodes = p[i].nodes
                    # each node has values: ('value', 'count', 'feature_idx', 'threshold', 'missing_go_to_left', 'left', 'right', 'gain', 'depth', 'is_leaf', 'bin_threshold')
                    tree = {
                        "children_left": np.array([-1 if n[9] else n[5] for n in nodes]),
                        "children_right": np.array([-1 if n[9] else n[6] for n in nodes]),
                        "children_default": np.array([-1 if n[9] else (n[5] if n[4] else n[6]) for n in nodes]),
                        "features": np.array([-2 if n[9] else n[2] for n in nodes]),
                        "thresholds": np.array([n[3] for n in nodes], dtype=np.float64),
                        "values": np.array([[n[0]] for n in nodes], dtype=np.float64),
                        "node_sample_weight": np.array([n[1] for n in nodes], dtype=np.float64),
                    }
                    self.trees.append(SingleTree(tree, data=data, data_missing=data_missing))
            self.objective = objective_name_map.get(model.loss, None)
            self.tree_output = "log_odds"
        elif safe_isinstance(
            model,
            [
                "sklearn.ensemble.GradientBoostingClassifier",
                "sklearn.ensemble._gb.GradientBoostingClassifier",
                "sklearn.ensemble.gradient_boosting.GradientBoostingClassifier",
            ],
        ):
            self.input_dtype = np.float32

            # TODO: deal with estimators for each class
            if model.estimators_.shape[1] > 1:
                emsg =  "GradientBoostingClassifier is only supported for binary classification right now!"
                raise InvalidModelError(emsg)

            # currently we only support the logs odds estimator
            if safe_isinstance(
                model.init_,
                [
                    "sklearn.ensemble.LogOddsEstimator",
                    "sklearn.ensemble.gradient_boosting.LogOddsEstimator",
                ],
            ):
                self.base_offset = model.init_.prior
                self.tree_output = "log_odds"
            elif safe_isinstance(model.init_, "sklearn.dummy.DummyClassifier"):
                self.base_offset = scipy.special.logit(model.init_.class_prior_[1])  # with two classes the trees only model the second class. # pylint: disable=no-member
                self.tree_output = "log_odds"
            else:
                emsg = f"Unsupported init model type: {type(model.init_)}"
                raise InvalidModelError(emsg)

            self.trees = [SingleTree(e.tree_, scaling=model.learning_rate, data=data, data_missing=data_missing) for e in model.estimators_[:,0]]
            self.objective = objective_name_map.get(model.criterion, None)
        elif "pyspark.ml" in str(type(model)):
            assert_import("pyspark")
            self.model_type = "pyspark"
            # model._java_obj.getImpurity() can be gini, entropy or variance.
            self.objective = objective_name_map.get(model._java_obj.getImpurity(), None)
            if "Classification" in str(type(model)):
                normalize = True
                self.tree_output = "probability"
            else:
                normalize = False
                self.tree_output = "raw_value"
            # Spark Random forest, create 1 weighted (avg) tree per sub-model
            if safe_isinstance(
                model,
                [
                    "pyspark.ml.classification.RandomForestClassificationModel",
                    "pyspark.ml.regression.RandomForestRegressionModel",
                ],
            ):
                sum_weight = sum(model.treeWeights)  # output is average of trees
                self.trees = [
                    SingleTree(tree, normalize=normalize, scaling=model.treeWeights[i] / sum_weight)
                    for i, tree in enumerate(model.trees)
                ]
            # Spark GBT, create 1 weighted (learning rate) tree per sub-model
            elif safe_isinstance(
                model,
                [
                    "pyspark.ml.classification.GBTClassificationModel",
                    "pyspark.ml.regression.GBTRegressionModel",
                ],
            ):
                self.objective = "squared_error"  # GBT subtree use the variance
                self.tree_output = "raw_value"
                self.trees = [
                    SingleTree(tree, normalize=False, scaling=model.treeWeights[i])
                    for i, tree in enumerate(model.trees)
                ]
            # Spark Basic model (single tree)
            elif safe_isinstance(
                model,
                [
                    "pyspark.ml.classification.DecisionTreeClassificationModel",
                    "pyspark.ml.regression.DecisionTreeRegressionModel",
                ],
            ):
                self.trees = [SingleTree(model, normalize=normalize, scaling=1)]
            else:
                emsg = f"Unsupported Spark model type: {type(model)}"
                raise NotImplementedError(emsg)
        elif safe_isinstance(model, "xgboost.core.Booster"):
            self.original_model = model
            self.model_type = "xgboost"
            xgb_loader = XGBTreeModelLoader(self.original_model)
            self.trees = xgb_loader.get_trees(data=data, data_missing=data_missing)
            self.base_offset = xgb_loader.base_score
            self.objective = objective_name_map.get(xgb_loader.name_obj, None)
            self.tree_output = tree_output_name_map.get(xgb_loader.name_obj, None)
            if xgb_loader.num_class > 0:
                self.num_stacked_models = xgb_loader.num_class
        elif safe_isinstance(model, "xgboost.sklearn.XGBClassifier"):
            self.input_dtype = np.float32
            self.model_type = "xgboost"
            self.original_model = model.get_booster()
            xgb_loader = XGBTreeModelLoader(self.original_model)
            self.trees = xgb_loader.get_trees(data=data, data_missing=data_missing)
            self.base_offset = xgb_loader.base_score
            self.objective = objective_name_map.get(xgb_loader.name_obj, None)
            self.tree_output = tree_output_name_map.get(xgb_loader.name_obj, None)
            # 'best_ntree_limit' is problematic
            # https://github.com/dmlc/xgboost/issues/6615
            if hasattr(model, 'best_iteration'):
                trees_per_iteration = xgb_loader.num_class if xgb_loader.num_class > 0 else 1
                self.tree_limit = (getattr(model, "best_iteration", None) + 1) * trees_per_iteration
            else:
                self.tree_limit = getattr(model, "best_ntree_limit", None)
            if xgb_loader.num_class > 0:
                self.num_stacked_models = xgb_loader.num_class
            if self.model_output == "predict_proba":
                if self.num_stacked_models == 1:
                    self.model_output = "probability_doubled" # with predict_proba we need to double the outputs to match
                else:
                    self.model_output = "probability"
        elif safe_isinstance(model, "xgboost.sklearn.XGBRegressor"):
            self.original_model = model.get_booster()
            self.model_type = "xgboost"
            xgb_loader = XGBTreeModelLoader(self.original_model)
            self.trees = xgb_loader.get_trees(data=data, data_missing=data_missing)
            self.base_offset = xgb_loader.base_score
            self.objective = objective_name_map.get(xgb_loader.name_obj, None)
            self.tree_output = tree_output_name_map.get(xgb_loader.name_obj, None)
            self.tree_limit = getattr(model, "best_ntree_limit", None)
            if xgb_loader.num_class > 0:
                self.num_stacked_models = xgb_loader.num_class
        elif safe_isinstance(model, "xgboost.sklearn.XGBRanker"):
            self.original_model = model.get_booster()
            self.model_type = "xgboost"
            xgb_loader = XGBTreeModelLoader(self.original_model)
            self.trees = xgb_loader.get_trees(data=data, data_missing=data_missing)
            self.base_offset = xgb_loader.base_score
            # Note: for ranker, leaving tree_output and objective as None as they
            # are not implemented in native code yet
            self.tree_limit = getattr(model, "best_ntree_limit", None)
            if xgb_loader.num_class > 0:
                self.num_stacked_models = xgb_loader.num_class
        elif safe_isinstance(model, "lightgbm.basic.Booster"):
            assert_import("lightgbm")
            self.model_type = "lightgbm"
            self.original_model = model
            tree_info = self.original_model.dump_model()["tree_info"]
            try:
                self.trees = [SingleTree(e, data=data, data_missing=data_missing) for e in tree_info]
            except Exception:
                self.trees = None # we get here because the cext can't handle categorical splits yet

            self.objective = objective_name_map.get(model.params.get("objective", "regression"), None)
            self.tree_output = tree_output_name_map.get(model.params.get("objective", "regression"), None)

        elif safe_isinstance(model, "gpboost.basic.Booster"):
            assert_import("gpboost")
            self.model_type = "gpboost"
            self.original_model = model
            tree_info = self.original_model.dump_model()["tree_info"]
            try:
                self.trees = [SingleTree(e, data=data, data_missing=data_missing) for e in tree_info]
            except Exception:
                self.trees = None # we get here because the cext can't handle categorical splits yet

            self.objective = objective_name_map.get(model.params.get("objective", "regression"), None)
            self.tree_output = tree_output_name_map.get(model.params.get("objective", "regression"), None)

        elif safe_isinstance(model, "lightgbm.sklearn.LGBMRegressor"):
            assert_import("lightgbm")
            self.model_type = "lightgbm"
            self.original_model = model.booster_
            tree_info = self.original_model.dump_model()["tree_info"]
            try:
                self.trees = [SingleTree(e, data=data, data_missing=data_missing) for e in tree_info]
            except Exception:
                self.trees = None # we get here because the cext can't handle categorical splits yet
            self.objective = objective_name_map.get(model.objective, None)
            self.tree_output = tree_output_name_map.get(model.objective, None)
            if model.objective is None:
                self.objective = "squared_error"
                self.tree_output = "raw_value"
        elif safe_isinstance(model, "lightgbm.sklearn.LGBMRanker"):
            assert_import("lightgbm")
            self.model_type = "lightgbm"
            self.original_model = model.booster_
            tree_info = self.original_model.dump_model()["tree_info"]
            try:
                self.trees = [SingleTree(e, data=data, data_missing=data_missing) for e in tree_info]
            except Exception:
                self.trees = None # we get here because the cext can't handle categorical splits yet
            # Note: for ranker, leaving tree_output and objective as None as they
            # are not implemented in native code yet
        elif safe_isinstance(model, "lightgbm.sklearn.LGBMClassifier"):
            assert_import("lightgbm")
            self.model_type = "lightgbm"
            if model.n_classes_ > 2:
                self.num_stacked_models = model.n_classes_
            self.original_model = model.booster_
            tree_info = self.original_model.dump_model()["tree_info"]
            try:
                self.trees = [SingleTree(e, data=data, data_missing=data_missing) for e in tree_info]
            except Exception:
                self.trees = None # we get here because the cext can't handle categorical splits yet
            self.objective = objective_name_map.get(model.objective, None)
            self.tree_output = tree_output_name_map.get(model.objective, None)
            if model.objective is None:
                self.objective = "binary_crossentropy"
                self.tree_output = "log_odds"
        elif safe_isinstance(model, "catboost.core.CatBoostRegressor"):
            assert_import("catboost")
            self.model_type = "catboost"
            self.original_model = model
            self.cat_feature_indices = model.get_cat_feature_indices()
        elif safe_isinstance(model, "catboost.core.CatBoostClassifier"):
            assert_import("catboost")
            self.model_type = "catboost"
            self.original_model = model
            self.input_dtype = np.float32
            try:
                cb_loader = CatBoostTreeModelLoader(model)
                self.trees = cb_loader.get_trees(data=data, data_missing=data_missing)
            except Exception:
                self.trees = None # we get here because the cext can't handle categorical splits yet
            self.tree_output = "log_odds"
            self.objective = "binary_crossentropy"
            self.cat_feature_indices = model.get_cat_feature_indices()
        elif safe_isinstance(model, "catboost.core.CatBoost"):
            assert_import("catboost")
            self.model_type = "catboost"
            self.original_model = model
            self.cat_feature_indices = model.get_cat_feature_indices()
        elif safe_isinstance(model, "imblearn.ensemble._forest.BalancedRandomForestClassifier"):
            self.input_dtype = np.float32
            scaling = 1.0 / len(model.estimators_) # output is average of trees
            self.trees = [SingleTree(e.tree_, normalize=True, scaling=scaling, data=data, data_missing=data_missing) for e in model.estimators_]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "probability"
        elif safe_isinstance(
            model,
            [
                "ngboost.ngboost.NGBoost",
                "ngboost.api.NGBRegressor",
                "ngboost.api.NGBClassifier",
            ],
        ):
            assert model.base_models, "The NGBoost model has empty `base_models`! Have you called `model.fit`?"
            if self.model_output == "raw":
                param_idx = 0 # default to the first parameter of the output distribution
                warnings.warn("Translating model_output=\"raw\" to model_output=0 for the 0-th parameter in the distribution. Use model_output=0 directly to avoid this warning.")
            elif type(self.model_output) is int:
                param_idx = self.model_output
                self.model_output = "raw" # note that after loading we have a new model_output type
            assert safe_isinstance(model.base_models[0][param_idx], ["sklearn.tree.DecisionTreeRegressor", "sklearn.tree.tree.DecisionTreeRegressor"]), "You must use default_tree_learner!"
            shap_trees = [trees[param_idx] for trees in model.base_models]
            self.internal_dtype = shap_trees[0].tree_.value.dtype.type
            self.input_dtype = np.float32
            scaling = - model.learning_rate * np.array(model.scalings) # output is weighted average of trees
            self.trees = [SingleTree(e.tree_, scaling=s, data=data, data_missing=data_missing) for e,s in zip(shap_trees,scaling)]
            self.objective = objective_name_map.get(shap_trees[0].criterion, None)
            self.tree_output = "raw_value"
            self.base_offset = model.init_params[param_idx]
        else:
            raise InvalidModelError("Model type not yet supported by TreeExplainer: " + str(type(model)))

        # build a dense numpy version of all the tree objects
        if self.trees is not None and self.trees:
            max_nodes = np.max([len(t.values) for t in self.trees])
            assert len(np.unique([t.values.shape[1] for t in self.trees])) == 1, "All trees in the ensemble must have the same output dimension!"
            num_trees = len(self.trees)
            if self.num_stacked_models > 1:
                assert len(self.trees) % self.num_stacked_models == 0, "Only stacked models with equal numbers of trees are supported!"
                assert self.trees[0].values.shape[1] == 1, "Only stacked models with single outputs per model are supported!"
                self.num_outputs = self.num_stacked_models
            else:
                self.num_outputs = self.trees[0].values.shape[1]

            # important to be -1 in unused sections!! This way we can tell which entries are valid.
            self.children_left = -np.ones((num_trees, max_nodes), dtype=np.int32)
            self.children_right = -np.ones((num_trees, max_nodes), dtype=np.int32)
            self.children_default = -np.ones((num_trees, max_nodes), dtype=np.int32)
            self.features = -np.ones((num_trees, max_nodes), dtype=np.int32)

            self.thresholds = np.zeros((num_trees, max_nodes), dtype=self.internal_dtype)
            self.values = np.zeros((num_trees, max_nodes, self.num_outputs), dtype=self.internal_dtype)
            self.node_sample_weight = np.zeros((num_trees, max_nodes), dtype=self.internal_dtype)

            for i in range(num_trees):
                self.children_left[i,:len(self.trees[i].children_left)] = self.trees[i].children_left
                self.children_right[i,:len(self.trees[i].children_right)] = self.trees[i].children_right
                self.children_default[i,:len(self.trees[i].children_default)] = self.trees[i].children_default
                self.features[i,:len(self.trees[i].features)] = self.trees[i].features
                self.thresholds[i,:len(self.trees[i].thresholds)] = self.trees[i].thresholds
                if self.num_stacked_models > 1:
                    # stack_pos = int(i // (num_trees / self.num_stacked_models))
                    stack_pos = i % self.num_stacked_models
                    self.values[i,:len(self.trees[i].values[:,0]),stack_pos] = self.trees[i].values[:,0]
                else:
                    self.values[i,:len(self.trees[i].values)] = self.trees[i].values
                self.node_sample_weight[i,:len(self.trees[i].node_sample_weight)] = self.trees[i].node_sample_weight

                # ensure that the passed background dataset lands in every leaf
                if np.min(self.trees[i].node_sample_weight) <= 0:
                    self.fully_defined_weighting = False

            self.num_nodes = np.array([len(t.values) for t in self.trees], dtype=np.int32)
            self.max_depth = np.max([t.max_depth for t in self.trees])

            # make sure the base offset is a 1D array
            if not hasattr(self.base_offset, "__len__") or len(self.base_offset) == 0:
                self.base_offset = (np.ones(self.num_outputs) * self.base_offset).astype(self.internal_dtype)
            self.base_offset = self.base_offset.flatten()
            assert len(self.base_offset) == self.num_outputs

    def get_transform(self):
        """ A consistent interface to make predictions from this model.
        """
        if self.model_output == "raw":
            transform = "identity"
        elif self.model_output in ("probability", "probability_doubled"):
            if self.tree_output == "log_odds":
                transform = "logistic"
            elif self.tree_output == "probability":
                transform = "identity"
            else:
                emsg = (
                    "model_output = \"probability\" is not yet supported when model.tree_output = "
                    f"\"{self.tree_output}\"!"
                )
                raise NotImplementedError(emsg)
        elif self.model_output == "log_loss":
            if self.objective == "squared_error":
                transform = "squared_loss"
            elif self.objective == "binary_crossentropy":
                transform = "logistic_nlogloss"
            else:
                emsg = (
                    "model_output = \"log_loss\" is not yet supported when model.objective = "
                    f"\"{self.objective}\"!"
                )
                raise NotImplementedError(emsg)
        else:
            emsg = (
                f"Unrecognized model_output parameter value: {str(self.model_output)}! "
                f"If `model.{str(self.model_output)}` is a valid function, open a Github issue to ask "
                "that this method be supported. If you want 'predict_proba' just use 'probability' for now."
            )
            raise ValueError(emsg)

        return transform

    def predict(self, X, y=None, output=None, tree_limit=None):
        """ A consistent interface to make predictions from this model.

        Parameters
        ----------
        tree_limit : None (default) or int
            Limit the number of trees used by the model. By default None means no use the limit of the
            original model, and -1 means no limit.
        """

        if output is None:
            output = self.model_output

        if self.model_type == "pyspark":
            #import pyspark
            # TODO: support predict for pyspark
            raise NotImplementedError("Predict with pyspark isn't implemented. Don't run 'interventional' as feature_perturbation.")

        # see if we have a default tree_limit in place.
        if tree_limit is None:
            tree_limit = -1 if self.tree_limit is None else self.tree_limit

        # convert dataframes
        if safe_isinstance(X, "pandas.core.series.Series"):
            X = X.values
        elif safe_isinstance(X, "pandas.core.frame.DataFrame"):
            X = X.values
        flat_output = False
        if len(X.shape) == 1:
            flat_output = True
            X = X.reshape(1, X.shape[0])
        if X.dtype.type != self.input_dtype:
            X = X.astype(self.input_dtype)
        X_missing = np.isnan(X, dtype=bool)
        assert isinstance(X, np.ndarray), "Unknown instance type: " + str(type(X))
        assert len(X.shape) == 2, "Passed input data matrix X must have 1 or 2 dimensions!"

        if tree_limit < 0 or tree_limit > self.values.shape[0]:
            tree_limit = self.values.shape[0]

        if output == "logloss":
            assert y is not None, "Both samples and labels must be provided when explaining the loss (i.e. `explainer.shap_values(X, y)`)!"
            assert X.shape[0] == len(y), "The number of labels (%d) does not match the number of samples to explain (%d)!" % (len(y), X.shape[0])
        transform = self.get_transform()
        assert_import("cext")
        output = np.zeros((X.shape[0], self.num_outputs))
        _cext.dense_tree_predict(
            self.children_left, self.children_right, self.children_default,
            self.features, self.thresholds, self.values,
            self.max_depth, tree_limit, self.base_offset, output_transform_codes[transform],
            X, X_missing, y, output
        )

        # drop dimensions we don't need
        if flat_output:
            if self.num_outputs == 1:
                return output.flatten()[0]
            else:
                return output.reshape(-1, self.num_outputs)
        else:
            if self.num_outputs == 1:
                return output.flatten()
            else:
                return output


class SingleTree:
    """A single decision tree.

    The primary point of this object is to parse many different tree types into a common format.

    Attributes
    ----------
    children_left : numpy.array
        A 1d array of length #nodes. The index ``i`` of this array contains the index of
        the left-child of the ``i-th`` node in the tree. An index of -1 is used to
        represent that the ``i-th`` node is a leaf/terminal node.

    children_right : numpy.array
        Same as ``children_left``, except it contains the index of the right child of
        each ``i-th`` node in the tree.

    children_default : numpy.array
        A 1d numpy array of length #nodes. The index ``i`` of this array contains either
        the index of the left-child / right-child of the ``i-th`` node in the tree,
        depending on whether the default split (for handling missing values) is left /
        right. An index of -1 is used to represent that the ``i-th`` node is a leaf
        node.

    features : numpy.array
        A 1d numpy array of length #nodes. The value at the ``i-th`` position is the
        index of the feature chosen for the split at node ``i``. Leaf nodes have no
        splits, so is -1.

    thresholds : numpy.array
        A 1d numpy array of length #nodes. The value at the ``i-th`` position is the
        threshold used for the split at node ``i``. Leaf nodes have no thresholds, so is
        -1.

    values : numpy.array
        A 1d numpy array of length #nodes. The index ``i`` of this array contains the
        raw predicted value that would be produced by node ``i`` if it were a leaf node.

    node_sample_weight : numpy.array
        A 1d numpy array of length #nodes. The index ``i`` contains the number of
        records (usually from the training data) that falls into node ``i``.

    max_depth : int
        The max depth of the tree.
    """
    def __init__(self, tree, normalize=False, scaling=1.0, data=None, data_missing=None):
        assert_import("cext")

        if safe_isinstance(tree, ["sklearn.tree._tree.Tree", "econml.tree._tree.Tree"]):
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

        elif type(tree) is dict and "features" in tree:
            self.children_left = tree["children_left"].astype(np.int32)
            self.children_right = tree["children_right"].astype(np.int32)
            self.children_default = tree["children_default"].astype(np.int32)
            self.features = tree["features"].astype(np.int32)
            self.thresholds = tree["thresholds"]
            self.values = tree["values"] * scaling
            self.node_sample_weight = tree["node_sample_weight"]

        # deprecated dictionary support (with sklearn singular style "feature" and "value" names)
        elif type(tree) is dict and "children_left" in tree:
            self.children_left = tree["children_left"].astype(np.int32)
            self.children_right = tree["children_right"].astype(np.int32)
            self.children_default = tree["children_default"].astype(np.int32)
            self.features = tree["feature"].astype(np.int32)
            self.thresholds = tree["threshold"]
            self.values = tree["value"] * scaling
            self.node_sample_weight = tree["node_sample_weight"]

        elif safe_isinstance(
            tree,
            [
                "pyspark.ml.classification.DecisionTreeClassificationModel",
                "pyspark.ml.regression.DecisionTreeRegressionModel",
            ],
        ):
            #model._java_obj.numNodes() doesn't give leaves, need to recompute the size
            def getNumNodes(node, size):
                size = size + 1
                if node.subtreeDepth() == 0:
                    return size
                else:
                    size = getNumNodes(node.leftChild(), size)
                    return getNumNodes(node.rightChild(), size)

            num_nodes = getNumNodes(tree._java_obj.rootNode(), 0)
            self.children_left = np.full(num_nodes, -2, dtype=np.int32)
            self.children_right = np.full(num_nodes, -2, dtype=np.int32)
            self.children_default = np.full(num_nodes, -2, dtype=np.int32)
            self.features = np.full(num_nodes, -2, dtype=np.int32)
            self.thresholds = np.full(num_nodes, -2, dtype=np.float64)
            self.values = [-2]*num_nodes
            self.node_sample_weight = np.full(num_nodes, -2, dtype=np.float64)
            def buildTree(index, node):
                index = index + 1
                if tree._java_obj.getImpurity() == 'variance':
                    self.values[index] = [node.prediction()]  # prediction for the node
                else:
                    self.values[index] = [e for e in node.impurityStats().stats()] #for gini: NDarray(numLabel): 1 per label: number of item for each label which went through this node
                self.node_sample_weight[index] = node.impurityStats().count()  # weighted count of element through this node

                if node.subtreeDepth() == 0:
                    return index
                else:
                    self.features[index] = node.split().featureIndex() #index of the feature we split on, not available for leaf, int
                    if str(node.split().getClass()).endswith('tree.CategoricalSplit'):
                        #Categorical split isn't implemented, TODO: could fake it by creating a fake node to split on the exact value?
                        raise NotImplementedError('CategoricalSplit are not yet implemented')
                    self.thresholds[index] = node.split().threshold() #threshold for the feature, not available for leaf, float

                    self.children_left[index] = index + 1
                    idx = buildTree(index, node.leftChild())
                    self.children_right[index] = idx + 1
                    idx = buildTree(idx, node.rightChild())
                    return idx

            buildTree(-1, tree._java_obj.rootNode())
            #default Not supported with mlib? (TODO)
            self.children_default = self.children_left
            self.values = np.asarray(self.values)
            if normalize:
                self.values = (self.values.T / self.values.sum(1)).T
            self.values = self.values * scaling

        # dictionary output from LightGBM `.dump_model()`
        elif isinstance(tree, dict) and "tree_structure" in tree:
            start = tree["tree_structure"]
            num_parents = tree["num_leaves"] - 1
            num_nodes = 2 * num_parents + 1
            self.children_left = np.empty(num_nodes, dtype=np.int32)
            self.children_right = np.empty(num_nodes, dtype=np.int32)
            self.children_default = np.empty(num_nodes, dtype=np.int32)
            self.features = np.empty(num_nodes, dtype=np.int32)
            self.thresholds = np.empty(num_nodes, dtype=np.float64)
            self.values = [-2 for _ in range(num_nodes)]
            self.node_sample_weight = np.empty(num_nodes, dtype=np.float64)

            # BFS traversal through the tree structure
            visited, queue = [], [start]
            while queue:
                vertex = queue.pop(0)  # TODO(perf): benchmark this against deque.popleft()
                is_branch_node = "split_index" in vertex
                if is_branch_node:
                    vsplit_idx: int = vertex["split_index"]
                    if vsplit_idx in visited:
                        continue

                    left_child: dict = vertex["left_child"]
                    right_child: dict = vertex["right_child"]
                    left_is_branch_node = "split_index" in left_child
                    if left_is_branch_node:
                        self.children_left[vsplit_idx] = left_child["split_index"]
                    else:
                        self.children_left[vsplit_idx] = left_child["leaf_index"] + num_parents
                    right_is_branch_node = "split_index" in right_child
                    if right_is_branch_node:
                        self.children_right[vsplit_idx] = right_child["split_index"]
                    else:
                        self.children_right[vsplit_idx] = right_child["leaf_index"] + num_parents
                    if vertex["default_left"]:
                        self.children_default[vsplit_idx] = self.children_left[vsplit_idx]
                    else:
                        self.children_default[vsplit_idx] = self.children_right[vsplit_idx]

                    self.features[vsplit_idx] = vertex["split_feature"]
                    self.thresholds[vsplit_idx] = vertex["threshold"]
                    self.values[vsplit_idx] = [vertex["internal_value"]]
                    self.node_sample_weight[vsplit_idx] = vertex["internal_count"]
                    visited.append(vsplit_idx)
                    queue.append(left_child)
                    queue.append(right_child)
                else:
                    vleaf_idx: int = vertex["leaf_index"] + num_parents
                    self.children_left[vleaf_idx] = -1
                    self.children_right[vleaf_idx] = -1
                    self.children_default[vleaf_idx] = -1
                    self.features[vleaf_idx] = -1
                    self.children_left[vleaf_idx] = -1
                    self.children_right[vleaf_idx] = -1
                    self.children_default[vleaf_idx] = -1
                    self.features[vleaf_idx] = -1
                    self.thresholds[vleaf_idx] = -1
                    self.values[vleaf_idx] = [vertex["leaf_value"]]
                    self.node_sample_weight[vleaf_idx] = vertex["leaf_count"]
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
            for n in nodes:
                nodes_dict[int(n.split(":")[0])] = n.split(":")[1]
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
                if "leaf" in value:
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
            raise TypeError("Unknown input to SingleTree constructor: " + str(tree))

        # Re-compute the number of samples that pass through each node if we are given data
        if data is not None and data_missing is not None:
            self.node_sample_weight.fill(0.0)
            _cext.dense_tree_update_weights(
                self.children_left, self.children_right, self.children_default, self.features,
                self.thresholds, self.values, 1, self.node_sample_weight, data, data_missing
            )

        # we compute the expectations to make sure they follow the SHAP logic
        self.max_depth = _cext.compute_expectations(
            self.children_left, self.children_right, self.node_sample_weight,
            self.values
        )


class IsoTree(SingleTree):
    """
    In sklearn the tree of the Isolation Forest does not calculated in a good way.
    """
    def __init__(self, tree, tree_features, normalize=False, scaling=1.0, data=None, data_missing=None):
        super().__init__(tree, normalize, scaling, data, data_missing)
        if safe_isinstance(tree, "sklearn.tree._tree.Tree"):
            from sklearn.ensemble._iforest import (
                _average_path_length,  # pylint: disable=no-name-in-module
            )

            def _recalculate_value(tree, i , level):
                if tree.children_left[i] == -1 and tree.children_right[i] == -1:
                    value = level + _average_path_length(np.array([tree.n_node_samples[i]]))[0]
                    self.values[i, 0] =  value
                    return value * tree.n_node_samples[i]
                else:
                    value_left = _recalculate_value(tree, tree.children_left[i] , level + 1)
                    value_right = _recalculate_value(tree, tree.children_right[i] , level + 1)
                    self.values[i, 0] =  (value_left + value_right) / tree.n_node_samples[i]
                    return value_left + value_right

            _recalculate_value(tree, 0, 0)
            if normalize:
                self.values = (self.values.T / self.values.sum(1)).T
            self.values = self.values * scaling
            # re-number the features if each tree gets a different set of features
            self.features = np.where(self.features >= 0, tree_features[self.features], self.features)


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


class XGBTreeModelLoader:
    """ This loads an XGBoost model directly from a raw memory dump.

    We can't use the JSON dump because due to numerical precision issues those
    tree can actually be wrong when feature values land almost on a threshold.
    """
    def __init__(self, xgb_model):
        # new in XGBoost 1.1, 'binf' is appended to the buffer
        self.buf = xgb_model.save_raw()
        if self.buf.startswith(b'binf'):
            self.buf = self.buf[4:]
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

        # new in XGBoost 1.0 is that the base_score is saved untransformed (https://github.com/dmlc/xgboost/pull/5101)
        # so we have to transform it depending on the objective
        import xgboost
        if version.parse(xgboost.__version__).major >= 1:
            if self.name_obj in ["binary:logistic", "reg:logistic"]:
                self.base_score = scipy.special.logit(self.base_score) # pylint: disable=no-member

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

            # load the stat nodes
            self.loss_chg.append(np.zeros(self.num_nodes[i], dtype=np.float32))
            self.sum_hess.append(np.zeros(self.num_nodes[i], dtype=np.float64))
            self.base_weight.append(np.zeros(self.num_nodes[i], dtype=np.float32))
            self.leaf_child_cnt.append(np.zeros(self.num_nodes[i], dtype=int))
            for j in range(self.num_nodes[i]):
                self.loss_chg[-1][j] = self.read('f')
                self.sum_hess[-1][j] = self.read('f')
                self.base_weight[-1][j] = self.read('f')
                self.leaf_child_cnt[-1][j] = self.read('i')

    def get_trees(self, data=None, data_missing=None):
        shape = (self.num_trees, self.num_nodes.max())
        self.children_default = np.zeros(shape, dtype=int)
        self.features = np.zeros(shape, dtype=int)
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
                    # Xgboost uses < for thresholds where shap uses <=
                    # Move the threshold down by the smallest possible increment
                    self.thresholds[i, j] = np.nextafter(self.node_info[i][j], - np.float32(np.inf))
                else:
                    self.values[i,j] = self.node_info[i][j]

            l = len(self.node_cleft[i])
            trees.append(SingleTree({
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


class CatBoostTreeModelLoader:
    def __init__(self, cb_model):
        # cb_model.save_model("cb_model.json", format="json")
        # self.loaded_cb_model = json.load(open("cb_model.json", "r"))
        import tempfile
        tmp_file = tempfile.NamedTemporaryFile()
        cb_model.save_model(tmp_file.name, format="json")
        self.loaded_cb_model = json.load(open(tmp_file.name))
        tmp_file.close()

        # load the CatBoost oblivious trees specific parameters
        self.num_trees = len(self.loaded_cb_model['oblivious_trees'])
        self.max_depth = self.loaded_cb_model['model_info']['params']['tree_learner_options']['depth']

    def get_trees(self, data=None, data_missing=None):
        # load each tree
        trees = []
        for tree_index in range(self.num_trees):

            # load the per-tree params
            #depth = len(self.loaded_cb_model['oblivious_trees'][tree_index]['splits'])

            # load the nodes

            # Re-compute the number of samples that pass through each node if we are given data
            leaf_weights = self.loaded_cb_model['oblivious_trees'][tree_index]['leaf_weights']
            leaf_weights_unraveled = [0] * (len(leaf_weights) - 1) + leaf_weights
            leaf_weights_unraveled[0] = sum(leaf_weights)
            for index in range(len(leaf_weights) - 2, 0, -1):
                leaf_weights_unraveled[index] = leaf_weights_unraveled[2 * index + 1] + leaf_weights_unraveled[2 * index + 2]

            leaf_values = self.loaded_cb_model['oblivious_trees'][tree_index]['leaf_values']
            leaf_values_unraveled = [0] * (len(leaf_values) - 1) + leaf_values

            children_left = [i * 2 + 1 for i in range(len(leaf_values) - 1)]
            children_left += [-1] * len(leaf_values)

            children_right = [i * 2 for i in range(1, len(leaf_values))]
            children_right += [-1] * len(leaf_values)

            children_default = [i * 2 + 1 for i in range(len(leaf_values) - 1)]
            children_default += [-1] * len(leaf_values)

            # load the split features and borders
            # split features and borders go from leafs to the root
            split_features_index = []
            borders = []

            # split features and borders go from leafs to the root
            for elem in self.loaded_cb_model['oblivious_trees'][tree_index]['splits']:
                split_type = elem.get('split_type')
                if split_type == 'FloatFeature':
                    split_feature_index = elem.get('float_feature_index')
                    borders.append(elem['border'])
                elif split_type == 'OneHotFeature':
                    split_feature_index = elem.get('cat_feature_index')
                    borders.append(elem['value'])
                else:
                    split_feature_index = elem.get('ctr_target_border_idx')
                    borders.append(elem['border'])
                split_features_index.append(split_feature_index)

            split_features_index_unraveled = []
            for counter, feature_index in enumerate(split_features_index[::-1]):
                split_features_index_unraveled += [feature_index] * (2 ** counter)
            split_features_index_unraveled += [0] * len(leaf_values)

            borders_unraveled = []
            for counter, border in enumerate(borders[::-1]):
                borders_unraveled += [border] * (2 ** counter)
            borders_unraveled += [0] * len(leaf_values)

            trees.append(SingleTree({"children_left": np.array(children_left),
                             "children_right": np.array(children_right),
                             "children_default": np.array(children_default),
                             "feature": np.array(split_features_index_unraveled),
                             "threshold": np.array(borders_unraveled),
                             "value": np.array(leaf_values_unraveled).reshape((-1,1)),
                             "node_sample_weight": np.array(leaf_weights_unraveled),
                            }, data=data, data_missing=data_missing))

        return trees

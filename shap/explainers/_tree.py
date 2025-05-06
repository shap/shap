from __future__ import annotations

import io
import json
import os
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse
import scipy.special
from packaging import version

from .. import maskers
from .._explanation import Explanation
from ..utils import assert_import, record_import_error, safe_isinstance
from ..utils._exceptions import (
    DimensionError,
    ExplainerError,
    InvalidFeaturePerturbationError,
    InvalidMaskerError,
    InvalidModelError,
)
from ..utils._legacy import DenseData
from ..utils._warnings import ExperimentalWarning
from ._explainer import Explainer
from .other._ubjson import decode_ubjson_buffer

try:
    from .. import _cext  # type: ignore
except ImportError as e:
    record_import_error("cext", "C extension was not built during install!", e)

try:
    import pyspark  # noqa
except ImportError as e:
    record_import_error("pyspark", "PySpark could not be imported!", e)

DEPRECATED_APPROX = object()

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


def _safe_check_tree_instance_experimental(tree_instance: Any) -> None:
    """
    This function checks if a tree instance has an experimental integration with shap TreeExplainer class.

    To add experimental message support for your library add package name and its versions
    verified to be used with shap to the 'experimental' dictionary below.

    Parameters
    ----------
    tree_instance: object, tree instance from an external library
    """
    experimental = {
        "causalml": "0.15.3",
    }

    safe_instance = None
    if hasattr(tree_instance, "__class__"):
        if hasattr(tree_instance.__class__, "__module__"):
            safe_instance = tree_instance

    if safe_instance:
        library = safe_instance.__class__.__module__.split(".")[0]
        if experimental.get(library):
            warnings.warn(
                f"You are using experimental integration with {library}. "
                f"The {library} support is verified for the following versions: {experimental.get(library)}. "
                f"As experimental functionality, this integration may be removed or significantly changed in future releases without following semantic versioning. Use in production systems at your own risk.",
                ExperimentalWarning,
            )
    else:
        warnings.warn(
            f"Unable to check experimental integration status for {tree_instance} object", ExperimentalWarning
        )


def _check_xgboost_version(v: str):
    if version.parse(v) < version.parse("1.6"):  # pragma: no cover
        raise RuntimeError(f"SHAP requires XGBoost >= v1.6 , but found version {v}. Please upgrade XGBoost.")


def _xgboost_n_iterations(tree_limit: int, num_stacked_models: int) -> int:
    """Convert number of trees to number of iterations for XGBoost models."""
    if tree_limit == -1:
        tree_limit = 0
    n_iterations = tree_limit // num_stacked_models
    return n_iterations


def _xgboost_cat_unsupported(model):
    if model.model_type == "xgboost" and model.cat_feature_indices is not None:
        raise NotImplementedError(
            "Categorical split is not yet supported. You can still use"
            " TreeExplainer with `feature_perturbation=tree_path_dependent`."
        )


class TreeExplainer(Explainer):
    """Uses Tree SHAP algorithms to explain the output of ensemble tree models.

    Tree SHAP is a fast and exact method to estimate SHAP values for tree models
    and ensembles of trees, under several different possible assumptions about
    feature dependence. It depends on fast C++ implementations either inside an
    external model package or in the local compiled C extension.

    Examples
    --------
    See `Tree explainer examples <https://shap.readthedocs.io/en/latest/api_examples/explainers/Tree.html>`_

    """

    def __init__(
        self,
        model,
        data=None,
        model_output="raw",
        feature_perturbation="auto",
        feature_names=None,
        approximate=DEPRECATED_APPROX,
        # FIXME: The `link` and `linearize_link` arguments are ignored. GH #3513
        link=None,
        linearize_link=None,
    ):
        """Build a new Tree explainer for the passed model.

        Parameters
        ----------
        model : model object
            The tree based machine learning model that we want to explain.
            XGBoost, LightGBM, CatBoost, Pyspark and most tree-based
            scikit-learn models are supported.

        data : numpy.array or pandas.DataFrame
            The background dataset to use for integrating out features.

            This argument is optional when
            ``feature_perturbation="tree_path_dependent"``, since in that case
            we can use the number of training samples that went down each tree
            path as our background dataset (this is recorded in the ``model``
            object).

        feature_perturbation : "auto" (default), "interventional" or "tree_path_dependent"
            Since SHAP values rely on conditional expectations, we need to
            decide how to handle correlated (or otherwise dependent) input
            features.

            - if ``"interventional"``, a background dataset ``data`` is required. The
              dependencies between features are handled according to the rules dictated
              by causal inference [1]_. The runtime scales linearly with the size of the
              background dataset you use: anywhere from 100 to 1000 random background
              samples are good sizes to use.
            - if ``"tree_path_dependent"``, no background dataset is required and the
              approach is to just follow the trees and use the number of training
              examples that went down each leaf to represent the background
              distribution.
            - if ``"auto"``, the "interventional" approach will be used when a
              background is provided, otherwise the "tree_path_dependent" approach will
              be used.

            .. versionadded:: 0.47
               The `"auto"` option was added.

            .. versionchanged:: 0.47
               The default behaviour will change from `"interventional"` to `"auto"` in 0.47.
               In the future, passing `feature_pertubation="interventional"` without providing
               a background dataset will raise an error.


        model_output : "raw", "probability", "log_loss", or model method name
            What output of the model should be explained.

            * If "raw", then we explain the raw output of the trees, which
              varies by model. For regression models, "raw" is the standard
              output. For binary classification in XGBoost, this is the log odds
              ratio.
            * If "probability", then we explain the output of the model
              transformed into probability space (note that this means the SHAP
              values now sum to the probability output of the model).
            * If "log_loss", then we explain the natural logarithm of the model
              loss function, so that the SHAP values sum up to the log loss of
              the model for each sample. This is helpful for breaking down model
              performance by feature.
            * If ``model_output`` is the name of a supported prediction method
              on the ``model`` object, then we explain the output of that model
              method name. For example, ``model_output="predict_proba"``
              explains the result of calling ``model.predict_proba``.

            Currently the "probability" and "log_loss" options are only
            supported when ``feature_perturbation="interventional"``.

        approximate : bool
            Deprecated, will be deprecated in v0.47.0 and removed in version v0.49.0.
            Please use the ``approximate`` argument in the :meth:`.shap_values` or ``__call__`` methods instead.

        References
        ----------
        .. [1] Janzing, Dominik, Lenon Minorics, and Patrick Blöbaum.
               "Feature relevance quantification in explainable AI: A causal problem."
               International Conference on artificial intelligence and statistics. PMLR, 2020.

        """
        if approximate is not DEPRECATED_APPROX:
            warnings.warn(
                "The approximate argument has been deprecated in version v0.47.0 and will be removed in version v0.48.0. "
                "Please use the approximate argument in the shap_values or the __call__ method instead.",
                DeprecationWarning,
            )
        if feature_names is not None:
            self.data_feature_names = feature_names
        elif isinstance(data, pd.DataFrame):
            self.data_feature_names = list(data.columns)

        masker = data
        super().__init__(model, masker, feature_names=feature_names)

        if type(self.masker) is maskers.Independent:
            data = self.masker.data
        elif masker is not None:
            raise InvalidMaskerError(f"Unsupported masker type: {str(type(self.masker))}!")

        if getattr(self.masker, "clustering", None) is not None:
            raise ExplainerError(
                "TreeExplainer does not support clustered data inputs! Please use shap.Explainer or pass an unclustered masker!"
            )

        if isinstance(data, pd.DataFrame):
            self.data = data.values
        elif isinstance(data, DenseData):
            self.data = data.data
        else:
            self.data = data

        if feature_perturbation == "auto":
            feature_perturbation = "interventional" if self.data is not None else "tree_path_dependent"
        elif feature_perturbation == "interventional":
            if self.data is None:
                # TODO: raise an error in 0.48
                warnings.warn(
                    "In the future, passing feature_perturbation='interventional' without providing a background dataset "
                    "will raise an error. Please provide a background dataset to continue using the interventional "
                    "approach or set feature_perturbation='auto' to automatically switch approaches.",
                    FutureWarning,
                )
                feature_perturbation = "tree_path_dependent"
        elif feature_perturbation != "tree_path_dependent":
            raise InvalidFeaturePerturbationError(
                "feature_perturbation must be 'auto', 'interventional', or 'tree_path_dependent'. "
                f"Got {feature_perturbation} instead."
            )

        elif feature_perturbation == "interventional" and self.data.shape[0] > 1_000:
            wmsg = (
                f"Passing {self.data.shape[0]} background samples may lead to slow runtimes. Consider "
                "using shap.sample(data, 100) to create a smaller background data set."
            )
            warnings.warn(wmsg)

        _safe_check_tree_instance_experimental(model)

        self.data_missing = None if self.data is None else pd.isna(self.data)
        self.feature_perturbation = feature_perturbation
        self.expected_value = None
        self.model = TreeEnsemble(model, self.data, self.data_missing, model_output)
        self.model_output = model_output
        # self.model_output = self.model.model_output # this allows the TreeEnsemble to translate model outputs types by how it loads the model

        # check for unsupported combinations of feature_perturbation and model_outputs
        if feature_perturbation == "tree_path_dependent":
            if self.model.model_output != "raw":
                raise ValueError('Only model_output="raw" is supported for feature_perturbation="tree_path_dependent"')
        elif data is None:
            raise ValueError(
                'A background dataset must be provided unless you are using feature_perturbation="tree_path_dependent"!'
            )

        if self.model.model_output != "raw":
            if self.model.objective is None and self.model.tree_output is None:
                emsg = (
                    "Model does not have a known objective or output type! When model_output is "
                    'not "raw" then we need to know the model\'s objective or link function.'
                )
                raise Exception(emsg)

        # A change in the signature of `xgboost.Booster.predict()` method has been introduced in XGBoost v1.4:
        # The introduced `iteration_range` parameter is used when obtaining SHAP (incl. interaction) values from XGBoost models.
        if self.model.model_type == "xgboost":
            import xgboost

            _check_xgboost_version(xgboost.__version__)

        # compute the expected value if we have a parsed tree for the cext
        if self.model.model_output == "log_loss":
            self.expected_value = self.__dynamic_expected_value
        elif data is not None:
            try:
                self.expected_value = self.model.predict(self.data).mean(0)
            except ValueError:
                raise ExplainerError(
                    "Currently TreeExplainer can only handle models with categorical splits when "
                    'feature_perturbation="tree_path_dependent" and no background data is passed. Please try again using '
                    'shap.TreeExplainer(model, feature_perturbation="tree_path_dependent").'
                )
            if hasattr(self.expected_value, "__len__") and len(self.expected_value) == 1:
                self.expected_value = self.expected_value[0]
        elif hasattr(self.model, "node_sample_weight"):
            self.expected_value = self.model.values[:, 0].sum(0)
            if self.expected_value.size == 1:
                self.expected_value = self.expected_value[0]
            self.expected_value += self.model.base_offset
            if self.model.model_output != "raw":
                self.expected_value = None  # we don't handle transforms in this case right now...

        # if our output format requires binary classification to be represented as two outputs then we do that here
        if self.model.model_output == "probability_doubled" and self.expected_value is not None:
            self.expected_value = [1 - self.expected_value, self.expected_value]

    def __dynamic_expected_value(self, y):
        """This computes the expected value conditioned on the given label value."""
        return self.model.predict(self.data, np.ones(self.data.shape[0]) * y).mean(0)

    def __call__(  # type: ignore
        self,
        X: Any,
        y: np.ndarray | pd.Series | None = None,
        interactions: bool = False,
        check_additivity: bool = True,
        approximate: bool = False,
    ) -> Explanation:
        """Calculate the SHAP values for the model applied to the data.

        Parameters
        ----------
        X : Any
            Can be a dataframe like object e.g. numpy.array, pandas.DataFrame or catboost.Pool (for catboost).
            A matrix of samples (# samples x # features) on which to explain the model's output.

        y : numpy.array, optional
            An array of label values for each sample. Used when explaining loss functions.

        approximate : bool
            Run fast, but only roughly approximate the Tree SHAP values. This runs a method
            previously proposed by Saabas which only considers a single feature ordering. Take care
            since this does not have the consistency guarantees of Shapley values and places too
            much weight on lower splits in the tree.

        interactions: bool
            Whether to compute the SHAP interaction values.

        check_additivity: bool
            Check if the sum of the SHAP values equals the output of the model.

        Returns
        -------
            shap.Explanation object containing the given data and the SHAP values.
        """
        start_time = time.time()

        feature_names: Any
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
        else:
            feature_names = getattr(self, "data_feature_names", None)

        if not interactions:
            v = self.shap_values(X, y=y, from_call=True, check_additivity=check_additivity, approximate=approximate)
            if isinstance(v, list):
                v = np.stack(v, axis=-1)  # put outputs at the end
        else:
            if approximate:
                raise NotImplementedError("Approximate computation not yet supported for interaction effects!")
            v = self.shap_interaction_values(X)

        # the Explanation object expects an `expected_value` for each row
        if hasattr(self.expected_value, "__len__") and len(self.expected_value) > 1:
            # `expected_value` is a list / array of numbers, length k, e.g. for multi-output scenarios
            # we repeat it N times along the first axis, so ev_tiled.shape == (N, k)
            if isinstance(v, list):
                num_rows = v[0].shape[0]
            else:
                num_rows = v.shape[0]
            ev_tiled = np.tile(self.expected_value, (num_rows, 1))
        else:
            # `expected_value` is a scalar / array of 1 number, so we simply repeat it for every row in `v`
            # ev_tiled.shape == (N,)
            ev_tiled = np.tile(self.expected_value, v.shape[0])

        X_data: np.ndarray | None | scipy.sparse.csr_matrix
        # cf. GH dsgibbons#66, this conversion to numpy array should be done AFTER
        # calculation of shap values
        if isinstance(X, pd.DataFrame):
            X_data = X.values
        elif safe_isinstance(X, "xgboost.core.DMatrix"):
            import xgboost

            if version.parse(xgboost.__version__) < version.parse("1.7.0"):  # pragma: no cover
                # cf. GH #3357
                wmsg = (
                    "`shap.Explanation` does not support `xgboost.DMatrix` objects for xgboost < 1.7, "
                    "so the `data` attribute of the `Explanation` object will be set to None. If "
                    "you require the `data` attribute (e.g. using `shap.plots`), then either "
                    "update your xgboost to >=1.7.0 or explicitly set `Explanation.data = X`, where "
                    "`X` is a numpy or scipy array."
                )
                warnings.warn(wmsg)
                X_data = None
            else:
                X_data = X.get_data()
        else:
            X_data = X

        return Explanation(
            v,
            base_values=ev_tiled,
            data=X_data,
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
        if isinstance(X, (pd.Series, pd.DataFrame)):
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
            if y is None:
                emsg = (
                    'Both samples and labels must be provided when model_output = "log_loss" '
                    "(i.e. `explainer.shap_values(X, y)`)!"
                )
                raise ExplainerError(emsg)
            if X.shape[0] != len(y):
                emsg = (
                    f"The number of labels ({len(y)}) does not match the number of samples to explain ({X.shape[0]})!"
                )
                raise DimensionError(emsg)

        if self.feature_perturbation == "tree_path_dependent":
            if not self.model.fully_defined_weighting:
                emsg = (
                    "The background dataset you provided does "
                    "not cover all the leaves in the model, "
                    "so TreeExplainer cannot run with the "
                    'feature_perturbation="tree_path_dependent" option! '
                    "Try providing a larger background "
                    "dataset, no background dataset, or using "
                    'feature_perturbation="interventional".'
                )
                raise ExplainerError(emsg)

        if check_additivity and self.model.model_type == "pyspark":
            warnings.warn(
                "check_additivity requires us to run predictions which is not supported with "
                "spark, "
                "ignoring."
                " Set check_additivity=False to remove this warning"
            )
            check_additivity = False

        return X, y, X_missing, flat_output, tree_limit, check_additivity

    def shap_values(
        self,
        X: Any,
        y: np.ndarray | pd.Series | None = None,
        tree_limit: int | None = None,
        approximate: bool = False,
        check_additivity: bool = True,
        from_call: bool = False,
    ):
        """Estimate the SHAP values for a set of samples.

        Parameters
        ----------
        X : Any
            Can be a dataframe like object, e.g. numpy.array, pandas.DataFrame or catboost.Pool (for catboost).
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
        np.array
            Estimated SHAP values, usually of shape ``(# samples x # features)``.

            Each row sums to the difference between the model output for that
            sample and the expected value of the model output (which is stored
            as the ``expected_value`` attribute of the explainer).

            The shape of the returned array depends on the number of model outputs:

            * one output: array of shape ``(#num_samples, *X.shape[1:])``.
            * multiple outputs: array of shape ``(#num_samples, *X.shape[1:],
              #num_outputs)``.

            .. versionchanged:: 0.45.0
                Return type for models with multiple outputs changed from list to np.ndarray.

        """
        # see if we have a default tree_limit in place.
        if tree_limit is None:
            tree_limit = -1 if self.model.tree_limit is None else self.model.tree_limit

        # shortcut using the C++ version of Tree SHAP in XGBoost, LightGBM, and CatBoost
        if (
            self.feature_perturbation == "tree_path_dependent"
            and self.model.model_type != "internal"
            and self.data is None
        ):
            model_output_vals = None
            phi = None
            if self.model.model_type == "xgboost":
                import xgboost

                n_iterations = _xgboost_n_iterations(tree_limit, self.model.num_stacked_models)
                if not isinstance(X, xgboost.core.DMatrix):
                    # Retrieve any DMatrix properties if they have been set on the TreeEnsemble Class
                    dmatrix_props = getattr(self.model, "_xgb_dmatrix_props", {})
                    X = xgboost.DMatrix(X, **dmatrix_props)
                phi = self.model.original_model.predict(
                    X,
                    iteration_range=(0, n_iterations),
                    pred_contribs=True,
                    approx_contribs=approximate,
                    validate_features=False,
                )
                if check_additivity and self.model.model_output == "raw":
                    model_output_vals = self.model.original_model.predict(
                        X, iteration_range=(0, n_iterations), output_margin=True, validate_features=False
                    )

            elif self.model.model_type == "lightgbm":
                assert not approximate, "approximate=True is not supported for LightGBM models!"
                phi = self.model.original_model.predict(X, num_iteration=tree_limit, pred_contrib=True)
                # Note: the data must be joined on the last axis
                if self.model.original_model.params["objective"] == "binary":
                    if not from_call:
                        warnings.warn(
                            "LightGBM binary classifier with TreeExplainer shap values output has changed to a list of ndarray"
                        )
                if phi.shape[1] != X.shape[1] + 1:
                    try:
                        phi = phi.reshape(X.shape[0], phi.shape[1] // (X.shape[1] + 1), X.shape[1] + 1)
                    except ValueError as e:
                        emsg = (
                            "This reshape error is often caused by passing a bad data matrix to SHAP. "
                            "See https://github.com/shap/shap/issues/580."
                        )
                        raise ValueError(emsg) from e

            elif self.model.model_type == "catboost":  # thanks to the CatBoost team for implementing this...
                assert not approximate, "approximate=True is not supported for CatBoost models!"
                assert tree_limit == -1, "tree_limit is not yet supported for CatBoost models!"
                import catboost

                if not isinstance(X, catboost.Pool):
                    X = catboost.Pool(X, cat_features=self.model.cat_feature_indices)
                phi = self.model.original_model.get_feature_importance(data=X, fstr_type="ShapValues")

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
                if isinstance(out, list):
                    out = np.stack(out, axis=-1)
                return out

        X, y, X_missing, flat_output, tree_limit, check_additivity = self._validate_inputs(
            X, y, tree_limit, check_additivity
        )
        transform = self.model.get_transform()
        _xgboost_cat_unsupported(self.model)

        # run the core algorithm using the C extension
        assert_import("cext")
        phi = np.zeros((X.shape[0], X.shape[1] + 1, self.model.num_outputs))

        if not approximate:
            _cext.dense_tree_shap(
                self.model.children_left,
                self.model.children_right,
                self.model.children_default,
                self.model.features,
                self.model.thresholds,
                self.model.values,
                self.model.node_sample_weight,
                self.model.max_depth,
                X,
                X_missing,
                y,
                self.data,
                self.data_missing,
                tree_limit,
                self.model.base_offset,
                phi,
                feature_perturbation_codes[self.feature_perturbation],
                output_transform_codes[transform],
                False,
            )
        else:
            _cext.dense_tree_saabas(
                self.model.children_left,
                self.model.children_right,
                self.model.children_default,
                self.model.features,
                self.model.thresholds,
                self.model.values,
                self.model.max_depth,
                tree_limit,
                self.model.base_offset,
                output_transform_codes[transform],
                X,
                X_missing,
                y,
                phi,
            )

        out = self._get_shap_output(phi, flat_output)
        if check_additivity and self.model.model_output == "raw":
            self.assert_additivity(out, self.model.predict(X))

        # This statements handles the case of multiple outputs
        # e.g. a multi-class classification problem, multi-target regression problem
        # in this case the output shape corresponds to [num_samples, num_features, num_outputs]
        if isinstance(out, list):
            out = np.stack(out, axis=-1)
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
        """Estimate the SHAP interaction values for a set of samples.

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
        np.array
            Returns a matrix. The shape depends on the number of model outputs:

            * one output: matrix of shape (#num_samples, #features, #features).
            * multiple outputs: matrix of shape (#num_samples, #features, #features, #num_outputs).

            The matrix (#num_samples, # features, # features) for each sample sums
            to the difference between the model output for that sample and the expected value of the model output
            (which is stored in the ``expected_value`` attribute of the explainer). Each row of this matrix sums to the
            SHAP value for that feature for that sample. The diagonal entries of the matrix represent the
            "main effect" of that feature on the prediction. The symmetric off-diagonal entries represent the
            interaction effects between all pairs of features for that sample.
            For models with vector outputs, this returns a list of tensors, one for each output.

            .. versionchanged:: 0.45.0
                Return type for models with multiple outputs changed from list to np.ndarray.

        """
        assert self.model.model_output == "raw", (
            'Only model_output = "raw" is supported for SHAP interaction values right now!'
        )
        # assert self.feature_perturbation == "tree_path_dependent", "Only feature_perturbation = \"tree_path_dependent\" is supported for SHAP interaction values right now!"
        transform = "identity"

        # see if we have a default tree_limit in place.
        if tree_limit is None:
            tree_limit = -1 if self.model.tree_limit is None else self.model.tree_limit

        # shortcut using the C++ version of Tree SHAP in XGBoost
        if self.model.model_type == "xgboost" and self.feature_perturbation == "tree_path_dependent":
            import xgboost

            if not isinstance(X, xgboost.core.DMatrix):
                X = xgboost.DMatrix(X)

            n_iterations = _xgboost_n_iterations(tree_limit, self.model.num_stacked_models)
            phi = self.model.original_model.predict(
                X, iteration_range=(0, n_iterations), pred_interactions=True, validate_features=False
            )

            # note we pull off the last column and keep it as our expected_value
            # multi-outputs
            if len(phi.shape) == 4:
                self.expected_value = [phi[0, i, -1, -1] for i in range(phi.shape[1])]
                # phi is given as [#num_observations, #num_classes, #features, #features]
                # slice out the expected values, then move the classes to the last dimension
                return np.swapaxes(phi[:, :, :-1, :-1], axis1=1, axis2=3)
            # regression and binary classification case
            else:
                self.expected_value = phi[0, -1, -1]
                return phi[:, :-1, :-1]
        elif (self.model.model_type == "catboost") and (
            self.feature_perturbation == "tree_path_dependent"
        ):  # thanks again to the CatBoost team for implementing this...
            assert tree_limit == -1, "tree_limit is not yet supported for CatBoost models!"
            import catboost

            if not isinstance(X, catboost.Pool):
                X = catboost.Pool(X, cat_features=self.model.cat_feature_indices)
            phi = self.model.original_model.get_feature_importance(data=X, fstr_type="ShapInteractionValues")
            # note we pull off the last column and keep it as our expected_value
            if len(phi.shape) == 4:
                self.expected_value = getattr(self, "expected_value", [phi[0, i, -1, -1] for i in range(phi.shape[1])])
                return [phi[:, i, :-1, :-1] for i in range(phi.shape[1])]
            else:
                self.expected_value = getattr(self, "expected_value", phi[0, -1, -1])
                return phi[:, :-1, :-1]

        X, y, X_missing, flat_output, tree_limit, _ = self._validate_inputs(X, y, tree_limit, False)
        # run the core algorithm using the C extension
        assert_import("cext")
        phi = np.zeros((X.shape[0], X.shape[1] + 1, X.shape[1] + 1, self.model.num_outputs))
        _cext.dense_tree_shap(
            self.model.children_left,
            self.model.children_right,
            self.model.children_default,
            self.model.features,
            self.model.thresholds,
            self.model.values,
            self.model.node_sample_weight,
            self.model.max_depth,
            X,
            X_missing,
            y,
            self.data,
            self.data_missing,
            tree_limit,
            self.model.base_offset,
            phi,
            feature_perturbation_codes[self.feature_perturbation],
            output_transform_codes[transform],
            True,
        )

        return self._get_shap_interactions_output(phi, flat_output)

    def _get_shap_interactions_output(self, phi, flat_output):
        """Pull off the last column and keep it as our expected_value"""
        if self.model.num_outputs == 1:
            # get expected value only if not already set
            self.expected_value = getattr(self, "expected_value", phi[0, -1, -1, 0])
            if flat_output:
                out = phi[0, :-1, :-1, 0]
            else:
                out = phi[:, :-1, :-1, 0]
        else:
            self.expected_value = [phi[0, -1, -1, i] for i in range(phi.shape[3])]
            if flat_output:
                out = np.stack([phi[0, :-1, :-1, i] for i in range(self.model.num_outputs)], axis=-1)
            else:
                out = np.stack([phi[:, :-1, :-1, i] for i in range(self.model.num_outputs)], axis=-1)
        return out

    def assert_additivity(self, phi, model_output):
        def check_sum(sum_val, model_output):
            diff = np.abs(sum_val - model_output)
            # TODO: add arguments for passing custom 'atol' and 'rtol' values to 'np.allclose'
            # would require change to interface i.e. '__call__' methods
            if not np.allclose(sum_val, model_output, atol=1e-2, rtol=1e-2):
                ind = np.argmax(diff)
                err_msg = (
                    "Additivity check failed in TreeExplainer! Please ensure the data matrix you passed to the "
                    "explainer is the same shape that the model was trained on. If your data shape is correct "
                    "then please report this on GitHub."
                )
                if self.feature_perturbation != "interventional":
                    err_msg += " Consider retrying with the feature_perturbation='interventional' option."
                err_msg += (
                    " This check failed because for one of the samples the sum of the SHAP values"
                    f" was {sum_val[ind]:f}, while the model output was {model_output[ind]:f}. If this"
                    " difference is acceptable you can set check_additivity=False to disable this check."
                )
                raise ExplainerError(err_msg)

        if isinstance(phi, list):
            for i in range(len(phi)):
                check_sum(self.expected_value[i] + phi[i].sum(-1), model_output[:, i])
        else:
            check_sum(self.expected_value + phi.sum(-1), model_output)

    @staticmethod
    def supports_model_with_masker(model, masker):
        """Determines if this explainer can handle the given model.

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
    """An ensemble of decision trees.

    This object provides a common interface to many different types of models.
    """

    def __init__(self, model, data=None, data_missing=None, model_output=None):
        self.model_type = "internal"
        self.trees = None
        self.base_offset = 0
        self.model_output = model_output
        self.objective = None  # what we explain when explaining the loss of the model
        self.tree_output = None  # what are the units of the values in the leaves of the trees
        self.internal_dtype = np.float64
        self.input_dtype = (
            np.float64
        )  # for sklearn we need to use np.float32 to always get exact matches to their predictions
        self.data = data
        self.data_missing = data_missing
        self.fully_defined_weighting = (
            True  # does the background dataset land in every leaf (making it valid for the tree_path_dependent method)
        )
        self.tree_limit = None  # used for limiting the number of trees we use by default (like from early stopping)
        self.num_stacked_models = 1  # If this is greater than 1 it means we have multiple stacked models with the same number of trees in each model (XGBoost multi-output style)
        self.cat_feature_indices = None  # If this is set it tells us which features are treated categorically

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

        if isinstance(model, dict) and "trees" in model:
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
        elif isinstance(model, list) and isinstance(model[0], SingleTree):  # old-style direct-load format
            self.trees = model
        elif safe_isinstance(
            model,
            [
                "sklearn.ensemble.RandomForestRegressor",
                "sklearn.ensemble.forest.RandomForestRegressor",
                "econml.grf._base_grf.BaseGRF",
                "causalml.inference.tree.CausalRandomForestRegressor",
            ],
        ):
            assert hasattr(model, "estimators_"), "Model has no `estimators_`! Have you called `model.fit`?"
            self.internal_dtype = model.estimators_[0].tree_.value.dtype.type
            self.input_dtype = np.float32
            scaling = 1.0 / len(model.estimators_)  # output is average of trees
            self.trees = [
                SingleTree(e.tree_, scaling=scaling, data=data, data_missing=data_missing) for e in model.estimators_
            ]
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
            scaling = 1.0 / len(model.estimators_)  # output is average of trees
            self.trees = [
                IsoTree(e.tree_, f, scaling=scaling, data=data, data_missing=data_missing)
                for e, f in zip(model.estimators_, model.estimators_features_)
            ]
            self.tree_output = "raw_value"
        elif safe_isinstance(model, ["pyod.models.iforest.IForest"]):
            self.dtype = np.float32
            scaling = 1.0 / len(model.estimators_)  # output is average of trees
            self.trees = [
                IsoTree(e.tree_, f, scaling=scaling, data=data, data_missing=data_missing)
                for e, f in zip(model.detector_.estimators_, model.detector_.estimators_features_)
            ]
            self.tree_output = "raw_value"
        elif safe_isinstance(
            model,
            [
                "sklearn.ensemble.ExtraTreesRegressor",
                "sklearn.ensemble.forest.ExtraTreesRegressor",
                "skopt.learning.forest.RandomForestRegressor",
                "skopt.learning.forest.ExtraTreesRegressor",
            ],
        ):
            assert hasattr(model, "estimators_"), "Model has no `estimators_`! Have you called `model.fit`?"
            self.internal_dtype = model.estimators_[0].tree_.value.dtype.type
            self.input_dtype = np.float32
            scaling = 1.0 / len(model.estimators_)  # output is average of trees
            self.trees = [
                SingleTree(e.tree_, scaling=scaling, data=data, data_missing=data_missing) for e in model.estimators_
            ]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "raw_value"
        elif safe_isinstance(
            model,
            [
                "sklearn.tree.DecisionTreeRegressor",
                "sklearn.tree.tree.DecisionTreeRegressor",
                "econml.grf._base_grftree.GRFTree",
                "causalml.inference.tree.causal.causaltree.CausalTreeRegressor",
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
                "sklearn.ensemble.ExtraTreesClassifier",
                "sklearn.ensemble.forest.ExtraTreesClassifier",
                "sklearn.ensemble.RandomForestClassifier",
                "sklearn.ensemble.forest.RandomForestClassifier",
            ],
        ):
            assert hasattr(model, "estimators_"), "Model has no `estimators_`! Have you called `model.fit`?"
            self.internal_dtype = model.estimators_[0].tree_.value.dtype.type
            self.input_dtype = np.float32
            scaling = 1.0 / len(model.estimators_)  # output is average of trees
            self.trees = [
                SingleTree(e.tree_, normalize=True, scaling=scaling, data=data, data_missing=data_missing)
                for e in model.estimators_
            ]
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

            self.trees = [
                SingleTree(e.tree_, scaling=model.learning_rate, data=data, data_missing=data_missing)
                for e in model.estimators_[:, 0]
            ]
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
                    'model_output="raw". See GitHub issue #1028.'
                )
                raise NotImplementedError(emsg)
            self.input_dtype = sklearn.ensemble._hist_gradient_boosting.common.X_DTYPE
            self.num_stacked_models = len(model._predictors[0])
            if self.model_output == "predict_proba":
                if self.num_stacked_models == 1:
                    self.model_output = (
                        "probability_doubled"  # with predict_proba we need to double the outputs to match
                    )
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
                emsg = "GradientBoostingClassifier is only supported for binary classification right now!"
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
                self.base_offset = scipy.special.logit(
                    model.init_.class_prior_[1]
                )  # with two classes the trees only model the second class.
                self.tree_output = "log_odds"
            else:
                emsg = f"Unsupported init model type: {type(model.init_)}"
                raise InvalidModelError(emsg)

            self.trees = [
                SingleTree(e.tree_, scaling=model.learning_rate, data=data, data_missing=data_missing)
                for e in model.estimators_[:, 0]
            ]
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
            self._set_xgboost_model_attributes(
                data,
                data_missing,
                objective_name_map,
                tree_output_name_map,
            )
        elif safe_isinstance(model, "xgboost.sklearn.XGBClassifier"):
            self.input_dtype = np.float32
            self.original_model = model.get_booster()
            self._set_xgboost_model_attributes(
                data,
                data_missing,
                objective_name_map,
                tree_output_name_map,
            )

            if self.model_output == "predict_proba":
                if self.num_stacked_models == 1:
                    # with predict_proba we need to double the outputs to match
                    self.model_output = "probability_doubled"
                else:
                    self.model_output = "probability"
            # Some properties of the sklearn API are passed to a DMatrix object in
            # xgboost We need to make sure we do the same here - GH #3313
            self._xgb_dmatrix_props = get_xgboost_dmatrix_properties(model)
        elif safe_isinstance(model, ["xgboost.sklearn.XGBRegressor", "xgboost.sklearn.XGBRanker"]):
            self.original_model = model.get_booster()
            self._set_xgboost_model_attributes(
                data,
                data_missing,
                objective_name_map,
                tree_output_name_map,
            )
            # Some properties of the sklearn API are passed to a DMatrix object in
            # xgboost We need to make sure we do the same here - GH #3313
            self._xgb_dmatrix_props = get_xgboost_dmatrix_properties(model)
        elif safe_isinstance(model, "lightgbm.basic.Booster"):
            assert_import("lightgbm")
            self.model_type = "lightgbm"
            self.original_model = model
            tree_info = self.original_model.dump_model()["tree_info"]
            try:
                self.trees = [SingleTree(e, data=data, data_missing=data_missing) for e in tree_info]
            except Exception:
                self.trees = None  # we get here because the cext can't handle categorical splits yet

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
                self.trees = None  # we get here because the cext can't handle categorical splits yet

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
                self.trees = None  # we get here because the cext can't handle categorical splits yet
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
                self.trees = None  # we get here because the cext can't handle categorical splits yet
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
                self.trees = None  # we get here because the cext can't handle categorical splits yet
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
            try:
                cb_loader = CatBoostTreeModelLoader(model)
                self.trees = cb_loader.get_trees(data=data, data_missing=data_missing)
            except Exception:
                self.trees = None  # we get here because the cext can't handle categorical splits yet
        elif safe_isinstance(model, "catboost.core.CatBoostClassifier"):
            assert_import("catboost")
            self.model_type = "catboost"
            self.original_model = model
            self.input_dtype = np.float32
            try:
                cb_loader = CatBoostTreeModelLoader(model)
                self.trees = cb_loader.get_trees(data=data, data_missing=data_missing)
            except Exception:
                self.trees = None  # we get here because the cext can't handle categorical splits yet
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
            scaling = 1.0 / len(model.estimators_)  # output is average of trees
            self.trees = [
                SingleTree(e.tree_, normalize=True, scaling=scaling, data=data, data_missing=data_missing)
                for e in model.estimators_
            ]
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
                param_idx = 0  # default to the first parameter of the output distribution
                warnings.warn(
                    'Translating model_output="raw" to model_output=0 for the 0-th parameter in the distribution. Use model_output=0 directly to avoid this warning.'
                )
            elif isinstance(self.model_output, int):
                param_idx = self.model_output
                self.model_output = "raw"  # note that after loading we have a new model_output type
            assert safe_isinstance(
                model.base_models[0][param_idx],
                ["sklearn.tree.DecisionTreeRegressor", "sklearn.tree.tree.DecisionTreeRegressor"],
            ), "You must use default_tree_learner!"
            shap_trees = [trees[param_idx] for trees in model.base_models]
            self.internal_dtype = shap_trees[0].tree_.value.dtype.type
            self.input_dtype = np.float32
            scaling = -model.learning_rate * np.array(model.scalings)  # output is weighted average of trees
            # ngboost reorders the features, so we need to map them back to the original order
            missing_col_idxs = [[i for i in range(model.n_features) if i not in col_idx] for col_idx in model.col_idxs]
            feature_mapping = [
                {i: col_idx for i, col_idx in enumerate(list(col_idxs) + missing_col_idx)}
                for col_idxs, missing_col_idx in zip(model.col_idxs, missing_col_idxs)
            ]
            self.trees = []
            for idx, shap_tree in enumerate(shap_trees):
                tree_ = shap_tree.tree_
                values = tree_.value.reshape(tree_.value.shape[0], tree_.value.shape[1] * tree_.value.shape[2])
                values = values * scaling[idx]
                tree = {
                    "children_left": tree_.children_left.astype(np.int32),
                    "children_right": tree_.children_right.astype(np.int32),
                    "children_default": tree_.children_left,
                    "features": np.array([feature_mapping[idx].get(i, i) for i in tree_.feature]),
                    "thresholds": tree_.threshold.astype(np.float64),
                    "values": values,
                    "node_sample_weight": tree_.weighted_n_node_samples.astype(np.float64),
                }
                self.trees.append(SingleTree(tree, data=data, data_missing=data_missing))
            self.objective = objective_name_map.get(shap_trees[0].criterion, None)
            self.tree_output = "raw_value"
            self.base_offset = model.init_params[param_idx]
        else:
            raise InvalidModelError("Model type not yet supported by TreeExplainer: " + str(type(model)))

        # build a dense numpy version of all the tree objects
        if self.trees is not None and self.trees:
            max_nodes = np.max([len(t.values) for t in self.trees])
            assert len(np.unique([t.values.shape[1] for t in self.trees])) == 1, (
                "All trees in the ensemble must have the same output dimension!"
            )
            num_trees = len(self.trees)
            # important to be -1 in unused sections!! This way we can tell which entries are valid.
            self.children_left = -np.ones((num_trees, max_nodes), dtype=np.int32)
            self.children_right = -np.ones((num_trees, max_nodes), dtype=np.int32)
            self.children_default = -np.ones((num_trees, max_nodes), dtype=np.int32)
            self.features = -np.ones((num_trees, max_nodes), dtype=np.int32)

            self.thresholds = np.zeros((num_trees, max_nodes), dtype=self.internal_dtype)
            self.values = np.zeros((num_trees, max_nodes, self.num_outputs), dtype=self.internal_dtype)
            self.node_sample_weight = np.zeros((num_trees, max_nodes), dtype=self.internal_dtype)

            for i in range(num_trees):
                self.children_left[i, : len(self.trees[i].children_left)] = self.trees[i].children_left
                self.children_right[i, : len(self.trees[i].children_right)] = self.trees[i].children_right
                self.children_default[i, : len(self.trees[i].children_default)] = self.trees[i].children_default
                self.features[i, : len(self.trees[i].features)] = self.trees[i].features
                self.thresholds[i, : len(self.trees[i].thresholds)] = self.trees[i].thresholds

                # XGBoost supports boosting forest, which is not compatible with the
                # current assumption here that the number of stacked models represents
                # the number of outputs.
                if self.model_type == "xgboost":
                    n_stacks = self.num_outputs
                else:
                    n_stacks = self.num_stacked_models

                if n_stacks > 1:
                    stack_pos = i % n_stacks
                    self.values[i, : len(self.trees[i].values[:, 0]), stack_pos] = self.trees[i].values[:, 0]
                else:
                    self.values[i, : len(self.trees[i].values)] = self.trees[i].values
                self.node_sample_weight[i, : len(self.trees[i].node_sample_weight)] = self.trees[i].node_sample_weight

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

    def _set_xgboost_model_attributes(
        self,
        data,
        data_missing,
        objective_name_map,
        tree_output_name_map,
    ):
        self.model_type = "xgboost"
        loader = XGBTreeModelLoader(self.original_model)

        self.trees = loader.get_trees(data=data, data_missing=data_missing)
        self.base_offset = loader.base_score
        self.objective = objective_name_map.get(loader.name_obj, None)
        self.tree_output = tree_output_name_map.get(loader.name_obj, None)

        self.num_stacked_models = loader.n_trees_per_iter
        self.cat_feature_indices = loader.cat_feature_indices
        best_iteration = getattr(
            self.original_model,
            "best_iteration",
            self.original_model.num_boosted_rounds() - 1,
        )
        self.tree_limit = (best_iteration + 1) * self.num_stacked_models
        self._xgboost_n_outputs = loader.n_targets

    @property
    def num_outputs(self) -> int:
        # Currrently, XGBoost models derive the num_outputs attribute from the input
        # models, which is set during model load.
        if self.model_type == "xgboost":
            assert hasattr(self, "_xgboost_n_outputs")
            return self._xgboost_n_outputs

        if self.num_stacked_models > 1:
            if len(self.trees) % self.num_stacked_models != 0:
                raise ValueError("Only stacked models with equal numbers of trees are supported!")
            if self.trees[0].values.shape[1] != 1:
                raise ValueError("Only stacked models with single outputs per model are supported!")
            return self.num_stacked_models
        else:
            return self.trees[0].values.shape[1]

    def get_transform(self):
        """A consistent interface to make predictions from this model."""
        if self.model_output == "raw":
            transform = "identity"
        elif self.model_output in ("probability", "probability_doubled"):
            if self.tree_output == "log_odds":
                transform = "logistic"
            elif self.tree_output == "probability":
                transform = "identity"
            else:
                emsg = (
                    f'model_output = "probability" is not yet supported when model.tree_output = "{self.tree_output}"!'
                )
                raise NotImplementedError(emsg)
        elif self.model_output == "log_loss":
            if self.objective == "squared_error":
                transform = "squared_loss"
            elif self.objective == "binary_crossentropy":
                transform = "logistic_nlogloss"
            else:
                emsg = f'model_output = "log_loss" is not yet supported when model.objective = "{self.objective}"!'
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
        """A consistent interface to make predictions from this model.

        Parameters
        ----------
        tree_limit : None (default) or int
            Limit the number of trees used by the model. By default None means no use the limit of the
            original model, and -1 means no limit.

        """
        if output is None:
            output = self.model_output

        if self.model_type == "pyspark":
            # import pyspark
            # TODO: support predict for pyspark
            raise NotImplementedError(
                "Predict with pyspark isn't implemented. Don't run 'interventional' as feature_perturbation."
            )
        if self.model_type == "xgboost" and self.num_stacked_models != self.num_outputs:
            # TODO: Support random forest in XGBoost.
            raise NotImplementedError("XGBoost with boosted random forest is not yet supported.")

        # see if we have a default tree_limit in place.
        if tree_limit is None:
            tree_limit = -1 if self.tree_limit is None else self.tree_limit

        # convert dataframes
        if isinstance(X, (pd.Series, pd.DataFrame)):
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
            if y is None:
                raise ValueError(
                    "Both samples and labels must be provided when explaining the loss"
                    " (i.e. `explainer.shap_values(X, y)`)!"
                )
            if X.shape[0] != len(y):
                raise ValueError(
                    f"The number of labels ({len(y)}) does not match the number of samples to explain ({X.shape[0]})!"
                )

        transform = self.get_transform()
        assert_import("cext")
        output = np.zeros((X.shape[0], self.num_outputs))
        _cext.dense_tree_predict(
            self.children_left,
            self.children_right,
            self.children_default,
            self.features,
            self.thresholds,
            self.values,
            self.max_depth,
            tree_limit,
            self.base_offset,
            output_transform_codes[transform],
            X,
            X_missing,
            y,
            output,
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

        if safe_isinstance(
            tree,
            [
                "sklearn.tree._tree.Tree",
                "econml.tree._tree.Tree",
                "causalml.inference.tree._tree._tree.Tree",
            ],
        ):
            self.children_left = tree.children_left.astype(np.int32)
            self.children_right = tree.children_right.astype(np.int32)
            self.children_default = self.children_left
            if hasattr(tree, "missing_go_to_left"):
                self.children_default = np.where(tree.missing_go_to_left, self.children_left, self.children_right)
            self.features = tree.feature.astype(np.int32)
            self.thresholds = tree.threshold.astype(np.float64)
            self.values = tree.value.reshape(tree.value.shape[0], tree.value.shape[1] * tree.value.shape[2])
            if normalize:
                self.values = (self.values.T / self.values.sum(1)).T
            self.values = self.values * scaling
            self.node_sample_weight = tree.weighted_n_node_samples.astype(np.float64)

        elif isinstance(tree, dict) and "features" in tree:
            self.children_left = tree["children_left"].astype(np.int32)
            self.children_right = tree["children_right"].astype(np.int32)
            self.children_default = tree["children_default"].astype(np.int32)
            self.features = tree["features"].astype(np.int32)
            self.thresholds = tree["thresholds"]
            self.values = tree["values"] * scaling
            self.node_sample_weight = tree["node_sample_weight"]

        # deprecated dictionary support (with sklearn singular style "feature" and "value" names)
        elif isinstance(tree, dict) and "children_left" in tree:
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
            # model._java_obj.numNodes() doesn't give leaves, need to recompute the size
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
            self.values = [-2] * num_nodes
            self.node_sample_weight = np.full(num_nodes, -2, dtype=np.float64)

            def buildTree(index, node):
                index = index + 1
                if tree._java_obj.getImpurity() == "variance":
                    self.values[index] = [node.prediction()]  # prediction for the node
                else:
                    self.values[index] = [
                        e for e in node.impurityStats().stats()
                    ]  # for gini: NDarray(numLabel): 1 per label: number of item for each label which went through this node
                self.node_sample_weight[index] = (
                    node.impurityStats().count()
                )  # weighted count of element through this node

                if node.subtreeDepth() == 0:
                    return index
                else:
                    self.features[index] = (
                        node.split().featureIndex()
                    )  # index of the feature we split on, not available for leaf, int
                    if str(node.split().getClass()).endswith("tree.CategoricalSplit"):
                        # Categorical split isn't implemented, TODO: could fake it by creating a fake node to split on the exact value?
                        raise NotImplementedError("CategoricalSplit are not yet implemented")
                    self.thresholds[index] = (
                        node.split().threshold()
                    )  # threshold for the feature, not available for leaf, float

                    self.children_left[index] = index + 1
                    idx = buildTree(index, node.leftChild())
                    self.children_right[index] = idx + 1
                    idx = buildTree(idx, node.rightChild())
                    return idx

            buildTree(-1, tree._java_obj.rootNode())
            # default Not supported with mlib? (TODO)
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
                    # NOTE: If "leaf_index" is not present as a key, it means we have a
                    # stump tree. I.e., num_nodes=1.
                    vleaf_idx: int = vertex.get("leaf_index", 0) + num_parents
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
                    # FIXME: "leaf_count" currently doesn't exist if we have a stump tree.
                    # We should be technically be assigning the number of samples used to
                    # train the model as the weight here, but unfortunately this info is
                    # currently unavailable in `tree`, so we set to 0 first.
                    # cf. https://github.com/microsoft/LightGBM/issues/5962
                    self.node_sample_weight[vleaf_idx] = vertex.get("leaf_count", 0)
            self.values = np.asarray(self.values)
            self.values = np.multiply(self.values, scaling)

        elif isinstance(tree, dict) and "nodeid" in tree:
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

        elif isinstance(tree, str):
            """ Build a tree from a text dump (with stats) of xgboost.
            """

            nodes = [t.lstrip() for t in tree[:-1].split("\n")]
            nodes_dict = {}
            for n in nodes:
                nodes_dict[int(n.split(":")[0])] = n.split(":")[1]
            m = max(nodes_dict.keys()) + 1
            children_left = -1 * np.ones(m, dtype="int32")
            children_right = -1 * np.ones(m, dtype="int32")
            children_default = -1 * np.ones(m, dtype="int32")
            features = -2 * np.ones(m, dtype="int32")
            thresholds = -1 * np.ones(m, dtype="float64")
            values = 1 * np.ones(m, dtype="float64")
            node_sample_weight = np.zeros(m, dtype="float64")
            values_lst = list(nodes_dict.values())
            keys_lst = list(nodes_dict.keys())
            for i in range(len(keys_lst)):
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
                    if "<" in feat_thres:
                        feature = int(feat_thres.split("<")[0][2:])
                        threshold = float(feat_thres.split("<")[1][:-1])
                    if "=" in feat_thres:
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
            self.values = values[:, np.newaxis] * scaling
            self.node_sample_weight = node_sample_weight
        else:
            raise TypeError("Unknown input to SingleTree constructor: " + str(tree))

        # Re-compute the number of samples that pass through each node if we are given data
        if data is not None and data_missing is not None:
            self.node_sample_weight.fill(0.0)
            _cext.dense_tree_update_weights(
                self.children_left,
                self.children_right,
                self.children_default,
                self.features,
                self.thresholds,
                self.values,
                1,
                self.node_sample_weight,
                data,
                data_missing,
            )

        # we compute the expectations to make sure they follow the SHAP logic
        self.max_depth = _cext.compute_expectations(
            self.children_left, self.children_right, self.node_sample_weight, self.values
        )


class IsoTree(SingleTree):
    """In sklearn the tree of the Isolation Forest does not calculated in a good way."""

    def __init__(self, tree, tree_features, normalize=False, scaling=1.0, data=None, data_missing=None):
        super().__init__(tree, normalize, scaling, data, data_missing)
        if safe_isinstance(tree, "sklearn.tree._tree.Tree"):
            from sklearn.ensemble._iforest import _average_path_length

            def _recalculate_value(tree, i, level):
                if tree.children_left[i] == -1 and tree.children_right[i] == -1:
                    value = level + _average_path_length(np.array([tree.n_node_samples[i]]))[0]
                    self.values[i, 0] = value
                    return value * tree.n_node_samples[i]
                else:
                    value_left = _recalculate_value(tree, tree.children_left[i], level + 1)
                    value_right = _recalculate_value(tree, tree.children_right[i], level + 1)
                    self.values[i, 0] = (value_left + value_right) / tree.n_node_samples[i]
                    return value_left + value_right

            _recalculate_value(tree, 0, 0)
            if normalize:
                self.values = (self.values.T / self.values.sum(1)).T
            self.values = self.values * scaling
            # re-number the features if each tree gets a different set of features
            self.features = np.where(self.features >= 0, tree_features[self.features], self.features)


def get_xgboost_dmatrix_properties(model):
    """Retrieves properties from an xgboost.sklearn.XGBModel instance that should be
    passed to the xgboost.core.DMatrix object before calling predict on the model.

    """
    properties_to_pass = ["missing", "n_jobs", "enable_categorical", "feature_types"]
    dmatrix_attributes = {}
    for attribute in properties_to_pass:
        if hasattr(model, attribute):
            dmatrix_attributes[attribute] = getattr(model, attribute)

    # Convert sklearn n_jobs to xgboost nthread
    if "n_jobs" in dmatrix_attributes:
        dmatrix_attributes["nthread"] = dmatrix_attributes.pop("n_jobs")
    return dmatrix_attributes


class XGBTreeModelLoader:
    """This loads an XGBoost model directly from a raw memory dump."""

    def __init__(self, xgb_model) -> None:
        import xgboost as xgb

        _check_xgboost_version(xgb.__version__)
        model: xgb.Booster = xgb_model

        raw = xgb_model.save_raw(raw_format="ubj")
        with io.BytesIO(raw) as fd:
            jmodel = decode_ubjson_buffer(fd)

        learner = jmodel["learner"]
        learner_model_param = learner["learner_model_param"]
        objective = learner["objective"]

        booster = learner["gradient_booster"]
        n_classes = max(int(learner_model_param["num_class"]), 1)
        n_targets = max(int(learner_model_param["num_target"]), 1)
        n_targets = max(n_targets, n_classes)

        # darts booster does not have the standard format.
        # Therefore we need to unpack the gbtree key.
        if "gbtree" in booster and "model" not in booster:
            booster = booster["gbtree"]
        # Check the input model doesn't have vector-leaf
        if booster["model"].get("iteration_indptr", None) is not None:
            # iteration_indptr was introduced in 2.0.
            iteration_indptr = np.asarray(booster["model"]["iteration_indptr"], dtype=np.int32)
            diff = np.diff(iteration_indptr)
        else:
            n_parallel_trees = int(booster["model"]["gbtree_model_param"]["num_parallel_tree"])
            diff = np.repeat(n_targets * n_parallel_trees, model.num_boosted_rounds())
        if np.any(diff != diff[0]):
            raise ValueError("vector-leaf is not yet supported.:", diff)

        # used to convert the number of iteration to the number of trees.
        # Accounts for number of classes, targets, forest size.
        self.n_trees_per_iter = int(diff[0])
        self.n_targets = n_targets
        self.base_score = float(learner_model_param["base_score"])
        assert self.n_trees_per_iter > 0

        self.name_obj = objective["name"]
        self.name_gbm = booster["name"]
        # handle the link function.
        base_score = float(learner_model_param["base_score"])
        if self.name_obj in ("binary:logistic", "reg:logistic"):
            self.base_score = scipy.special.logit(base_score)
        elif self.name_obj in (
            "reg:gamma",
            "reg:tweedie",
            "count:poisson",
            "survival:cox",
            "survival:aft",
        ):
            # exp family
            self.base_score = np.log(self.base_score)
        else:
            self.base_score = base_score

        self.num_feature = int(learner_model_param["num_feature"])
        self.num_class = int(learner_model_param["num_class"])

        trees = booster["model"]["trees"]
        self.num_trees = len(trees)

        self.node_parents = []
        self.node_cleft = []
        self.node_cright = []
        self.node_sindex = []
        self.children_default: list[np.ndarray] = []
        self.sum_hess = []

        self.values = []
        self.thresholds = []
        self.features = []

        # Categorical features, not supported by the SHAP package yet.
        self.split_types = []
        self.categories = []

        feature_types = model.feature_types
        if feature_types is not None:
            cat_feature_indices: np.ndarray = np.where(np.asarray(feature_types) == "c")[0]
            if len(cat_feature_indices) == 0:
                self.cat_feature_indices: np.ndarray | None = None
            else:
                self.cat_feature_indices = cat_feature_indices
        else:
            self.cat_feature_indices = None

        def to_integers(data: list[int]) -> np.ndarray:
            """Handle u8 array from UBJSON."""
            assert isinstance(data, list)
            return np.asanyarray(data, dtype=np.uint8)

        for i in range(self.num_trees):
            tree = trees[i]
            parents = np.asarray(tree["parents"])
            self.node_parents.append(parents)
            self.node_cleft.append(np.asarray(tree["left_children"], dtype=np.int32))
            self.node_cright.append(np.asarray(tree["right_children"], dtype=np.int32))
            self.node_sindex.append(np.asarray(tree["split_indices"], dtype=np.uint32))

            base_weight = np.asarray(tree["base_weights"], dtype=np.float32)
            if base_weight.size != self.node_cleft[-1].size:
                raise ValueError("vector-leaf is not yet supported.")

            default_left = to_integers(tree["default_left"])
            default_child = np.where(default_left == 1, self.node_cleft[-1], self.node_cright[-1]).astype(np.int64)
            self.children_default.append(default_child)
            self.sum_hess.append(np.asarray(tree["sum_hessian"], dtype=np.float64))

            is_leaf = self.node_cleft[-1] == -1

            # XGBoost stores split condition and leaf weight in the same field.
            split_cond = np.asarray(tree["split_conditions"], dtype=np.float32)
            leaf_weight = np.where(is_leaf, split_cond, 0.0)
            thresholds = np.where(is_leaf, 0.0, split_cond)

            # Xgboost uses < for thresholds where shap uses <= Move the threshold down
            # by the smallest possible increment
            thresholds = np.where(is_leaf, 0.0, np.nextafter(thresholds, -np.float32(np.inf)))

            self.values.append(leaf_weight.reshape(leaf_weight.size, 1))
            self.thresholds.append(thresholds)

            split_idx = np.asarray(tree["split_indices"], dtype=np.int64)
            self.features.append(split_idx)

            # - categorical features
            # when ubjson is used, this is a byte array with each element as uint8
            split_types = to_integers(tree["split_type"])
            self.split_types.append(split_types)
            # categories for each node is stored in a CSR style storage with segment as
            # the begin ptr and the `categories' as values.
            cat_segments: list[int] = tree["categories_segments"]
            cat_sizes: list[int] = tree["categories_sizes"]
            # node index for categorical nodes
            cat_nodes: list[int] = tree["categories_nodes"]
            assert len(cat_segments) == len(cat_sizes) == len(cat_nodes)
            cats = tree["categories"]

            tree_categories = self.parse_categories(cat_nodes, cat_segments, cat_sizes, cats, self.node_cleft[-1])
            self.categories.append(tree_categories)

    @staticmethod
    def parse_categories(
        cat_nodes: list[int],
        cat_segments: list[int],
        cat_sizes: list[int],
        cats: list[int],
        left_children: np.ndarray,
    ) -> list[list[int]]:
        """Parse the JSON model to extract partitions of categories for each
        node. Returns a list, in which each element is a list of categories for tree
        split. For a numerical split, the list is empty.

        This is not used yet, only implemented for future reference.

        """
        # The storage for categories is only defined for categorical nodes to prevent
        # unnecessary overhead for numerical splits, we track the categorical node that
        # are processed using a counter.
        cat_cnt = 0
        if cat_nodes:
            last_cat_node = cat_nodes[cat_cnt]
        else:
            last_cat_node = -1
        node_categories: list[list[int]] = []
        for node_id in range(len(left_children)):
            if node_id == last_cat_node:
                beg = cat_segments[cat_cnt]
                size = cat_sizes[cat_cnt]
                end = beg + size
                # categories for this node
                node_cats = cats[beg:end]
                # categories are unique for each node
                assert len(set(node_cats)) == len(node_cats)
                cat_cnt += 1
                if cat_cnt == len(cat_nodes):
                    last_cat_node = -1  # continue to process the rest of the nodes
                else:
                    last_cat_node = cat_nodes[cat_cnt]
                assert node_cats
                node_categories.append(node_cats)
            else:
                # append an empty node, it's either a numerical node or a leaf.
                node_categories.append([])
        return node_categories

    def get_trees(self, data=None, data_missing=None) -> list[SingleTree]:
        trees = []
        for i in range(self.num_trees):
            info = {
                "children_left": self.node_cleft[i],
                "children_right": self.node_cright[i],
                "children_default": self.children_default[i],
                "feature": self.features[i],
                "threshold": self.thresholds[i],
                "value": self.values[i],
                "node_sample_weight": self.sum_hess[i],
            }
            trees.append(SingleTree(info, data=data, data_missing=data_missing))
        return trees

    def print_info(self) -> None:
        print("--- global parameters ---")
        print("base_score =", self.base_score)
        print("num_feature =", self.num_feature)
        print("num_class =", self.num_class)
        print("name_obj =", self.name_obj)
        print("name_gbm =", self.name_gbm)
        print()
        print("--- gbtree specific parameters ---")
        print("num_feature =", self.num_feature)


class CatBoostTreeModelLoader:
    def __init__(self, cb_model):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_file = os.path.join(tmp_dir, "model.json")
            cb_model.save_model(tmp_file, format="json")
            with open(tmp_file, encoding="utf-8") as fh:
                self.loaded_cb_model = json.load(fh)

        # load the CatBoost oblivious trees specific parameters
        self.num_trees = len(self.loaded_cb_model["oblivious_trees"])
        self.max_depth = self.loaded_cb_model["model_info"]["params"]["tree_learner_options"]["depth"]

    def get_trees(self, data=None, data_missing=None):
        # load each tree
        trees = []
        for tree_index in range(self.num_trees):
            # load the per-tree params
            # depth = len(self.loaded_cb_model['oblivious_trees'][tree_index]['splits'])

            # load the nodes

            # Re-compute the number of samples that pass through each node if we are given data
            leaf_weights = self.loaded_cb_model["oblivious_trees"][tree_index]["leaf_weights"]
            leaf_weights_unraveled = [0] * (len(leaf_weights) - 1) + leaf_weights
            leaf_weights_unraveled[0] = sum(leaf_weights)
            for index in range(len(leaf_weights) - 2, 0, -1):
                leaf_weights_unraveled[index] = (
                    leaf_weights_unraveled[2 * index + 1] + leaf_weights_unraveled[2 * index + 2]
                )

            leaf_values = self.loaded_cb_model["oblivious_trees"][tree_index]["leaf_values"]
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
            for elem in self.loaded_cb_model["oblivious_trees"][tree_index]["splits"]:
                split_type = elem.get("split_type")
                if split_type == "FloatFeature":
                    split_feature_index = elem.get("float_feature_index")
                    borders.append(elem["border"])
                elif split_type == "OneHotFeature":
                    split_feature_index = elem.get("cat_feature_index")
                    borders.append(elem["value"])
                else:
                    split_feature_index = elem.get("ctr_target_border_idx")
                    borders.append(elem["border"])
                split_features_index.append(split_feature_index)

            split_features_index_unraveled = []
            for counter, feature_index in enumerate(split_features_index[::-1]):
                split_features_index_unraveled += [feature_index] * (2**counter)
            split_features_index_unraveled += [0] * len(leaf_values)

            borders_unraveled = []
            for counter, border in enumerate(borders[::-1]):
                borders_unraveled += [border] * (2**counter)
            borders_unraveled += [0] * len(leaf_values)

            trees.append(
                SingleTree(
                    {
                        "children_left": np.array(children_left),
                        "children_right": np.array(children_right),
                        "children_default": np.array(children_default),
                        "feature": np.array(split_features_index_unraveled),
                        "threshold": np.array(borders_unraveled),
                        "value": np.array(leaf_values_unraveled).reshape((-1, 1)),
                        "node_sample_weight": np.array(leaf_weights_unraveled),
                    },
                    data=data,
                    data_missing=data_missing,
                )
            )

        return trees

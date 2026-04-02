"""Scikit-learn tree model support (and compatible libraries like econml, skopt)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import scipy.special

from ...utils import safe_isinstance
from ...utils._exceptions import InvalidModelError
from .._tree import IsoTree, SingleTree
from ._base import OBJECTIVE_NAME_MAP, TreeModelHandler

if TYPE_CHECKING:
    from .._tree import TreeEnsemble


class SklearnHandler(TreeModelHandler):
    """Handles all scikit-learn tree-based models.

    Includes sklearn's own models as well as compatible libraries that
    use sklearn's tree structure (econml, causalml, skopt).
    """

    @staticmethod
    def can_handle(model: Any) -> bool:
        return safe_isinstance(
            model,
            [
                # RandomForest Regressor
                "sklearn.ensemble.RandomForestRegressor",
                "sklearn.ensemble.forest.RandomForestRegressor",
                "econml.grf._base_grf.BaseGRF",
                "causalml.inference.tree.CausalRandomForestRegressor",
                # IsolationForest
                "sklearn.ensemble.IsolationForest",
                "sklearn.ensemble._iforest.IsolationForest",
                # ExtraTrees Regressor (+ skopt)
                "sklearn.ensemble.ExtraTreesRegressor",
                "sklearn.ensemble.forest.ExtraTreesRegressor",
                "skopt.learning.forest.RandomForestRegressor",
                "skopt.learning.forest.ExtraTreesRegressor",
                # DecisionTree Regressor
                "sklearn.tree.DecisionTreeRegressor",
                "sklearn.tree.tree.DecisionTreeRegressor",
                "econml.grf._base_grftree.GRFTree",
                "causalml.inference.tree.causal.causaltree.CausalTreeRegressor",
                # DecisionTree Classifier
                "sklearn.tree.DecisionTreeClassifier",
                "sklearn.tree.tree.DecisionTreeClassifier",
                # ExtraTrees / RandomForest Classifier
                "sklearn.ensemble.ExtraTreesClassifier",
                "sklearn.ensemble.forest.ExtraTreesClassifier",
                "sklearn.ensemble.RandomForestClassifier",
                "sklearn.ensemble.forest.RandomForestClassifier",
                # GradientBoosting Regressor
                "sklearn.ensemble.GradientBoostingRegressor",
                "sklearn.ensemble.gradient_boosting.GradientBoostingRegressor",
                # HistGradientBoosting Regressor
                "sklearn.ensemble.HistGradientBoostingRegressor",
                # HistGradientBoosting Classifier
                "sklearn.ensemble.HistGradientBoostingClassifier",
                # GradientBoosting Classifier
                "sklearn.ensemble.GradientBoostingClassifier",
                "sklearn.ensemble._gb.GradientBoostingClassifier",
                "sklearn.ensemble.gradient_boosting.GradientBoostingClassifier",
            ],
        )

    @staticmethod
    def handle(
        model: Any,
        ensemble: TreeEnsemble,
        data: npt.NDArray[Any] | None,
        data_missing: npt.NDArray[np.bool_] | None,
    ) -> None:
        # --- RandomForest Regressor ---
        if safe_isinstance(
            model,
            [
                "sklearn.ensemble.RandomForestRegressor",
                "sklearn.ensemble.forest.RandomForestRegressor",
                "econml.grf._base_grf.BaseGRF",
                "causalml.inference.tree.CausalRandomForestRegressor",
            ],
        ):
            assert hasattr(model, "estimators_"), "Model has no `estimators_`! Have you called `model.fit`?"
            ensemble.internal_dtype = model.estimators_[0].tree_.value.dtype.type
            ensemble.input_dtype = np.float32
            scaling = 1.0 / len(model.estimators_)
            ensemble.trees = [
                SingleTree(e.tree_, scaling=scaling, data=data, data_missing=data_missing) for e in model.estimators_
            ]
            ensemble.objective = OBJECTIVE_NAME_MAP.get(model.criterion, None)
            ensemble.tree_output = "raw_value"

        # --- IsolationForest ---
        elif safe_isinstance(
            model,
            [
                "sklearn.ensemble.IsolationForest",
                "sklearn.ensemble._iforest.IsolationForest",
            ],
        ):
            ensemble.dtype = np.float32
            scaling = 1.0 / len(model.estimators_)
            ensemble.trees = [
                IsoTree(e.tree_, f, scaling=scaling, data=data, data_missing=data_missing)
                for e, f in zip(model.estimators_, model.estimators_features_)
            ]
            ensemble.tree_output = "raw_value"

        # --- ExtraTrees Regressor (+ skopt) ---
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
            ensemble.internal_dtype = model.estimators_[0].tree_.value.dtype.type
            ensemble.input_dtype = np.float32
            scaling = 1.0 / len(model.estimators_)
            ensemble.trees = [
                SingleTree(e.tree_, scaling=scaling, data=data, data_missing=data_missing) for e in model.estimators_
            ]
            ensemble.objective = OBJECTIVE_NAME_MAP.get(model.criterion, None)
            ensemble.tree_output = "raw_value"

        # --- DecisionTree Regressor ---
        elif safe_isinstance(
            model,
            [
                "sklearn.tree.DecisionTreeRegressor",
                "sklearn.tree.tree.DecisionTreeRegressor",
                "econml.grf._base_grftree.GRFTree",
                "causalml.inference.tree.causal.causaltree.CausalTreeRegressor",
            ],
        ):
            ensemble.internal_dtype = model.tree_.value.dtype.type
            ensemble.input_dtype = np.float32
            ensemble.trees = [SingleTree(model.tree_, data=data, data_missing=data_missing)]
            ensemble.objective = OBJECTIVE_NAME_MAP.get(model.criterion, None)
            ensemble.tree_output = "raw_value"

        # --- DecisionTree Classifier ---
        elif safe_isinstance(
            model,
            [
                "sklearn.tree.DecisionTreeClassifier",
                "sklearn.tree.tree.DecisionTreeClassifier",
            ],
        ):
            ensemble.internal_dtype = model.tree_.value.dtype.type
            ensemble.input_dtype = np.float32
            ensemble.trees = [SingleTree(model.tree_, normalize=True, data=data, data_missing=data_missing)]
            ensemble.objective = OBJECTIVE_NAME_MAP.get(model.criterion, None)
            ensemble.tree_output = "probability"

        # --- ExtraTrees / RandomForest Classifier ---
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
            ensemble.internal_dtype = model.estimators_[0].tree_.value.dtype.type
            ensemble.input_dtype = np.float32
            scaling = 1.0 / len(model.estimators_)
            ensemble.trees = [
                SingleTree(e.tree_, normalize=True, scaling=scaling, data=data, data_missing=data_missing)
                for e in model.estimators_
            ]
            ensemble.objective = OBJECTIVE_NAME_MAP.get(model.criterion, None)
            ensemble.tree_output = "probability"

        # --- GradientBoosting Regressor ---
        elif safe_isinstance(
            model,
            [
                "sklearn.ensemble.GradientBoostingRegressor",
                "sklearn.ensemble.gradient_boosting.GradientBoostingRegressor",
            ],
        ):
            ensemble.input_dtype = np.float32
            if safe_isinstance(
                model.init_,
                [
                    "sklearn.ensemble.MeanEstimator",
                    "sklearn.ensemble.gradient_boosting.MeanEstimator",
                ],
            ):
                ensemble.base_offset = model.init_.mean
            elif safe_isinstance(
                model.init_,
                [
                    "sklearn.ensemble.QuantileEstimator",
                    "sklearn.ensemble.gradient_boosting.QuantileEstimator",
                ],
            ):
                ensemble.base_offset = model.init_.quantile
            elif safe_isinstance(model.init_, "sklearn.dummy.DummyRegressor"):
                ensemble.base_offset = model.init_.constant_[0]
            else:
                emsg = f"Unsupported init model type: {type(model.init_)}"
                raise InvalidModelError(emsg)

            ensemble.trees = [
                SingleTree(e.tree_, scaling=model.learning_rate, data=data, data_missing=data_missing)
                for e in model.estimators_[:, 0]
            ]
            ensemble.objective = OBJECTIVE_NAME_MAP.get(model.criterion, None)
            ensemble.tree_output = "raw_value"

        # --- HistGradientBoosting Regressor ---
        elif safe_isinstance(model, ["sklearn.ensemble.HistGradientBoostingRegressor"]):
            import sklearn

            if ensemble.model_output == "predict":
                ensemble.model_output = "raw"
            ensemble.input_dtype = sklearn.ensemble._hist_gradient_boosting.common.X_DTYPE
            ensemble.base_offset = model._baseline_prediction
            ensemble.trees = []
            for p in model._predictors:
                nodes = p[0].nodes
                tree = {
                    "children_left": np.array([-1 if n[9] else n[5] for n in nodes]),
                    "children_right": np.array([-1 if n[9] else n[6] for n in nodes]),
                    "children_default": np.array([-1 if n[9] else (n[5] if n[4] else n[6]) for n in nodes]),
                    "features": np.array([-2 if n[9] else n[2] for n in nodes]),
                    "thresholds": np.array([n[3] for n in nodes], dtype=np.float64),
                    "values": np.array([[n[0]] for n in nodes], dtype=np.float64),
                    "node_sample_weight": np.array([n[1] for n in nodes], dtype=np.float64),
                }
                ensemble.trees.append(SingleTree(tree, data=data, data_missing=data_missing))
            ensemble.objective = OBJECTIVE_NAME_MAP.get(model.loss, None)
            ensemble.tree_output = "raw_value"

        # --- HistGradientBoosting Classifier ---
        elif safe_isinstance(model, ["sklearn.ensemble.HistGradientBoostingClassifier"]):
            import sklearn

            ensemble.base_offset = model._baseline_prediction
            has_len = hasattr(ensemble.base_offset, "__len__")
            if has_len and ensemble.base_offset.shape == (1, 1):
                ensemble.base_offset = ensemble.base_offset[0, 0]
                has_len = False
            if has_len and ensemble.model_output != "raw":
                emsg = (
                    "Multi-output HistGradientBoostingClassifier models are not yet supported unless "
                    'model_output="raw". See GitHub issue #1028.'
                )
                raise NotImplementedError(emsg)
            ensemble.input_dtype = sklearn.ensemble._hist_gradient_boosting.common.X_DTYPE
            ensemble.num_stacked_models = len(model._predictors[0])
            if ensemble.model_output == "predict_proba":
                if ensemble.num_stacked_models == 1:
                    ensemble.model_output = "probability_doubled"
                else:
                    ensemble.model_output = "probability"
            ensemble.trees = []
            for p in model._predictors:
                for i in range(ensemble.num_stacked_models):
                    nodes = p[i].nodes
                    tree = {
                        "children_left": np.array([-1 if n[9] else n[5] for n in nodes]),
                        "children_right": np.array([-1 if n[9] else n[6] for n in nodes]),
                        "children_default": np.array([-1 if n[9] else (n[5] if n[4] else n[6]) for n in nodes]),
                        "features": np.array([-2 if n[9] else n[2] for n in nodes]),
                        "thresholds": np.array([n[3] for n in nodes], dtype=np.float64),
                        "values": np.array([[n[0]] for n in nodes], dtype=np.float64),
                        "node_sample_weight": np.array([n[1] for n in nodes], dtype=np.float64),
                    }
                    ensemble.trees.append(SingleTree(tree, data=data, data_missing=data_missing))
            ensemble.objective = OBJECTIVE_NAME_MAP.get(model.loss, None)
            ensemble.tree_output = "log_odds"

        # --- GradientBoosting Classifier ---
        elif safe_isinstance(
            model,
            [
                "sklearn.ensemble.GradientBoostingClassifier",
                "sklearn.ensemble._gb.GradientBoostingClassifier",
                "sklearn.ensemble.gradient_boosting.GradientBoostingClassifier",
            ],
        ):
            ensemble.input_dtype = np.float32
            if model.estimators_.shape[1] > 1:
                emsg = "GradientBoostingClassifier is only supported for binary classification right now!"
                raise InvalidModelError(emsg)

            if safe_isinstance(
                model.init_,
                [
                    "sklearn.ensemble.LogOddsEstimator",
                    "sklearn.ensemble.gradient_boosting.LogOddsEstimator",
                ],
            ):
                ensemble.base_offset = model.init_.prior
                ensemble.tree_output = "log_odds"
            elif safe_isinstance(model.init_, "sklearn.dummy.DummyClassifier"):
                ensemble.base_offset = scipy.special.logit(model.init_.class_prior_[1])
                ensemble.tree_output = "log_odds"
            else:
                emsg = f"Unsupported init model type: {type(model.init_)}"
                raise InvalidModelError(emsg)

            ensemble.trees = [
                SingleTree(e.tree_, scaling=model.learning_rate, data=data, data_missing=data_missing)
                for e in model.estimators_[:, 0]
            ]
            ensemble.objective = OBJECTIVE_NAME_MAP.get(model.criterion, None)

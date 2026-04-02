"""XGBoost model support (Booster, XGBClassifier, XGBRegressor, XGBRanker)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from ...utils import safe_isinstance
from .._tree import XGBTreeModelLoader, get_xgboost_dmatrix_properties
from ._base import OBJECTIVE_NAME_MAP, TREE_OUTPUT_NAME_MAP, TreeModelHandler

if TYPE_CHECKING:
    from .._tree import TreeEnsemble


class XGBoostHandler(TreeModelHandler):
    """Handles all XGBoost tree-based models.

    Includes xgboost.core.Booster, XGBClassifier, XGBRegressor, XGBRanker.
    """

    @staticmethod
    def can_handle(model: Any) -> bool:
        return safe_isinstance(
            model,
            [
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
                "xgboost.sklearn.XGBRegressor",
                "xgboost.sklearn.XGBRanker",
            ],
        )

    @staticmethod
    def handle(
        model: Any,
        ensemble: TreeEnsemble,
        data: npt.NDArray[Any] | None,
        data_missing: npt.NDArray[np.bool_] | None,
    ) -> None:
        if safe_isinstance(model, "xgboost.core.Booster"):
            ensemble.original_model = model
            _set_xgboost_model_attributes(ensemble, data, data_missing)

        elif safe_isinstance(model, "xgboost.sklearn.XGBClassifier"):
            ensemble.input_dtype = np.float32
            ensemble.original_model = model.get_booster()
            _set_xgboost_model_attributes(ensemble, data, data_missing)

            if ensemble.model_output == "predict_proba":
                if ensemble.num_stacked_models == 1:
                    ensemble.model_output = "probability_doubled"
                else:
                    ensemble.model_output = "probability"
            ensemble._xgb_dmatrix_props = get_xgboost_dmatrix_properties(model)

        elif safe_isinstance(model, ["xgboost.sklearn.XGBRegressor", "xgboost.sklearn.XGBRanker"]):
            ensemble.original_model = model.get_booster()
            _set_xgboost_model_attributes(ensemble, data, data_missing)
            ensemble._xgb_dmatrix_props = get_xgboost_dmatrix_properties(model)


def _set_xgboost_model_attributes(
    ensemble: TreeEnsemble,
    data: npt.NDArray[Any] | None,
    data_missing: npt.NDArray[np.bool_] | None,
) -> None:
    """Shared logic for all XGBoost model types."""
    ensemble.model_type = "xgboost"
    loader = XGBTreeModelLoader(ensemble.original_model)

    ensemble.trees = loader.get_trees(data=data, data_missing=data_missing)
    ensemble.base_offset = loader.base_score
    ensemble.objective = OBJECTIVE_NAME_MAP.get(loader.name_obj, None)
    ensemble.tree_output = TREE_OUTPUT_NAME_MAP.get(loader.name_obj, None)

    ensemble.num_stacked_models = loader.n_trees_per_iter
    ensemble.cat_feature_indices = loader.cat_feature_indices
    best_iteration = getattr(
        ensemble.original_model,
        "best_iteration",
        ensemble.original_model.num_boosted_rounds() - 1,
    )
    ensemble.tree_limit = (best_iteration + 1) * ensemble.num_stacked_models
    ensemble._xgboost_n_outputs = loader.n_targets

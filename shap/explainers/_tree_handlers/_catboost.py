"""CatBoost model support."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from ...utils import assert_import, safe_isinstance
from .._tree import CatBoostTreeModelLoader
from ._base import TreeModelHandler

if TYPE_CHECKING:
    from .._tree import TreeEnsemble


class CatBoostHandler(TreeModelHandler):
    """Handles all CatBoost tree-based models.

    Includes CatBoostRegressor, CatBoostClassifier, CatBoost.
    """

    @staticmethod
    def can_handle(model: Any) -> bool:
        return safe_isinstance(
            model,
            [
                "catboost.core.CatBoostRegressor",
                "catboost.core.CatBoostClassifier",
                "catboost.core.CatBoost",
            ],
        )

    @staticmethod
    def handle(
        model: Any,
        ensemble: TreeEnsemble,
        data: npt.NDArray[Any] | None,
        data_missing: npt.NDArray[np.bool_] | None,
    ) -> None:
        assert_import("catboost")
        ensemble.model_type = "catboost"
        ensemble.original_model = model

        if safe_isinstance(model, "catboost.core.CatBoostRegressor"):
            ensemble.cat_feature_indices = model.get_cat_feature_indices()
            try:
                cb_loader = CatBoostTreeModelLoader(model)
                ensemble.trees = cb_loader.get_trees(data=data, data_missing=data_missing)
            except Exception:
                ensemble.trees = None  # cext can't handle categorical splits yet

        elif safe_isinstance(model, "catboost.core.CatBoostClassifier"):
            ensemble.input_dtype = np.float32
            try:
                cb_loader = CatBoostTreeModelLoader(model)
                ensemble.trees = cb_loader.get_trees(data=data, data_missing=data_missing)
            except Exception:
                ensemble.trees = None
            ensemble.tree_output = "log_odds"
            ensemble.objective = "binary_crossentropy"
            ensemble.cat_feature_indices = model.get_cat_feature_indices()

        elif safe_isinstance(model, "catboost.core.CatBoost"):
            ensemble.cat_feature_indices = model.get_cat_feature_indices()

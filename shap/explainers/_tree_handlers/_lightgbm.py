"""LightGBM model support (Booster, LGBMRegressor, LGBMClassifier, etc.)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...utils import assert_import, safe_isinstance
from .._tree import SingleTree
from ._base import OBJECTIVE_NAME_MAP, TREE_OUTPUT_NAME_MAP, TreeModelHandler

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from .._tree import TreeEnsemble


class LightGBMHandler(TreeModelHandler):
    """Handles all LightGBM tree-based models.

    Includes lightgbm.basic.Booster, LGBMRegressor, LGBMRanker, LGBMClassifier.
    """

    @staticmethod
    def can_handle(model: Any) -> bool:
        return safe_isinstance(
            model,
            [
                "lightgbm.basic.Booster",
                "lightgbm.sklearn.LGBMRegressor",
                "lightgbm.sklearn.LGBMRanker",
                "lightgbm.sklearn.LGBMClassifier",
            ],
        )

    @staticmethod
    def handle(
        model: Any,
        ensemble: TreeEnsemble,
        data: npt.NDArray[Any] | None,
        data_missing: npt.NDArray[np.bool_] | None,
    ) -> None:
        assert_import("lightgbm")
        ensemble.model_type = "lightgbm"

        if safe_isinstance(model, "lightgbm.basic.Booster"):
            ensemble.original_model = model
            tree_info = model.dump_model()["tree_info"]
            try:
                ensemble.trees = [SingleTree(e, data=data, data_missing=data_missing) for e in tree_info]
            except Exception:
                ensemble.trees = None  # cext can't handle categorical splits yet
            ensemble.objective = OBJECTIVE_NAME_MAP.get(model.params.get("objective", "regression"), None)
            ensemble.tree_output = TREE_OUTPUT_NAME_MAP.get(model.params.get("objective", "regression"), None)

        elif safe_isinstance(model, "lightgbm.sklearn.LGBMRegressor"):
            ensemble.original_model = model.booster_
            tree_info = ensemble.original_model.dump_model()["tree_info"]
            try:
                ensemble.trees = [SingleTree(e, data=data, data_missing=data_missing) for e in tree_info]
            except Exception:
                ensemble.trees = None
            ensemble.objective = OBJECTIVE_NAME_MAP.get(model.objective, None)
            ensemble.tree_output = TREE_OUTPUT_NAME_MAP.get(model.objective, None)
            if model.objective is None:
                ensemble.objective = "squared_error"
                ensemble.tree_output = "raw_value"

        elif safe_isinstance(model, "lightgbm.sklearn.LGBMRanker"):
            ensemble.original_model = model.booster_
            tree_info = ensemble.original_model.dump_model()["tree_info"]
            try:
                ensemble.trees = [SingleTree(e, data=data, data_missing=data_missing) for e in tree_info]
            except Exception:
                ensemble.trees = None
            # Note: for ranker, leaving tree_output and objective as None as they
            # are not implemented in native code yet

        elif safe_isinstance(model, "lightgbm.sklearn.LGBMClassifier"):
            if model.n_classes_ > 2:
                ensemble.num_stacked_models = model.n_classes_
            ensemble.original_model = model.booster_
            tree_info = ensemble.original_model.dump_model()["tree_info"]
            try:
                ensemble.trees = [SingleTree(e, data=data, data_missing=data_missing) for e in tree_info]
            except Exception:
                ensemble.trees = None
            ensemble.objective = OBJECTIVE_NAME_MAP.get(model.objective, None)
            ensemble.tree_output = TREE_OUTPUT_NAME_MAP.get(model.objective, None)
            if model.objective is None:
                ensemble.objective = "binary_crossentropy"
                ensemble.tree_output = "log_odds"

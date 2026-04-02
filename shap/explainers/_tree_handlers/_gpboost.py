"""GPBoost support (gpboost.basic.Booster)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...utils import assert_import, safe_isinstance
from .._tree import SingleTree
from ._base import OBJECTIVE_NAME_MAP, TREE_OUTPUT_NAME_MAP, TreeModelHandler

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from .._tree import TreeEnsemble


class GPBoostHandler(TreeModelHandler):
    """Handles GPBoost tree-based models."""

    @staticmethod
    def can_handle(model: Any) -> bool:
        return safe_isinstance(model, "gpboost.basic.Booster")

    @staticmethod
    def handle(
        model: Any,
        ensemble: TreeEnsemble,
        data: npt.NDArray[Any] | None,
        data_missing: npt.NDArray[np.bool_] | None,
    ) -> None:
        assert_import("gpboost")
        ensemble.model_type = "gpboost"
        ensemble.original_model = model
        tree_info = model.dump_model()["tree_info"]
        try:
            ensemble.trees = [SingleTree(e, data=data, data_missing=data_missing) for e in tree_info]
        except Exception:
            ensemble.trees = None  # cext can't handle categorical splits yet

        ensemble.objective = OBJECTIVE_NAME_MAP.get(model.params.get("objective", "regression"), None)
        ensemble.tree_output = TREE_OUTPUT_NAME_MAP.get(model.params.get("objective", "regression"), None)

"""Misc handlers: raw dict/list models, imblearn, pyod."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from ...utils import safe_isinstance
from .._tree import IsoTree, SingleTree
from ._base import OBJECTIVE_NAME_MAP, TreeModelHandler

if TYPE_CHECKING:
    from .._tree import TreeEnsemble


class DictHandler(TreeModelHandler):
    """Handles dictionary and list-based model representations."""

    @staticmethod
    def can_handle(model: Any) -> bool:
        if isinstance(model, dict) and "trees" in model:
            return True
        if isinstance(model, list) and len(model) > 0 and isinstance(model[0], SingleTree):
            return True
        return False

    @staticmethod
    def handle(
        model: Any,
        ensemble: TreeEnsemble,
        data: npt.NDArray[Any] | None,
        data_missing: npt.NDArray[np.bool_] | None,
    ) -> None:
        if isinstance(model, dict) and "trees" in model:
            if "internal_dtype" in model:
                ensemble.internal_dtype = model["internal_dtype"]
            if "input_dtype" in model:
                ensemble.input_dtype = model["input_dtype"]
            if "objective" in model:
                ensemble.objective = model["objective"]
            if "tree_output" in model:
                ensemble.tree_output = model["tree_output"]
            if "base_offset" in model:
                ensemble.base_offset = model["base_offset"]
            ensemble.trees = [SingleTree(t, data=data, data_missing=data_missing) for t in model["trees"]]
        elif isinstance(model, list) and isinstance(model[0], SingleTree):
            ensemble.trees = model


class ImbLearnHandler(TreeModelHandler):
    """Handles imbalanced-learn BalancedRandomForestClassifier."""

    @staticmethod
    def can_handle(model: Any) -> bool:
        return safe_isinstance(model, "imblearn.ensemble._forest.BalancedRandomForestClassifier")

    @staticmethod
    def handle(
        model: Any,
        ensemble: TreeEnsemble,
        data: npt.NDArray[Any] | None,
        data_missing: npt.NDArray[np.bool_] | None,
    ) -> None:
        ensemble.input_dtype = np.float32
        scaling = 1.0 / len(model.estimators_)
        ensemble.trees = [
            SingleTree(e.tree_, normalize=True, scaling=scaling, data=data, data_missing=data_missing)
            for e in model.estimators_
        ]
        ensemble.objective = OBJECTIVE_NAME_MAP.get(model.criterion, None)
        ensemble.tree_output = "probability"


class PyODHandler(TreeModelHandler):
    """Handles PyOD IsolationForest."""

    @staticmethod
    def can_handle(model: Any) -> bool:
        return safe_isinstance(model, ["pyod.models.iforest.IForest"])

    @staticmethod
    def handle(
        model: Any,
        ensemble: TreeEnsemble,
        data: npt.NDArray[Any] | None,
        data_missing: npt.NDArray[np.bool_] | None,
    ) -> None:
        ensemble.dtype = np.float32
        scaling = 1.0 / len(model.estimators_)
        ensemble.trees = [
            IsoTree(e.tree_, f, scaling=scaling, data=data, data_missing=data_missing)
            for e, f in zip(model.detector_.estimators_, model.detector_.estimators_features_)
        ]
        ensemble.tree_output = "raw_value"

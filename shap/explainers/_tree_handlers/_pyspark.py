"""PySpark ML tree model support."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...utils import assert_import, safe_isinstance
from .._tree import SingleTree
from ._base import OBJECTIVE_NAME_MAP, TreeModelHandler

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from .._tree import TreeEnsemble


class PySparkHandler(TreeModelHandler):
    """Handles all PySpark ML tree-based models."""

    @staticmethod
    def can_handle(model: Any) -> bool:
        return "pyspark.ml" in str(type(model))

    @staticmethod
    def handle(
        model: Any,
        ensemble: TreeEnsemble,
        data: npt.NDArray[Any] | None,
        data_missing: npt.NDArray[np.bool_] | None,
    ) -> None:
        assert_import("pyspark")
        ensemble.model_type = "pyspark"
        ensemble.objective = OBJECTIVE_NAME_MAP.get(model._java_obj.getImpurity(), None)

        if "Classification" in str(type(model)):
            normalize = True
            ensemble.tree_output = "probability"
        else:
            normalize = False
            ensemble.tree_output = "raw_value"

        # Spark Random forest
        if safe_isinstance(
            model,
            [
                "pyspark.ml.classification.RandomForestClassificationModel",
                "pyspark.ml.regression.RandomForestRegressionModel",
            ],
        ):
            sum_weight = sum(model.treeWeights)
            ensemble.trees = [
                SingleTree(tree, normalize=normalize, scaling=model.treeWeights[i] / sum_weight)
                for i, tree in enumerate(model.trees)
            ]

        # Spark GBT
        elif safe_isinstance(
            model,
            [
                "pyspark.ml.classification.GBTClassificationModel",
                "pyspark.ml.regression.GBTRegressionModel",
            ],
        ):
            ensemble.objective = "squared_error"
            ensemble.tree_output = "raw_value"
            ensemble.trees = [
                SingleTree(tree, normalize=False, scaling=model.treeWeights[i]) for i, tree in enumerate(model.trees)
            ]

        # Spark Basic model (single tree)
        elif safe_isinstance(
            model,
            [
                "pyspark.ml.classification.DecisionTreeClassificationModel",
                "pyspark.ml.regression.DecisionTreeRegressionModel",
            ],
        ):
            ensemble.trees = [SingleTree(model, normalize=normalize, scaling=1)]

        else:
            emsg = f"Unsupported Spark model type: {type(model)}"
            raise NotImplementedError(emsg)

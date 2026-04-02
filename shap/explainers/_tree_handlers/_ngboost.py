"""NGBoost model support."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from ...utils import safe_isinstance
from .._tree import SingleTree
from ._base import OBJECTIVE_NAME_MAP, TreeModelHandler

if TYPE_CHECKING:
    from .._tree import TreeEnsemble


class NGBoostHandler(TreeModelHandler):
    """Handles all NGBoost tree-based models.

    Includes NGBoost, NGBRegressor, NGBClassifier.
    """

    @staticmethod
    def can_handle(model: Any) -> bool:
        return safe_isinstance(
            model,
            [
                "ngboost.ngboost.NGBoost",
                "ngboost.api.NGBRegressor",
                "ngboost.api.NGBClassifier",
            ],
        )

    @staticmethod
    def handle(
        model: Any,
        ensemble: TreeEnsemble,
        data: npt.NDArray[Any] | None,
        data_missing: npt.NDArray[np.bool_] | None,
    ) -> None:
        assert model.base_models, "The NGBoost model has empty `base_models`! Have you called `model.fit`?"
        if ensemble.model_output == "raw":
            param_idx = 0
            warnings.warn(
                'Translating model_output="raw" to model_output=0 for the 0-th parameter in the distribution. '
                "Use model_output=0 directly to avoid this warning."
            )
        elif isinstance(ensemble.model_output, int):
            param_idx = ensemble.model_output
            ensemble.model_output = "raw"
        else:
            param_idx = 0  # default fallback

        assert safe_isinstance(
            model.base_models[0][param_idx],
            ["sklearn.tree.DecisionTreeRegressor", "sklearn.tree.tree.DecisionTreeRegressor"],
        ), "You must use default_tree_learner!"

        shap_trees = [trees[param_idx] for trees in model.base_models]
        ensemble.internal_dtype = shap_trees[0].tree_.value.dtype.type
        ensemble.input_dtype = np.float32
        scaling = -model.learning_rate * np.array(model.scalings)

        # ngboost reorders features, map them back to original order
        missing_col_idxs = [[i for i in range(model.n_features) if i not in col_idx] for col_idx in model.col_idxs]
        feature_mapping = [
            {i: col_idx for i, col_idx in enumerate(list(col_idxs) + missing_col_idx)}
            for col_idxs, missing_col_idx in zip(model.col_idxs, missing_col_idxs)
        ]

        ensemble.trees = []
        for idx, shap_tree in enumerate(shap_trees):
            tree_ = shap_tree.tree_
            values = tree_.value.reshape(tree_.value.shape[0], tree_.value.shape[1] * tree_.value.shape[2])
            values = values * scaling[idx]  # type: ignore[index]
            tree = {
                "children_left": tree_.children_left.astype(np.int32),
                "children_right": tree_.children_right.astype(np.int32),
                "children_default": tree_.children_left,
                "features": np.array([feature_mapping[idx].get(i, i) for i in tree_.feature]),
                "thresholds": tree_.threshold.astype(np.float64),
                "values": values,
                "node_sample_weight": tree_.weighted_n_node_samples.astype(np.float64),
            }
            ensemble.trees.append(SingleTree(tree, data=data, data_missing=data_missing))

        ensemble.objective = OBJECTIVE_NAME_MAP.get(shap_trees[0].criterion, None)
        ensemble.tree_output = "raw_value"
        ensemble.base_offset = model.init_params[param_idx]

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from shap import links
from shap.models import Model
from shap.utils import MaskedModel

from .._explainer import Explainer

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


class Random(Explainer):
    """Simply returns random (normally distributed) feature attributions.

    This is only for benchmark comparisons. It supports both fully random attributions and random
    attributions that are constant across all explanations.
    """

    def __init__(
        self,
        model: Any,
        masker: Any,
        link: Callable[..., Any] = links.identity,
        feature_names: list[str] | list[list[str]] | None = None,
        linearize_link: bool = True,
        constant: bool = False,
        **call_args: Any,
    ) -> None:
        super().__init__(model, masker, link=link, linearize_link=linearize_link, feature_names=feature_names)

        if not isinstance(model, Model):
            self.model = Model(model)

        for arg in call_args:
            self.__call__.__kwdefaults__[arg] = call_args[arg]  # type: ignore[index]

        self.constant: bool = constant
        self.constant_attributions: NDArray[np.floating[Any]] | None = None

    def explain_row(
        self,
        *row_args: Any,
        max_evals: int | Literal["auto"],
        main_effects: bool,
        error_bounds: bool,
        outputs: Any,
        silent: bool,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Explain a single row and return feature attributions."""
        # build a masked version of the model for the current input sample
        fm = MaskedModel(self.model, self.masker, self.link, self.linearize_link, *row_args)

        # compute any custom clustering for this row
        row_clustering: NDArray[np.floating[Any]] | None = None
        if getattr(self.masker, "clustering", None) is not None:
            if isinstance(self.masker.clustering, np.ndarray):
                row_clustering = self.masker.clustering
            elif callable(self.masker.clustering):
                row_clustering = self.masker.clustering(*row_args)
            else:
                raise NotImplementedError(
                    "The masker passed has a .clustering attribute that is not yet supported by the Permutation explainer!"
                )

        # compute the correct expected values
        masks = np.zeros(1, dtype=int)
        outputs = fm(masks, zero_index=0, batch_size=1)
        expected_value = outputs[0]

        # generate random feature attributions
        # we produce small values so our explanation errors are similar to a constant function
        row_values: NDArray[np.floating[Any]] = np.random.randn(*((len(fm),) + outputs.shape[1:])) * 0.001

        return {
            "values": row_values,
            "expected_values": expected_value,
            "mask_shapes": fm.mask_shapes,
            "main_effects": None,
            "clustering": row_clustering,
            "error_std": None,
            "output_names": self.model.output_names if hasattr(self.model, "output_names") else None,
        }

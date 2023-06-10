import numpy as np

from shap import links
from shap.models import Model
from shap.utils import MaskedModel

from .._explainer import Explainer


class Random(Explainer):
    """ Simply returns random (normally distributed) feature attributions.

    This is only for benchmark comparisons. It supports both fully random attributions and random
    attributions that are constant across all explainations.
    """
    def __init__(self, model, masker, link=links.identity, feature_names=None, linearize_link=True, constant=False, **call_args):
        super().__init__(model, masker, link=link, linearize_link=linearize_link, feature_names=feature_names)

        if not isinstance(model, Model):
            self.model = Model(model)

        for arg in call_args:
            self.__call__.__kwdefaults__[arg] = call_args[arg]

        self.constant = constant
        self.constant_attributions = None

    def explain_row(self, *row_args, max_evals, main_effects, error_bounds, batch_size, outputs, silent):
        """ Explains a single row.
        """

        # build a masked version of the model for the current input sample
        fm = MaskedModel(self.model, self.masker, self.link, self.linearize_link, *row_args)

        # compute any custom clustering for this row
        row_clustering = None
        if getattr(self.masker, "clustering", None) is not None:
            if isinstance(self.masker.clustering, np.ndarray):
                row_clustering = self.masker.clustering
            elif callable(self.masker.clustering):
                row_clustering = self.masker.clustering(*row_args)
            else:
                raise NotImplementedError("The masker passed has a .clustering attribute that is not yet supported by the Permutation explainer!")

        # compute the correct expected value
        masks = np.zeros(1, dtype=int)
        outputs = fm(masks, zero_index=0, batch_size=1)
        expected_value = outputs[0]

        # generate random feature attributions
        # we produce small values so our explanation errors are similar to a constant function
        row_values = np.random.randn(*((len(fm),) + outputs.shape[1:])) * 0.001

        return {
            "values": row_values,
            "expected_values": expected_value,
            "mask_shapes": fm.mask_shapes,
            "main_effects": None,
            "clustering": row_clustering,
            "error_std": None,
            "output_names": self.model.output_names if hasattr(self.model, "output_names") else None
        }

    # def __call__(self, X):
    #     start_time = time.time()
    #     if self.constant:
    #         if self.constant_attributions is None:
    #             self.constant_attributions = np.random.randn(X.shape[1])
    #         return Explanation(np.tile(self.constant_attributions, (X.shape[0],1)), X, compute_time=time.time() - start_time)
    #     else:
    #         return Explanation(np.random.randn(*X.shape), X, compute_time=time.time() - start_time)

    # def attributions(self, X):
    #     if self.constant:
    #         if self.constant_attributions is None:
    #             self.constant_attributions = np.random.randn(X.shape[1])
    #         return np.tile(self.constant_attributions, (X.shape[0],1))
    #     else:
    #         return np.random.randn(*X.shape)

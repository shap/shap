import numpy as np

from .. import links
from ..models import Model
from ..utils import MaskedModel, partition_tree_shuffle
from ._explainer import Explainer


class PermutationExplainer(Explainer):
    """This method approximates the Shapley values by iterating through permutations of the inputs.

    This is a model agnostic explainer that guarantees local accuracy (additivity) by iterating completely
    through an entire permutation of the features in both forward and reverse directions (antithetic sampling).
    If we do this once, then we get the exact SHAP values for models with up to second order interaction effects.
    We can iterate this many times over many random permutations to get better SHAP value estimates for models
    with higher order interactions. This sequential ordering formulation also allows for easy reuse of
    model evaluations and the ability to efficiently avoid evaluating the model when the background values
    for a feature are the same as the current input value. We can also account for hierarchical data
    structures with partition trees, something not currently implemented for KernalExplainer or SamplingExplainer.
    """

    def __init__(
        self, model, masker, link=links.identity, feature_names=None, linearize_link=True, seed=None, **call_args
    ):
        """Build an explainers.Permutation object for the given model using the given masker object.

        Parameters
        ----------
        model : function
            A callable python object that executes the model given a set of input data samples.

        masker : function or numpy.array or pandas.DataFrame
            A callable python object used to "mask" out hidden features of the form ``masker(binary_mask, x)``.
            It takes a single input sample and a binary mask and returns a matrix of masked samples. These
            masked samples are evaluated using the model function and the outputs are then averaged.
            As a shortcut for the standard masking using by SHAP you can pass a background data matrix
            instead of a function and that matrix will be used for masking. To use a clustering
            game structure you can pass a ``shap.maskers.Tabular(data, clustering="correlation")`` object.

        seed: None or int
            Seed for reproducibility

        **call_args : valid argument to the __call__ method
            These arguments are saved and passed to the __call__ method as the new default values for these arguments.

        """
        # setting seed for random generation: if seed is not None, then shap values computation should be reproducible
        np.random.seed(seed)

        if masker is None:
            raise ValueError("masker cannot be None.")

        super().__init__(model, masker, link=link, linearize_link=linearize_link, feature_names=feature_names)

        if not isinstance(self.model, Model):
            self.model = Model(self.model)

        # if we have gotten default arguments for the call function we need to wrap ourselves in a new class that
        # has a call function with those new default arguments
        if len(call_args) > 0:
            # this signature should match the __call__ signature of the class defined below
            class PermutationExplainer(self.__class__):
                def __call__(
                    self,
                    *args,
                    max_evals=500,
                    main_effects=False,
                    error_bounds=False,
                    batch_size="auto",
                    outputs=None,
                    silent=False,
                ):
                    return super().__call__(
                        *args,
                        max_evals=max_evals,
                        main_effects=main_effects,
                        error_bounds=error_bounds,
                        batch_size=batch_size,
                        outputs=outputs,
                        silent=silent,
                    )

            PermutationExplainer.__call__.__doc__ = self.__class__.__call__.__doc__
            self.__class__ = PermutationExplainer
            for k, v in call_args.items():
                self.__call__.__kwdefaults__[k] = v

    # note that changes to this function signature should be copied to the default call argument wrapper above
    def __call__(
        self,
        *args,
        max_evals=500,
        main_effects=False,
        error_bounds=False,
        batch_size="auto",
        outputs=None,
        silent=False,
    ):
        """Explain the output of the model on the given arguments."""
        return super().__call__(
            *args,
            max_evals=max_evals,
            main_effects=main_effects,
            error_bounds=error_bounds,
            batch_size=batch_size,
            outputs=outputs,
            silent=silent,
        )

    def explain_row(self, *row_args, max_evals, main_effects, error_bounds, batch_size, outputs, silent):
        """Explains a single row and returns the tuple (row_values, row_expected_values, row_mask_shapes)."""
        # build a masked version of the model for the current input sample
        fm = MaskedModel(self.model, self.masker, self.link, self.linearize_link, *row_args)

        # by default we run 10 permutations forward and backward
        if max_evals == "auto":
            max_evals = 10 * 2 * len(fm)

        # compute any custom clustering for this row
        row_clustering = None
        if getattr(self.masker, "clustering", None) is not None:
            if isinstance(self.masker.clustering, np.ndarray):
                row_clustering = self.masker.clustering
            elif callable(self.masker.clustering):
                row_clustering = self.masker.clustering(*row_args)
            else:
                raise NotImplementedError(
                    "The masker passed has a .clustering attribute that is not yet supported by the Permutation explainer!"
                )

        # loop over many permutations
        inds = fm.varying_inputs()
        inds_mask = np.zeros(len(fm), dtype=bool)
        inds_mask[inds] = True
        masks = np.zeros(2 * len(inds) + 1, dtype=int)
        masks[0] = MaskedModel.delta_mask_noop_value
        npermutations = max_evals // (2 * len(inds) + 1)
        row_values = None
        row_values_history = None
        history_pos = 0
        main_effect_values = None
        if len(inds) > 0:
            for _ in range(npermutations):
                # shuffle the indexes so we get a random permutation ordering
                if row_clustering is not None:
                    # [TODO] This is shuffle does not work when inds is not a complete set of integers from 0 to M TODO: still true?
                    # assert len(inds) == len(fm), "Need to support partition shuffle when not all the inds vary!!"
                    partition_tree_shuffle(inds, inds_mask, row_clustering)
                else:
                    np.random.shuffle(inds)

                # create a large batch of masks to evaluate
                i = 1
                for ind in inds:
                    masks[i] = ind
                    i += 1
                for ind in inds:
                    masks[i] = ind
                    i += 1

                # evaluate the masked model
                outputs = fm(masks, zero_index=0, batch_size=batch_size)

                if row_values is None:
                    row_values = np.zeros((len(fm),) + outputs.shape[1:])

                    if error_bounds:
                        row_values_history = np.zeros(
                            (
                                2 * npermutations,
                                len(fm),
                            )
                            + outputs.shape[1:]
                        )

                # update our SHAP value estimates
                i = 0
                for ind in inds:  # forward
                    row_values[ind] += outputs[i + 1] - outputs[i]
                    if error_bounds:
                        row_values_history[history_pos][ind] = outputs[i + 1] - outputs[i]
                    i += 1
                history_pos += 1
                for ind in inds:  # backward
                    row_values[ind] += outputs[i] - outputs[i + 1]
                    if error_bounds:
                        row_values_history[history_pos][ind] = outputs[i] - outputs[i + 1]
                    i += 1
                history_pos += 1

            if npermutations == 0:
                raise ValueError(
                    f"max_evals={max_evals} is too low for the Permutation explainer, it must be at least 2 * num_features + 1 = {2 * len(inds) + 1}!"
                )

            expected_value = outputs[0]

            # compute the main effects if we need to
            if main_effects:
                main_effect_values = fm.main_effects(inds, batch_size=batch_size)
        else:
            masks = np.zeros(1, dtype=int)
            outputs = fm(masks, zero_index=0, batch_size=1)
            expected_value = outputs[0]
            row_values = np.zeros((len(fm),) + outputs.shape[1:])
            if error_bounds:
                row_values_history = np.zeros(
                    (
                        2 * npermutations,
                        len(fm),
                    )
                    + outputs.shape[1:]
                )

        return {
            "values": row_values / (2 * npermutations),
            "expected_values": expected_value,
            "mask_shapes": fm.mask_shapes,
            "main_effects": main_effect_values,
            "clustering": row_clustering,
            "error_std": None if row_values_history is None else row_values_history.std(0),
            "output_names": self.model.output_names if hasattr(self.model, "output_names") else None,
        }

    def shap_values(self, X, npermutations=10, main_effects=False, error_bounds=False, batch_evals=True, silent=False):
        """Legacy interface to estimate the SHAP values for a set of samples.

        Parameters
        ----------
        X : numpy.array or pandas.DataFrame or any scipy.sparse matrix
            A matrix of samples (# samples x # features) on which to explain the model's output.

        npermutations : int
            Number of times to cycle through all the features, re-evaluating the model at each step.
            Each cycle evaluates the model function 2 * (# features + 1) times on a data matrix of
            (# background data samples) rows. An exception to this is when PermutationExplainer can
            avoid evaluating the model because a feature's value is the same in X and the background
            dataset (which is common for example with sparse features).

        Returns
        -------
        array or list
            For models with a single output this returns a matrix of SHAP values
            (# samples x # features). Each row sums to the difference between the model output for that
            sample and the expected value of the model output (which is stored as expected_value
            attribute of the explainer). For models with vector outputs this returns a list
            of such matrices, one for each output.

        """
        explanation = self(X, max_evals=npermutations * X.shape[1], main_effects=main_effects)
        return explanation.values

    def __str__(self):
        return "shap.explainers.PermutationExplainer()"

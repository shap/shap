import time

import numpy as np
from tqdm.auto import tqdm

from shap import Explanation, links
from shap.maskers import FixedComposite, Image, Text
from shap.utils import MaskedModel, partition_tree_shuffle, safe_isinstance

from ._result import BenchmarkResult


class ExplanationError:
    """ A measure of the explanation error relative to a model's actual output.

    This benchmark metric measures the discrepancy between the output of the model predicted by an
    attribution explanation vs. the actual output of the model. This discrepancy is measured over
    many masking patterns drawn from permutations of the input features.

    For explanations (like Shapley values) that explain the difference between one alternative and another
    (for example a current sample and typical background feature values) there is possible explanation error
    for every pattern of mixing foreground and background, or other words every possible masking pattern.
    In this class we compute the standard deviation over these explanation errors where masking patterns
    are drawn from prefixes of random feature permutations. This seems natural, and aligns with Shapley value
    computations, but of course you could choose to summarize explanation errors in others ways as well.
    """

    def __init__(self, masker, model, *model_args, batch_size=500, num_permutations=10, link=links.identity, linearize_link=True, seed=38923):
        """ Build a new explanation error benchmarker with the given masker, model, and model args.

        Parameters
        ----------
        masker : function or shap.Masker
            The masker defines how we hide features during the perturbation process.

        model : function or shap.Model
            The model we want to evaluate explanations against.

        model_args : ...
            The list of arguments we will give to the model that we will have explained. When we later call this benchmark
            object we should pass explanations that have been computed on this same data.

        batch_size : int
            The maximum batch size we should use when calling the model. For some large NLP models this needs to be set
            lower (at say 1) to avoid running out of GPU memory.

        num_permutations : int
            How many permutations we will use to estimate the average explanation error for each sample. If you are running
            this benchmark on a large dataset with many samples then you can reduce this value since the final result is
            averaged over samples as well and the averages of both directly combine to reduce variance. So for 10k samples
            num_permutations=1 is appropreiate.

        link : function
            Allows for a non-linear link function to be used to bringe between the model output space and the explanation
            space.

        linearize_link : bool
            Non-linear links can destroy additive separation in generalized linear models, so by linearizing the link we can
            retain additive separation. See upcoming paper/doc for details.
        """

        self.masker = masker
        self.model = model
        self.model_args = model_args
        self.num_permutations = num_permutations
        self.link = link
        self.linearize_link = linearize_link
        self.model_args = model_args
        self.batch_size = batch_size
        self.seed = seed

        # user must give valid masker
        underlying_masker = masker.masker if isinstance(masker, FixedComposite) else masker
        if isinstance(underlying_masker, Text):
            self.data_type = "text"
        elif isinstance(underlying_masker, Image):
            self.data_type = "image"
        else:
            self.data_type = "tabular"

    def __call__(self, explanation, name, step_fraction=0.01, indices=[], silent=False):
        """ Run this benchmark on the given explanation.
        """

        if safe_isinstance(explanation, "numpy.ndarray"):
            attributions = explanation
        elif isinstance(explanation, Explanation):
            attributions = explanation.values
        else:
            raise ValueError("The passed explanation must be either of type numpy.ndarray or shap.Explanation!")

        assert len(attributions) == len(self.model_args[0]), "The explanation passed must have the same number of rows as " + \
                                                             "the self.model_args that were passed!"

        # it is important that we choose the same permutations for the different explanations we are comparing
        # so as to avoid needless noise
        old_seed = np.random.seed()
        np.random.seed(self.seed)

        pbar = None
        start_time = time.time()
        svals = []
        mask_vals = []

        for i, args in enumerate(zip(*self.model_args)):

            if len(args[0].shape) != len(attributions[i].shape):
                raise ValueError("The passed explanation must have the same dim as the model_args and must not have a vector output!")

            feature_size = np.prod(attributions[i].shape)
            sample_attributions = attributions[i].flatten()

            # compute any custom clustering for this row
            row_clustering = None
            if getattr(self.masker, "clustering", None) is not None:
                if isinstance(self.masker.clustering, np.ndarray):
                    row_clustering = self.masker.clustering
                elif callable(self.masker.clustering):
                    row_clustering = self.masker.clustering(*args)
                else:
                    raise NotImplementedError("The masker passed has a .clustering attribute that is not yet supported by the ExplanationError benchmark!")

            masked_model = MaskedModel(self.model, self.masker, self.link, self.linearize_link, *args)

            total_values = None
            for _ in range(self.num_permutations):
                masks = []
                mask = np.zeros(feature_size, dtype=bool)
                masks.append(mask.copy())
                ordered_inds = np.arange(feature_size)

                # shuffle the indexes so we get a random permutation ordering
                if row_clustering is not None:
                    inds_mask = np.ones(feature_size, dtype=bool)
                    partition_tree_shuffle(ordered_inds, inds_mask, row_clustering)
                else:
                    np.random.shuffle(ordered_inds)

                increment = max(1, int(feature_size * step_fraction))
                for j in range(0, feature_size, increment):
                    mask[ordered_inds[np.arange(j, min(feature_size, j+increment))]] = True
                    masks.append(mask.copy())
                mask_vals.append(masks)

                values = []
                masks_arr = np.array(masks)
                for j in range(0, len(masks_arr), self.batch_size):
                    values.append(masked_model(masks_arr[j:j + self.batch_size]))
                values = np.concatenate(values)
                base_value = values[0]
                for l, v in enumerate(values):
                    values[l] = (v - (base_value + np.sum(sample_attributions[masks_arr[l]])))**2

                if total_values is None:
                    total_values = values
                else:
                    total_values += values
            total_values /= self.num_permutations

            svals.append(total_values)

            if pbar is None and time.time() - start_time > 5:
                pbar = tqdm(total=len(self.model_args[0]), disable=silent, leave=False, desc=f"ExplanationError for {name}")
                pbar.update(i+1)
            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        svals = np.array(svals)

        # reset the random seed so we don't mess up the caller
        np.random.seed(old_seed)

        return BenchmarkResult("explanation error", name, value=np.sqrt(np.sum(total_values)/len(total_values)))

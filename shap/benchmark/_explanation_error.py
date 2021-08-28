import time
import numpy as np
from tqdm import tqdm
from shap.utils import safe_isinstance, MaskedModel, partition_tree_shuffle
from shap import Explanation, links
from shap.maskers import Text, Image, FixedComposite
from . import BenchmarkResult


class ExplanationError():
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
            lower (at say 10) to avoid running out of GPU memory.

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

        # it is important that we choose the same permutations for difference explanations we are comparing
        # so as to avoid needless noise
        old_seed = np.random.seed()
        np.random.seed(self.seed)

        pbar = None
        start_time = time.time()
        svals = []
        mask_vals = []

        for i, args in enumerate(zip(*self.model_args)):
            if self.data_type == "image":
                x_shape, y_shape = attributions[i].shape[0], attributions[i].shape[1]
                feature_size = np.prod([x_shape, y_shape])
                sample_attributions = attributions[i].mean(2).reshape(feature_size, -1)
                data = X[i].flatten()
                mask_shape = X[i].shape
            else:
                feature_size = attributions[i].shape[0]
                sample_attributions = attributions[i]
                # data = X[i]
                mask_shape = feature_size

            # compute any custom clustering for this row
            row_clustering = None
            if getattr(self.masker, "clustering", None) is not None:
                if isinstance(self.masker.clustering, np.ndarray):
                    row_clustering = self.masker.clustering
                elif callable(self.masker.clustering):
                    row_clustering = self.masker.clustering(*args)
                else:
                    raise Exception("The masker passed has a .clustering attribute that is not yet supported by the ExplanationError benchmark!")

            masked_model = MaskedModel(self.model, self.masker, self.link, self.linearize_link, *args)

            if len(attributions[i].shape) == 1 or self.data_type == "tabular":
                output_size = 1
            else:
                output_size = attributions[i].shape[-1]

            total_values = None
            for _ in range(self.num_permutations):
                masks = []
                for k in range(output_size):
                    mask = np.zeros(mask_shape, dtype=np.bool)
                    masks.append(mask.copy().flatten())

                    if output_size != 1:
                        test_attributions = sample_attributions[:,k]
                    else:
                        test_attributions = sample_attributions

                    ordered_inds = np.arange(len(test_attributions))

                    # shuffle the indexes so we get a random permutation ordering
                    if row_clustering is not None:
                        inds_mask = np.ones(len(test_attributions), dtype=np.bool)
                        partition_tree_shuffle(ordered_inds, inds_mask, row_clustering)
                    else:
                        np.random.shuffle(ordered_inds)

                    #ordered_inds = np.random.permutation(len(test_attributions))
                    increment = max(1, int(feature_size * step_fraction))
                    for j in range(0, feature_size, increment):
                        oind_list = [ordered_inds[l] for l in range(j, min(feature_size, j+increment))]
                        for oind in oind_list:
                            if self.data_type == "image":
                                xoind, yoind = oind // attributions[i].shape[1], oind % attributions[i].shape[1]
                                mask[xoind][yoind] = True
                            else:
                                mask[oind] = True

                        masks.append(mask.copy().flatten())

                mask_vals.append(masks)

                mask_size = len(range(0, feature_size, increment)) + 1
                values = []
                masks_arr = np.array(masks)
                for j in range(0, len(masks_arr), self.batch_size):
                    values.append(masked_model(masks_arr[j:j + self.batch_size]))
                values = np.concatenate(values)
                base_value = values[0]
                for l, v in enumerate(values):
                    values[l] = (v - (base_value + np.sum(test_attributions[masks_arr[l]])))**2

                if total_values is None:
                    total_values = values
                else:
                    total_values += values
            total_values /= self.num_permutations
            if len(indices) == 0:
                outputs = range(output_size)
            else:
                outputs = indices

            index = 0
            for k in outputs:
                if output_size == 1:
                    svals.append(total_values[index:index+mask_size])
                else:
                    svals.append(total_values[index:index+mask_size,k])
                index += mask_size

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

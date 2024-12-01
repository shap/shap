import queue
import time

import numpy as np
from numba import njit
from tqdm.auto import tqdm

import math
from itertools import chain, combinations, product

from .. import Explanation, links
from ..models import Model
from ..utils import MaskedModel, OpChain, make_masks, safe_isinstance
from ._explainer import Explainer


class PartitionExplainer(Explainer):
    """Uses the Partition SHAP method to explain the output of any function.

    Partition SHAP computes Shapley values recursively through a hierarchy of features, this
    hierarchy defines feature coalitions and results in the Owen values from game theory.

    The PartitionExplainer has two particularly nice properties:

    1) PartitionExplainer is model-agnostic but when using a balanced partition tree only has
       quadratic exact runtime (in term of the number of input features). This is in contrast to the
       exponential exact runtime of KernelExplainer or SamplingExplainer.
    2) PartitionExplainer always assigns to groups of correlated features the credit that set of features
       would have had if treated as a group. This means if the hierarchical clustering given to
       PartitionExplainer groups correlated features together, then feature correlations are
       "accounted for" in the sense that the total credit assigned to a group of tightly dependent features
       does not depend on how they behave if their correlation structure was broken during the explanation's
       perturbation process.

    Note that for linear models the Owen values that PartitionExplainer returns are the same as the standard
    non-hierarchical Shapley values.
    """

    def __init__(
        self,
        model,
        masker,
        *,
        output_names=None,
        link=links.identity,
        linearize_link=True,
        feature_names=None,
        partition_tree=None,
        **call_args,
    ):
        """Build a PartitionExplainer for the given model with the given masker.

        Parameters
        ----------
        model : function
            User supplied function that takes a matrix of samples (# samples x # features) and
            computes the output of the model for those samples.

        masker : function or numpy.array or pandas.DataFrame or tokenizer
            The function used to "mask" out hidden features of the form `masker(mask, x)`. It takes a
            single input sample and a binary mask and returns a matrix of masked samples. These
            masked samples will then be evaluated using the model function and the outputs averaged.
            As a shortcut for the standard masking using by SHAP you can pass a background data matrix
            instead of a function and that matrix will be used for masking. Domain specific masking
            functions are available in shap such as shap.maksers.Image for images and shap.maskers.Text
            for text.

        partition_tree : None or function or numpy.array
            A hierarchical clustering of the input features represented by a matrix that follows the format
            used by scipy.cluster.hierarchy (see the notebooks_html/partition_explainer directory an example).
            If this is a function then the function produces a clustering matrix when given a single input
            example. If you are using a standard SHAP masker object then you can pass masker.clustering
            to use that masker's built-in clustering of the features, or if partition_tree is None then
            masker.clustering will be used by default.

        Examples
        --------
        See `Partition explainer examples <https://shap.readthedocs.io/en/latest/api_examples/explainers/PartitionExplainer.html>`_

        """
        super().__init__(
            model,
            masker,
            link=link,
            linearize_link=linearize_link,
            algorithm="partition",
            output_names=output_names,
            feature_names=feature_names,
        )

        # convert dataframes
        # if isinstance(masker, pd.DataFrame):
        #     masker = TabularMasker(masker)
        # elif isinstance(masker, np.ndarray) and len(masker.shape) == 2:
        #     masker = TabularMasker(masker)
        # elif safe_isinstance(masker, "transformers.PreTrainedTokenizer"):
        #     masker = TextMasker(masker)
        # self.masker = masker

        # TODO: maybe? if we have a tabular masker then we build a PermutationExplainer that we
        # will use for sampling
        self.input_shape = masker.shape[1:] if hasattr(masker, "shape") and not callable(masker.shape) else None
        # self.output_names = output_names
        if not safe_isinstance(self.model, "shap.models.Model"):
            self.model = Model(self.model)

        self.expected_value = None
        self._curr_base_value = None

        if self.input_shape is not None and len(self.input_shape) > 1:
            self._reshaped_model = lambda x: self.model(x.reshape(x.shape[0], *self.input_shape))
        else:
            self._reshaped_model = self.model

        self.partition_tree = partition_tree

        if not callable(self.masker.clustering):
            self._clustering = self.masker.clustering
            self._mask_matrix = make_masks(self._clustering)

        # if we have gotten default arguments for the call function we need to wrap ourselves in a new class that
        # has a call function with those new default arguments
        if len(call_args) > 0:

            class PartitionExplainer(self.__class__):
                # this signature should match the __call__ signature of the class defined below
                def __call__(
                    self,
                    *args,
                    max_evals=500,
                    fixed_context=None,
                    main_effects=False,
                    error_bounds=False,
                    batch_size="auto",
                    outputs=None,
                    silent=False,
                ):
                    return super().__call__(
                        *args,
                        max_evals=max_evals,
                        fixed_context=fixed_context,
                        main_effects=main_effects,
                        error_bounds=error_bounds,
                        batch_size=batch_size,
                        outputs=outputs,
                        silent=silent,
                    )

            PartitionExplainer.__call__.__doc__ = self.__class__.__call__.__doc__
            self.__class__ = PartitionExplainer
            for k, v in call_args.items():
                self.__call__.__kwdefaults__[k] = v

    # note that changes to this function signature should be copied to the default call argument wrapper above
    def __call__(
        self,
        *args,
        max_evals=500,
        fixed_context=None,
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
            fixed_context=fixed_context,
            main_effects=main_effects,
            error_bounds=error_bounds,
            batch_size=batch_size,
            outputs=outputs,
            silent=silent,
        )

    def explain_row(
        self,
        *row_args,
        max_evals,
        main_effects,
        error_bounds,
        batch_size,
        outputs,
        silent,
        fixed_context="auto",
    ):
        """Explains a single row and returns the tuple (row_values, row_expected_values, row_mask_shapes)."""
        if fixed_context == "auto":
            # if isinstance(self.masker, maskers.Text):
            #     fixed_context = 1 # we err on the side of speed for text models
            # else:
            fixed_context = None
        elif fixed_context not in [0, 1, None]:
            raise ValueError(f"Unknown fixed_context value passed (must be 0, 1 or None): {fixed_context}")

        # build a masked version of the model for the current input sample
        fm = MaskedModel(self.model, self.masker, self.link, self.linearize_link, *row_args)

        # make sure we have the base value and current value outputs
        M = len(fm)
        m00 = np.zeros(M, dtype=bool)
        # if not fixed background or no base value assigned then compute base value for a row
        if self._curr_base_value is None or not getattr(self.masker, "fixed_background", False):
            self._curr_base_value = fm(m00.reshape(1, -1), zero_index=0)[
                0
            ]  # the zero index param tells the masked model what the baseline is
        f11 = fm(~m00.reshape(1, -1))[0]

        if self.partition_tree is not None:
            return self.explain_with_partition_tree(fm, self._curr_base_value, outputs)
        else:
            return self.explain_with_clustering(
                fm, self._curr_base_value, f11, max_evals, outputs, fixed_context, batch_size, silent, row_args
            )

    def explain_with_clustering(self, fm, f00, f11, max_evals, outputs, fixed_context, batch_size, silent, row_args):
        if callable(self.masker.clustering):
            self._clustering = self.masker.clustering(*row_args)
            self._mask_matrix = make_masks(self._clustering)
        M = len(fm)
        # m00 = np.zeros(M, dtype=bool)
        if hasattr(self._curr_base_value, "shape") and len(self._curr_base_value.shape) > 0:
            if outputs is None:
                outputs = np.arange(len(self._curr_base_value))
            elif isinstance(outputs, OpChain):
                outputs = outputs.apply(Explanation(f11)).values

            out_shape = (2 * self._clustering.shape[0] + 1, len(outputs))
        else:
            out_shape = (2 * self._clustering.shape[0] + 1,)

        if max_evals == "auto":
            max_evals = 500

        self.values = np.zeros(out_shape)
        self.dvalues = np.zeros(out_shape)

        self.winter(
            fm,
            self._curr_base_value,
            f11,
            max_evals - 2,
            outputs,
            fixed_context,
            batch_size,
            silent,
        )
        self.values[:] = self.dvalues
        lower_credit(len(self.dvalues) - 1, 0, M, self.values, self._clustering)
        return {
            "values": self.values[:M].copy(),
            "expected_values": self._curr_base_value if outputs is None else self._curr_base_value[outputs],
            "mask_shapes": [s + out_shape[1:] for s in fm.mask_shapes],
            "main_effects": None,
            "hierarchical_values": self.dvalues.copy(),
            "clustering": self._clustering,
            "output_indices": outputs,
            "output_names": getattr(self.model, "output_names", None),
        }

    def explain_with_partition_tree(self, fm, f00, outputs):
        # Step 1: Generate all unique masks
        self.root = Node("Root")
        build_tree(self.partition_tree, self.root)
        self.combinations_list = generate_paths_and_combinations(self.root)
        # print("self.combinations",self.combinations_list)
        self.masks, self.keys = create_masks1(self.root, self.masker.feature_names)
        self.masks_dict = dict(zip(self.keys, self.masks))
        self.mask_permutations = create_combined_masks(self.combinations_list, self.masks_dict)
        # print("mask_permutations", self.mask_permutations, "\n")
        self.masks_list = [mask for _, mask, _ in self.mask_permutations]
        self.unique_masks_set = set(map(tuple, self.masks_list))
        self.unique_masks = [np.array(mask) for mask in self.unique_masks_set]

        # Step 2: Compute model results for all unique masks
        mask_results = {}
        for mask in self.unique_masks:
            result = fm(mask.reshape(1, -1))
            mask_results[tuple(mask)] = result

        # Step 3: Compute marginals for permutations
        shap_values = np.zeros(len(fm))
        last_key_to_off_indexes, last_key_to_on_indexes, weights = map_combinations_to_unique_masks(
            self.mask_permutations, self.unique_masks
        )

        feature_name_to_index = {name: idx for idx, name in enumerate(self.masker.feature_names)}
        # collected_weights = []
        # Step 4: Implement Owen values weighting
        for last_key in last_key_to_off_indexes:
            off_indexes = last_key_to_off_indexes[last_key]
            on_indexes = last_key_to_on_indexes[last_key]
            weight_list = weights[last_key]

            for off_index, on_index, weight in zip(off_indexes, on_indexes, weight_list):
                # print(weight)
                off_result = mask_results[tuple(self.unique_masks[off_index])]
                on_result = mask_results[tuple(self.unique_masks[on_index])]
                # print(off_result)
                # print(on_result)
                # print("weight before calculation", weight) # this might be interesting to plot
                # collected_weights.append(weight)

                marginal_contribution = (on_result - off_result) * weight
                # print(marginal_contribution)
                shap_values[feature_name_to_index[last_key]] += marginal_contribution  # .item()

        # Step 5: Return results
        return {
            "values": shap_values.copy(),
            "expected_values": f00,
            "mask_shapes": [s + () for s in fm.mask_shapes],
            "main_effects": None,
            "hierarchical_values": shap_values,
            "clustering": None,
            "output_indices": outputs,
            "output_names": getattr(self.model, "output_names", None),
        }

    def __str__(self):
        return "shap.explainers.PartitionExplainer()"

    def winter(self, fm, f00, f11, max_evals, output_indexes, fixed_context, batch_size, silent):
        """Compute a nested set of recursive Owen values based on an ordering recursion."""
        M = len(fm)
        m00 = np.zeros(M, dtype=bool)
        base_value = f00
        ind = len(self.dvalues) - 1

        if output_indexes is not None:
            f00 = f00[output_indexes]
            f11 = f11[output_indexes]

        q = queue.PriorityQueue()
        q.put((0, 0, (m00, f00, f11, ind, 1.0)))
        eval_count = 0
        total_evals = min(max_evals, (M - 1) * M)
        pbar = None
        start_time = time.time()
        while not q.empty():
            if eval_count >= max_evals:
                while not q.empty():
                    m00, f00, f11, ind, weight = q.get()[2]
                    self.dvalues[ind] += (f11 - f00) * weight
                break

            # create a batch of work to do
            batch_args = []
            batch_masks = []
            while not q.empty() and len(batch_masks) < batch_size and eval_count + len(batch_masks) < max_evals:
                # get our next set of arguments
                m00, f00, f11, ind, weight = q.get()[2]

                # get the left and right children of this cluster
                lind = int(self._clustering[ind - M, 0]) if ind >= M else -1
                rind = int(self._clustering[ind - M, 1]) if ind >= M else -1

                # get the distance of this cluster's children
                if ind < M:
                    distance = -1
                else:
                    if self._clustering.shape[1] >= 3:
                        distance = self._clustering[ind - M, 2]
                    else:
                        distance = 1

                # check if we are a leaf node (or other negative distance cluster) and so should terminate our decent
                if distance < 0:
                    self.dvalues[ind] += (f11 - f00) * weight
                    continue

                # build the masks
                m10 = m00.copy()  # we separate the copy from the add so as to not get converted to a matrix
                m10[:] += self._mask_matrix[lind, :]
                m01 = m00.copy()
                m01[:] += self._mask_matrix[rind, :]

                batch_args.append((m00, m10, m01, f00, f11, ind, lind, rind, weight))
                batch_masks.append(m10)
                batch_masks.append(m01)

            batch_masks = np.array(batch_masks)

            # run the batch
            if len(batch_args) > 0:
                fout = fm(batch_masks)
                if output_indexes is not None:
                    fout = fout[:, output_indexes]

                eval_count += len(batch_masks)

                if pbar is None and time.time() - start_time > 5:
                    pbar = tqdm(total=total_evals, disable=silent, leave=False)
                    pbar.update(eval_count)
                if pbar is not None:
                    pbar.update(len(batch_masks))

            # use the results of the batch to add new nodes
            for i in range(len(batch_args)):
                m00, m10, m01, f00, f11, ind, lind, rind, weight = batch_args[i]

                # get the evaluated model output on the two new masked inputs
                f10 = fout[2 * i]
                f01 = fout[2 * i + 1]

                new_weight = weight
                if fixed_context is None:
                    new_weight /= 2

                if fixed_context is None or fixed_context == 0:
                    args = (m00, f00, f10, lind, new_weight)
                    q.put((-np.max(np.abs(f10 - f00)) * new_weight, np.random.randn(), args))

                    # recurse on the right node with zero context
                    args = (m00, f00, f01, rind, new_weight)
                    q.put((-np.max(np.abs(f01 - f00)) * new_weight, np.random.randn(), args))

                if fixed_context is None or fixed_context == 1:
                    # recurse on the left node with one context
                    args = (m01, f01, f11, lind, new_weight)
                    q.put((-np.max(np.abs(f11 - f01)) * new_weight, np.random.randn(), args))

                    # recurse on the right node with one context
                    args = (m10, f10, f11, rind, new_weight)
                    q.put((-np.max(np.abs(f11 - f10)) * new_weight, np.random.randn(), args))

        if pbar is not None:
            pbar.close()

        self.last_eval_count = eval_count

        return output_indexes, base_value

    def owen3(self, fm, f00, f11, max_evals, output_indexes, fixed_context, batch_size, silent):
        """Compute a nested set of recursive Owen values based on an ordering recursion."""
        # f = self._reshaped_model
        # r = self.masker
        # masks = np.zeros(2*len(inds)+1, dtype=int)
        M = len(fm)
        m00 = np.zeros(M, dtype=bool)
        # f00 = fm(m00.reshape(1,-1))[0]
        base_value = f00
        # f11 = fm(~m00.reshape(1,-1))[0]
        # f11 = self._reshaped_model(r(~m00, x)).mean(0)
        ind = len(self.dvalues) - 1

        # make sure output_indexes is a list of indexes
        if output_indexes is not None:
            # assert self.multi_output, "output_indexes is only valid for multi-output models!"
            # inds = output_indexes.apply(f11, 0)
            # out_len = output_indexes_len(output_indexes)
            # if output_indexes.startswith("max("):
            #     output_indexes = np.argsort(-f11)[:out_len]
            # elif output_indexes.startswith("min("):
            #     output_indexes = np.argsort(f11)[:out_len]
            # elif output_indexes.startswith("max(abs("):
            #     output_indexes = np.argsort(np.abs(f11))[:out_len]

            f00 = f00[output_indexes]
            f11 = f11[output_indexes]

        # our starting plan is to evaluate all the nodes with a fixed_context
        evals_planned = M

        q = queue.PriorityQueue()
        q.put((0, 0, (m00, f00, f11, ind, 1.0, fixed_context)))  # (m00, f00, f11, tree_index, weight)
        eval_count = 0
        total_evals = min(
            max_evals, (M - 1) * M
        )  # TODO: (M-1)*M is only right for balanced clusterings, but this is just for plotting progress...
        pbar = None
        start_time = time.time()
        while not q.empty():
            # if we passed our execution limit then leave everything else on the internal nodes
            if eval_count >= max_evals:
                while not q.empty():
                    m00, f00, f11, ind, weight, _ = q.get()[2]
                    self.dvalues[ind] += (f11 - f00) * weight
                break

            # create a batch of work to do
            batch_args = []
            batch_masks = []
            while not q.empty() and len(batch_masks) < batch_size and eval_count < max_evals:
                # get our next set of arguments
                m00, f00, f11, ind, weight, context = q.get()[2]

                # get the left and right children of this cluster
                lind = int(self._clustering[ind - M, 0]) if ind >= M else -1
                rind = int(self._clustering[ind - M, 1]) if ind >= M else -1

                # get the distance of this cluster's children
                if ind < M:
                    distance = -1
                else:
                    distance = self._clustering[ind - M, 2]

                # check if we are a leaf node (or other negative distance cluster) and so should terminate our decent
                if distance < 0:
                    self.dvalues[ind] += (f11 - f00) * weight
                    continue

                # build the masks
                m10 = m00.copy()  # we separate the copy from the add so as to not get converted to a matrix
                m10[:] += self._mask_matrix[lind, :]
                m01 = m00.copy()
                m01[:] += self._mask_matrix[rind, :]

                batch_args.append((m00, m10, m01, f00, f11, ind, lind, rind, weight, context))
                batch_masks.append(m10)
                batch_masks.append(m01)

            batch_masks = np.array(batch_masks)

            # run the batch
            if len(batch_args) > 0:
                fout = fm(batch_masks)
                if output_indexes is not None:
                    fout = fout[:, output_indexes]

                eval_count += len(batch_masks)

                if pbar is None and time.time() - start_time > 5:
                    pbar = tqdm(total=total_evals, disable=silent, leave=False)
                    pbar.update(eval_count)
                if pbar is not None:
                    pbar.update(len(batch_masks))

            # use the results of the batch to add new nodes
            for i in range(len(batch_args)):
                m00, m10, m01, f00, f11, ind, lind, rind, weight, context = batch_args[i]

                # get the the number of leaves in this cluster
                if ind < M:
                    num_leaves = 0
                else:
                    num_leaves = self._clustering[ind - M, 3]

                # get the evaluated model output on the two new masked inputs
                f10 = fout[2 * i]
                f01 = fout[2 * i + 1]

                # see if we have enough evaluations left to get both sides of a fixed context
                if max_evals - evals_planned > num_leaves:
                    evals_planned += num_leaves
                    ignore_context = True
                else:
                    ignore_context = False

                new_weight = weight
                if context is None or ignore_context:
                    new_weight /= 2

                if context is None or context == 0 or ignore_context:
                    self.dvalues[ind] += (
                        f11 - f10 - f01 + f00
                    ) * weight  # leave the interaction effect on the internal node

                    # recurse on the left node with zero context, flip the context for all descendents if we are ignoring it
                    args = (m00, f00, f10, lind, new_weight, 0 if context == 1 else context)
                    q.put((-np.max(np.abs(f10 - f00)) * new_weight, np.random.randn(), args))

                    # recurse on the right node with zero context, flip the context for all descendents if we are ignoring it
                    args = (m00, f00, f01, rind, new_weight, 0 if context == 1 else context)
                    q.put((-np.max(np.abs(f01 - f00)) * new_weight, np.random.randn(), args))

                if context is None or context == 1 or ignore_context:
                    self.dvalues[ind] -= (
                        f11 - f10 - f01 + f00
                    ) * weight  # leave the interaction effect on the internal node

                    # recurse on the left node with one context, flip the context for all descendents if we are ignoring it
                    args = (m01, f01, f11, lind, new_weight, 1 if context == 0 else context)
                    q.put((-np.max(np.abs(f11 - f01)) * new_weight, np.random.randn(), args))

                    # recurse on the right node with one context, flip the context for all descendents if we are ignoring it
                    args = (m10, f10, f11, rind, new_weight, 1 if context == 0 else context)
                    q.put((-np.max(np.abs(f11 - f10)) * new_weight, np.random.randn(), args))

        if pbar is not None:
            pbar.close()

        self.last_eval_count = eval_count

        return output_indexes, base_value

    # def owen2(self, fm, f00, f11, max_evals, output_indexes, fixed_context, batch_size, silent):
    #     """ Compute a nested set of recursive Owen values based on an ordering recursion.
    #     """

    #     #f = self._reshaped_model
    #     #r = self.masker
    #     #masks = np.zeros(2*len(inds)+1, dtype=int)
    #     M = len(fm)
    #     m00 = np.zeros(M, dtype=bool)
    #     #f00 = fm(m00.reshape(1,-1))[0]
    #     base_value = f00
    #     #f11 = fm(~m00.reshape(1,-1))[0]
    #     #f11 = self._reshaped_model(r(~m00, x)).mean(0)
    #     ind = len(self.dvalues)-1

    #     # make sure output_indexes is a list of indexes
    #     if output_indexes is not None:
    #         # assert self.multi_output, "output_indexes is only valid for multi-output models!"
    #         # inds = output_indexes.apply(f11, 0)
    #         # out_len = output_indexes_len(output_indexes)
    #         # if output_indexes.startswith("max("):
    #         #     output_indexes = np.argsort(-f11)[:out_len]
    #         # elif output_indexes.startswith("min("):
    #         #     output_indexes = np.argsort(f11)[:out_len]
    #         # elif output_indexes.startswith("max(abs("):
    #         #     output_indexes = np.argsort(np.abs(f11))[:out_len]

    #         f00 = f00[output_indexes]
    #         f11 = f11[output_indexes]

    #     fc_owen(m00, m11, 1)
    #     fc_owen(m00, m11, 0)

    #     def fc_owen(m00, m11, context):

    #         # recurse on the left node with zero context
    #         args = (m00, f00, f10, lind, new_weight)
    #         q.put((-np.max(np.abs(f10 - f00)) * new_weight, np.random.randn(), args))

    #         # recurse on the right node with zero context
    #         args = (m00, f00, f01, rind, new_weight)
    #         q.put((-np.max(np.abs(f01 - f00)) * new_weight, np.random.randn(), args))
    #         fc_owen(m00, m11, 1)
    #     m00 m11
    #     owen(fc=1)
    #     owen(fc=0)

    #     q = queue.PriorityQueue()
    #     q.put((0, 0, (m00, f00, f11, ind, 1.0, 1)))
    #     eval_count = 0
    #     total_evals = min(max_evals, (M-1)*M) # TODO: (M-1)*M is only right for balanced clusterings, but this is just for plotting progress...
    #     pbar = None
    #     start_time = time.time()
    #     while not q.empty():

    #         # if we passed our execution limit then leave everything else on the internal nodes
    #         if eval_count >= max_evals:
    #             while not q.empty():
    #                 m00, f00, f11, ind, weight, _ = q.get()[2]
    #                 self.dvalues[ind] += (f11 - f00) * weight
    #             break

    #         # create a batch of work to do
    #         batch_args = []
    #         batch_masks = []
    #         while not q.empty() and len(batch_masks) < batch_size and eval_count < max_evals:

    #             # get our next set of arguments
    #             m00, f00, f11, ind, weight, context = q.get()[2]

    #             # get the left and right children of this cluster
    #             lind = int(self._clustering[ind-M, 0]) if ind >= M else -1
    #             rind = int(self._clustering[ind-M, 1]) if ind >= M else -1

    #             # get the distance of this cluster's children
    #             if ind < M:
    #                 distance = -1
    #             else:
    #                 if self._clustering.shape[1] >= 3:
    #                     distance = self._clustering[ind-M, 2]
    #                 else:
    #                     distance = 1

    #             # check if we are a leaf node (or other negative distance cluster) and so should terminate our decent
    #             if distance < 0:
    #                 self.dvalues[ind] += (f11 - f00) * weight
    #                 continue

    #             # build the masks
    #             m10 = m00.copy() # we separate the copy from the add so as to not get converted to a matrix
    #             m10[:] += self._mask_matrix[lind, :]
    #             m01 = m00.copy()
    #             m01[:] += self._mask_matrix[rind, :]

    #             batch_args.append((m00, m10, m01, f00, f11, ind, lind, rind, weight, context))
    #             batch_masks.append(m10)
    #             batch_masks.append(m01)

    #         batch_masks = np.array(batch_masks)

    #         # run the batch
    #         if len(batch_args) > 0:
    #             fout = fm(batch_masks)
    #             if output_indexes is not None:
    #                 fout = fout[:,output_indexes]

    #             eval_count += len(batch_masks)

    #             if pbar is None and time.time() - start_time > 5:
    #                 pbar = tqdm(total=total_evals, disable=silent, leave=False)
    #                 pbar.update(eval_count)
    #             if pbar is not None:
    #                 pbar.update(len(batch_masks))

    #         # use the results of the batch to add new nodes
    #         for i in range(len(batch_args)):

    #             m00, m10, m01, f00, f11, ind, lind, rind, weight, context = batch_args[i]

    #             # get the evaluated model output on the two new masked inputs
    #             f10 = fout[2*i]
    #             f01 = fout[2*i+1]

    #             new_weight = weight
    #             if fixed_context is None:
    #                 new_weight /= 2
    #             elif fixed_context == 0:
    #                 self.dvalues[ind] += (f11 - f10 - f01 + f00) * weight # leave the interaction effect on the internal node
    #             elif fixed_context == 1:
    #                 self.dvalues[ind] -= (f11 - f10 - f01 + f00) * weight # leave the interaction effect on the internal node

    #             if fixed_context is None or fixed_context == 0:
    #                 self.dvalues[ind] += (f11 - f10 - f01 + f00) * weight # leave the interaction effect on the internal node

    #                 # recurse on the left node with zero context
    #                 args = (m00, f00, f10, lind, new_weight)
    #                 q.put((-np.max(np.abs(f10 - f00)) * new_weight, np.random.randn(), args))

    #                 # recurse on the right node with zero context
    #                 args = (m00, f00, f01, rind, new_weight)
    #                 q.put((-np.max(np.abs(f01 - f00)) * new_weight, np.random.randn(), args))

    #             if fixed_context is None or fixed_context == 1:
    #                 self.dvalues[ind] -= (f11 - f10 - f01 + f00) * weight # leave the interaction effect on the internal node

    #                 # recurse on the left node with one context
    #                 args = (m01, f01, f11, lind, new_weight)
    #                 q.put((-np.max(np.abs(f11 - f01)) * new_weight, np.random.randn(), args))

    #                 # recurse on the right node with one context
    #                 args = (m10, f10, f11, rind, new_weight)
    #                 q.put((-np.max(np.abs(f11 - f10)) * new_weight, np.random.randn(), args))

    #     if pbar is not None:
    #         pbar.close()

    #     return output_indexes, base_value


def output_indexes_len(output_indexes):
    if output_indexes.startswith("max("):
        return int(output_indexes[4:-1])
    elif output_indexes.startswith("min("):
        return int(output_indexes[4:-1])
    elif output_indexes.startswith("max(abs("):
        return int(output_indexes[8:-2])
    elif not isinstance(output_indexes, str):
        return len(output_indexes)


@njit
def lower_credit(i, value, M, values, clustering):
    if i < M:
        values[i] += value
        return
    li = int(clustering[i - M, 0])
    ri = int(clustering[i - M, 1])
    group_size = int(clustering[i - M, 3])
    lsize = int(clustering[li - M, 3]) if li >= M else 1
    rsize = int(clustering[ri - M, 3]) if ri >= M else 1
    assert lsize + rsize == group_size
    values[i] += value
    lower_credit(li, values[i] * lsize / group_size, M, values, clustering)
    lower_credit(ri, values[i] * rsize / group_size, M, values, clustering)


class Node:
    def __init__(self, key):
        self.key = key
        self.child = []
        self.permutations = []  # this may not be the greatest idea??
        self.weights = []

    def __repr__(self):
        return f"({self.key}): {self.child} -> {self.permutations} \\ {self.weights}"


# This function is to encode the dictionary to our specific structure
def build_tree(d, root):
    if isinstance(d, dict):
        for key, value in d.items():
            node = Node(key)
            root.child.append(node)
            build_tree(value, node)
    elif isinstance(d, list):
        for item in d:
            node = Node(item)
            root.child.append(node)
    # get all the sibling permutations
    generate_permutations(root)


# generate all permutations of sibling nodes and assign it to the nodes
def generate_permutations(node):
    if not node.child:  # Leaf node
        node.permutations = []
        return

    children_keys = [child.key for child in node.child]
    node.permutations = {}

    for i, child in enumerate(node.child):
        excluded = children_keys[:i] + children_keys[i + 1 :]
        generate_permutations(child)

        # Generate all unique combinations of permutations for each child
        child.permutations = list(all_subsets(excluded))
        # print(len(children_keys))
        # print([len(permutation) for permutation in child.permutations])
        child.weights = [compute_weight(len(children_keys), len(permutation)) for permutation in child.permutations]
        # print(child.weights)


def compute_weight(total, selected):
    return 1 / (total * math.comb(total - 1, selected))


def all_subsets(iterable):
    return chain.from_iterable(combinations(iterable, n) for n in range(len(iterable) + 1))


def get_all_leaf_values(node):
    leaves = []
    if not node.child:
        leaves.append(node.key)
    else:
        for child in node.child:
            leaves.extend(get_all_leaf_values(child))
    return leaves


def create_masks1(node, columns):
    masks = [np.zeros(len(columns), dtype=bool)]
    keys = [()]

    if not node.child:
        mask = columns == node.key
        masks.append(mask)
        keys.append(node.key)
    else:
        current_node_mask = columns.isin(get_all_leaf_values(node))
        masks.append(current_node_mask)
        keys.append(node.key)

        for subset in node.child:
            child_masks, child_keys = create_masks1(subset, columns)
            masks.extend(child_masks)
            keys.extend(child_keys)

    return masks, keys


def generate_paths_and_combinations(node):
    paths = []

    def dfs(current_node, current_path):
        current_path.append((current_node.key, current_node.permutations, current_node.weights))

        if not current_node.child:  # Leaf node
            paths.append(current_path[:])  # Make a copy of current_path
        else:
            for child in current_node.child:
                dfs(child, current_path)

        current_path.pop()  # Backtrack

    dfs(node, [])

    combinations_list = []
    # print("the paths",paths)

    for path in paths:
        filtered_path = [(key, perms, weight) for key, perms, weight in path if perms]
        # print("filtered_path", filtered_path, "\n")
        if filtered_path:
            node_keys, permutations, weights = zip(*filtered_path)
            path_combinations = list(product(*permutations))
            weight_combinations = list(product(*weights))
            # print("path combos", path_combinations, len(path_combinations))
            # print("the weight combs", weight_combinations, len(weight_combinations))
            weight_products = [np.prod(weight_tuple) for weight_tuple in weight_combinations]
            # print(weight_products)

            last_key = node_keys[-1]
            for i, combination in enumerate(path_combinations):
                combinations_list.append((last_key, combination, weight_products[i]))

    return combinations_list


def combine_masks(masks):
    combined_mask = np.logical_or.reduce(masks)
    return combined_mask


def create_combined_masks(combinations, masks_dict):
    combined_masks = []
    for last_key, combination, weights in combinations:
        masks = []
        for keys in combination:
            if isinstance(keys, tuple) and not keys:
                continue
            for key in keys:
                if key in masks_dict:
                    masks.append(masks_dict[key])

        if masks:
            combined_mask = combine_masks(masks)
            combined_masks.append((last_key, combined_mask, weights))

            if last_key in masks_dict:
                combined_mask_with_last_key = combine_masks(masks + [masks_dict[last_key]])
                combined_masks.append((last_key, combined_mask_with_last_key, weights))
        else:
            combined_mask = np.zeros_like(list(masks_dict.values())[0])
            combined_masks.append((last_key, combined_mask, weights))

            if last_key in masks_dict:
                combined_mask_with_last_key = combine_masks([combined_mask, masks_dict[last_key]])
                combined_masks.append((last_key, combined_mask_with_last_key, weights))
    return combined_masks


def map_combinations_to_unique_masks(combined_masks, unique_masks):
    unique_mask_index_map = {tuple(mask): idx for idx, mask in enumerate(unique_masks)}
    last_key_to_off_indexes = {}
    last_key_to_on_indexes = {}
    weights = {}

    for i, (last_key, combined_mask, weight) in enumerate(combined_masks):
        mask_tuple = tuple(combined_mask)
        unique_index = unique_mask_index_map[mask_tuple]

        if i % 2 == 0:  # Even index -> OFF value
            if last_key not in last_key_to_off_indexes:
                last_key_to_off_indexes[last_key] = []
                weights[last_key] = []
            last_key_to_off_indexes[last_key].append(unique_index)
            weights[last_key].append(weight)
        else:  # Odd index -> ON value
            if last_key not in last_key_to_on_indexes:
                last_key_to_on_indexes[last_key] = []
            last_key_to_on_indexes[last_key].append(unique_index)

    return last_key_to_off_indexes, last_key_to_on_indexes, weights


def create_partition_hierarchy(linkage_matrix, columns):
    def build_hierarchy(node, linkage_matrix, columns):
        if node < len(columns):
            return {columns[node]: columns[node]}
        else:
            left_child = int(linkage_matrix[node - len(columns), 0])
            right_child = int(linkage_matrix[node - len(columns), 1])
            left_subtree = build_hierarchy(left_child, linkage_matrix, columns)
            right_subtree = build_hierarchy(right_child, linkage_matrix, columns)
            return {f"cluster_{node}": {**left_subtree, **right_subtree}}

    root_node = len(linkage_matrix) + len(columns) - 1
    hierarchy = build_hierarchy(root_node, linkage_matrix, columns)
    return hierarchy[f"cluster_{root_node}"]

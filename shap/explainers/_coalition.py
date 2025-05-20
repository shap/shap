# mypy: ignore-errors
from __future__ import annotations

import math
from itertools import chain, combinations, product

import numpy as np  # numpy base

from .. import links  # shap modules
from ..explainers._explainer import Explainer
from ..models import Model
from ..utils import MaskedModel, make_masks, safe_isinstance


class CoalitionExplainer(Explainer):
    """A coalition-based explainer that uses Winter values, also called recursive Owen values, to explain model predictions.

    This explainer implements a coalition-based approach to compute feature attributions
    using Winter values, which extend Shapley values to handle hierarchical feature groupings.
    Essentially the attributions are computed using the marginals respecting the partition tree, reducing the complexity of computation.

    It is particularly useful when features can be grouped into coalitions or
    hierarchies, in the case of temporal, multimodal data (e.g., demographic features, financial features, etc.).
    Textual and image data is not yet implemented.

    The explainer supports both single and multi-output models, and can handle various
    types of input data through the provided masker.

    Example usage
    --------
    >>> import shap
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import load_iris
    >>>
    >>> # Load data and train model
    >>> X, y = load_iris(return_X_y=True)
    >>> model = RandomForestClassifier().fit(X, y)
    >>>
    >>> # Define feature groups
    >>> coalition_tree = {
    ...     "Sepal": ["sepal length (cm)", "sepal width (cm)"],
    ...     "Petal": ["petal length (cm)", "petal width (cm)"]
    ... }
    >>> # Define feature names, or you can pass X as a DataFrame
    >>> feature_names = ["sepal length (cm)", "sepal width (cm)",
                 "petal length (cm)", "petal width (cm)"]
    >>> masker = shap.maskers.Partition(X)
    >>> masker.feature_names = feature_names
    >>>
    >>> # Create explainer
    >>> explainer = shap.CoalitionExplainer(
    ...     model.predict,
    ...     masker,
    ...     partition_tree=coalition_tree
    ... )
    >>>
    >>> # Compute SHAP values
    >>> shap_values = explainer(X[:5]
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
    ):
        """Initialize the coalition explainer with a model and masker.

        Parameters
        ----------
        model : callable or shap.models.Model
            A callable that takes a matrix of samples (# samples x # features) and
            computes the output of the model for those samples. The output can be a vector
            (# samples) or a matrix (# samples x # outputs).

        masker : shap.maskers.Masker
            A masker object that defines how to mask features and compute background
            values. This should be compatible with the input data format.

        output_names : list of str, optional
            Names for each of the model outputs. If None, the output names will be
            determined from the model if possible.

        link : callable, optional
            The link function used to map between the output units of the model and the
            SHAP value units. By default, the identity function is used.

        linearize_link : bool, optional
            If True, the link function is linearized around the expected value to
            improve the accuracy of the SHAP values. Default is True.

        feature_names : list of str, required
            Names for each of the input features. If None, feature names will be
            determined from the masker if possible.

        partition_tree : dict, required
            A dictionary defining the hierarchical grouping of features. Each key
            represents a group name, and its value is either a list of feature names
            or another dictionary defining subgroups, note all input features must be included in the leaf nodes.
            For example:
            {
                "Demographics": ["Age", "Gender", "Education"],
                "Financial": {
                    "Income": ["Salary", "Bonus"],
                    "Assets": ["Savings", "Investments"]
                }
            }

        Notes
        -----
        - The explainer supports both single and multi-output models.
        - The partition_tree parameter is used to define feature coalitions for
          computing Owen values, which can provide more meaningful explanations
          when features are naturally grouped.
        - The masker should be compatible with the input data format and provide
          appropriate background values for computing SHAP values.
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
        self.input_shape = masker.shape[1:] if hasattr(masker, "shape") and not callable(masker.shape) else None
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
        **kwargs,
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
            **kwargs,
        )

    # mypy: disable=override
    def explain_row(
        self,
        *row_args,
        max_evals=100,
        main_effects=False,
        error_bounds=False,
        batch_size="auto",
        outputs=None,
        silent=False,
        fixed_context="auto",
    ):
        if fixed_context == "auto":
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
            base_output = fm(m00.reshape(1, -1), zero_index=0)[0]
            self._curr_base_value = np.array(base_output) if isinstance(base_output, (list, tuple)) else base_output

        # Handle multi-output predictions
        if isinstance(self._curr_base_value, np.ndarray) and self._curr_base_value.ndim > 0:
            num_outputs = len(self._curr_base_value)
            shap_values = np.zeros((M, num_outputs))
        else:
            num_outputs = 1
            shap_values = np.zeros(M)

        # Step 1: build the hierarchy
        self.root = Node("Root")
        _build_tree(self.partition_tree, self.root)  # generate partition tree specified
        self.combinations_list = _generate_paths_and_combinations(
            self.root
        )  # generate permutations of neighbours consistent with partition tree, and related weights
        self.masks, self.keys = _create_masks(
            self.root, self.masker.feature_names
        )  # turn the premutations into valid masks for inference
        self.masks_dict = dict(zip(self.keys, self.masks))
        self.mask_permutations = _create_combined_masks(
            self.combinations_list, self.masks_dict
        )  # add up masks to leave nodes
        self.masks_list = [mask for _, mask, _ in self.mask_permutations]
        self.unique_masks_set = set(map(tuple, self.masks_list))
        self.unique_masks = [np.array(mask) for mask in self.unique_masks_set]  # unique masks for inference

        # Step 2: Compute model results for all unique masks
        mask_results = {}
        for mask in self.unique_masks:
            result = fm(mask.reshape(1, -1))
            # Ensure result is properly shaped for multi-output
            if isinstance(result, (list, tuple)):
                result = np.array(result)
            elif not isinstance(result, np.ndarray):
                result = np.array([result])
            mask_results[tuple(mask)] = result

        # Step 3: Compute marginals for permutations
        last_key_to_off_indexes, last_key_to_on_indexes, weights = _map_combinations_to_unique_masks(
            self.mask_permutations, self.unique_masks
        )

        feature_name_to_index = {name: idx for idx, name in enumerate(self.masker.feature_names)}

        # Step 4: Implement Owen values weighting
        for last_key in last_key_to_off_indexes:
            off_indexes = last_key_to_off_indexes[last_key]
            on_indexes = last_key_to_on_indexes[last_key]
            weight_list = weights[last_key]

            for off_index, on_index, weight in zip(off_indexes, on_indexes, weight_list):
                off_result = mask_results[tuple(self.unique_masks[off_index])]
                on_result = mask_results[tuple(self.unique_masks[on_index])]

                if num_outputs > 1:
                    # Ensure results are properly shaped for multi-output
                    off_result = np.asarray(off_result).reshape(-1)
                    on_result = np.asarray(on_result).reshape(-1)
                    for i in range(num_outputs):
                        marginal_contribution = float((on_result[i] - off_result[i]) * weight)
                        shap_values[feature_name_to_index[last_key], i] += marginal_contribution
                else:
                    marginal_contribution = float((on_result - off_result) * weight)
                    shap_values[feature_name_to_index[last_key]] += marginal_contribution

        # Step 5: Return results
        return {
            "values": shap_values.copy(),
            "expected_values": self._curr_base_value,
            "mask_shapes": [s + () for s in fm.mask_shapes],
            "main_effects": None,
            "hierarchical_values": shap_values,
            "clustering": None,
            "output_indices": outputs,
            "output_names": getattr(self.model, "output_names", None),
        }

    def __str__(self):
        return "shap.explainers.CoalitionExplainer()"


####################### HELPER FUNCTIONS THAT PROBABLY CAN STAY####################
class Node:
    def __init__(self, key):
        self.key = key
        self.child = []
        self.permutations = []  # this may not be the greatest idea??
        self.weights = []

    def __repr__(self):
        return f"({self.key}): {self.child} -> {self.permutations} \\ {self.weights}"


# This function is to encode the dictionary to our specific structure
def _build_tree(d, root):
    if isinstance(d, dict):
        for key, value in d.items():
            node = Node(key)
            root.child.append(node)
            _build_tree(value, node)
    elif isinstance(d, list):
        for item in d:
            node = Node(item)
            root.child.append(node)
    # get all the sibling permutations
    _generate_permutations(root)


def create_partition_hierarchy(
    linkage_matrix, columns
):  # this is a helper to turn scipy linkage matrix to partition_tree dict
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


def _combine_masks(masks):
    combined_mask = np.logical_or.reduce(masks)
    return combined_mask


def _compute_weight(total, selected):
    return 1 / (total * math.comb(total - 1, selected))


def _all_subsets(iterable):
    return chain.from_iterable(combinations(iterable, n) for n in range(len(iterable) + 1))


def _get_all_leaf_values(node):
    leaves = []
    if not node.child:
        leaves.append(node.key)
    else:
        for child in node.child:
            leaves.extend(_get_all_leaf_values(child))
    return leaves


# generate all permutations of sibling nodes and assign it to the nodes
def _generate_permutations(node):
    if not node.child:  # Leaf node
        node.permutations = []
        return

    children_keys = [child.key for child in node.child]
    node.permutations = {}

    for i, child in enumerate(node.child):
        excluded = children_keys[:i] + children_keys[i + 1 :]
        _generate_permutations(child)

        # Generate all unique combinations of permutations for each child
        child.permutations = list(_all_subsets(excluded))
        # print(len(children_keys))
        # print([len(permutation) for permutation in child.permutations])
        child.weights = [_compute_weight(len(children_keys), len(permutation)) for permutation in child.permutations]
        # print(child.weights)


##########################################################


def _create_masks(node, columns):
    masks = [np.zeros(len(columns), dtype=bool)]
    keys = [()]

    if not node.child:
        if hasattr(columns, "isin"):
            mask = columns.isin([node.key])
        else:
            mask = np.array([col == node.key for col in columns])
        masks.append(mask)
        keys.append(node.key)
    else:
        if hasattr(columns, "isin"):
            current_node_mask = columns.isin(_get_all_leaf_values(node))
        else:
            leaf_values = _get_all_leaf_values(node)
            current_node_mask = np.array([col in leaf_values for col in columns])
        masks.append(current_node_mask)
        keys.append(node.key)

        for subset in node.child:
            child_masks, child_keys = _create_masks(subset, columns)
            masks.extend(child_masks)
            keys.extend(child_keys)

    return masks, keys


def _generate_paths_and_combinations(node):
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

    for path in paths:
        filtered_path = [(key, perms, weight) for key, perms, weight in path if perms]

        if filtered_path:
            node_keys, permutations, weights = zip(*filtered_path)
            path_combinations = list(product(*permutations))
            weight_combinations = list(product(*weights))

            weight_products = [np.prod(weight_tuple) for weight_tuple in weight_combinations]

            last_key = node_keys[-1]
            for i, combination in enumerate(path_combinations):
                combinations_list.append((last_key, combination, weight_products[i]))

    return combinations_list


def _create_combined_masks(combinations, masks_dict):
    combined_masks = []
    for last_key, combination, weights in combinations:
        masks = []
        for keys in combination:
            if isinstance(keys, tuple) and not keys:
                continue
            for key in keys:
                if key in masks_dict:
                    masks.append(masks_dict[key])

        if len(masks) > 0:
            combined_mask = _combine_masks(masks)
            combined_masks.append((last_key, combined_mask, weights))

            if last_key in masks_dict:
                combined_mask_with_last_key = _combine_masks(masks + [masks_dict[last_key]])
                combined_masks.append((last_key, combined_mask_with_last_key, weights))
        else:
            combined_mask = np.zeros_like(list(masks_dict.values())[0])
            combined_masks.append((last_key, combined_mask, weights))

            if last_key in masks_dict:
                combined_mask_with_last_key = _combine_masks([combined_mask, masks_dict[last_key]])
                combined_masks.append((last_key, combined_mask_with_last_key, weights))
    return combined_masks


def _map_combinations_to_unique_masks(combined_masks, unique_masks):
    unique_mask_index_map = {tuple(mask): idx for idx, mask in enumerate(unique_masks)}
    last_key_to_off_indexes: dict[int | str, list[int]] = {}
    last_key_to_on_indexes: dict[int | str, list[int]] = {}
    weights: dict[int | str, list[int]] = {}

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

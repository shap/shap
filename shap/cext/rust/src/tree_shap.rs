use crate::types::*;
use rayon::prelude::*;

/// Extend the decision path with a fraction of one and zero extensions
#[inline]
pub fn extend_path(
    unique_path: &mut [PathElement],
    unique_depth: usize,
    zero_fraction: TFloat,
    one_fraction: TFloat,
    feature_index: i32,
) {
    unique_path[unique_depth].feature_index = feature_index;
    unique_path[unique_depth].zero_fraction = zero_fraction;
    unique_path[unique_depth].one_fraction = one_fraction;
    unique_path[unique_depth].pweight = if unique_depth == 0 { 1.0 } else { 0.0 };

    for i in (0..unique_depth).rev() {
        unique_path[i + 1].pweight += one_fraction * unique_path[i].pweight * (i + 1) as TFloat
            / (unique_depth + 1) as TFloat;
        unique_path[i].pweight = zero_fraction * unique_path[i].pweight * (unique_depth - i) as TFloat
            / (unique_depth + 1) as TFloat;
    }
}

/// Undo a previous extension of the decision path
#[inline]
pub fn unwind_path(
    unique_path: &mut [PathElement],
    unique_depth: usize,
    path_index: usize,
) {
    let one_fraction = unique_path[path_index].one_fraction;
    let zero_fraction = unique_path[path_index].zero_fraction;
    let mut next_one_portion = unique_path[unique_depth].pweight;

    for i in (0..unique_depth).rev() {
        if one_fraction != 0.0 {
            let tmp = unique_path[i].pweight;
            unique_path[i].pweight = next_one_portion * (unique_depth + 1) as TFloat
                / ((i + 1) as TFloat * one_fraction);
            next_one_portion = tmp - unique_path[i].pweight * zero_fraction * (unique_depth - i) as TFloat
                / (unique_depth + 1) as TFloat;
        } else {
            unique_path[i].pweight = (unique_path[i].pweight * (unique_depth + 1) as TFloat)
                / (zero_fraction * (unique_depth - i) as TFloat);
        }
    }

    for i in path_index..unique_depth {
        unique_path[i] = unique_path[i + 1];
    }
}

/// Determine what the total permutation weight would be if we unwound a previous extension
#[inline]
pub fn unwound_path_sum(
    unique_path: &[PathElement],
    unique_depth: usize,
    path_index: usize,
) -> TFloat {
    let one_fraction = unique_path[path_index].one_fraction;
    let zero_fraction = unique_path[path_index].zero_fraction;
    let mut next_one_portion = unique_path[unique_depth].pweight;
    let mut total = 0.0;

    if one_fraction != 0.0 {
        for i in (0..unique_depth).rev() {
            let tmp = next_one_portion / ((i + 1) as TFloat * one_fraction);
            total += tmp;
            next_one_portion = unique_path[i].pweight - tmp * zero_fraction * (unique_depth - i) as TFloat;
        }
    } else {
        for i in (0..unique_depth).rev() {
            total += unique_path[i].pweight / (zero_fraction * (unique_depth - i) as TFloat);
        }
    }
    total * (unique_depth + 1) as TFloat
}

/// Recursive computation of SHAP values for a decision tree
#[allow(clippy::too_many_arguments)]
pub fn tree_shap_recursive(
    num_outputs: u32,
    children_left: &[i32],
    children_right: &[i32],
    children_default: &[i32],
    features: &[i32],
    thresholds: &[TFloat],
    threshold_types: &[i32],
    values: &[TFloat],
    node_sample_weight: &[TFloat],
    x: &[TFloat],
    x_missing: &[bool],
    phi: &mut [TFloat],
    node_index: usize,
    unique_depth: usize,
    parent_unique_path: &mut [PathElement],
    parent_zero_fraction: TFloat,
    parent_one_fraction: TFloat,
    parent_feature_index: i32,
    condition: i32,
    condition_feature: u32,
    condition_fraction: TFloat,
) {
    // Stop if we have no weight coming down to us
    if condition_fraction == 0.0 {
        return;
    }

    // Extend the unique path
    let unique_path_start = unique_depth + 1;
    let path_len = unique_depth + 1;

    // Copy parent path to current path
    parent_unique_path.copy_within(0..path_len, unique_path_start);

    let unique_path = &mut parent_unique_path[unique_path_start..];

    if condition == 0 || condition_feature != parent_feature_index as u32 {
        extend_path(unique_path, unique_depth, parent_zero_fraction, parent_one_fraction, parent_feature_index);
    }

    let split_index = features[node_index] as usize;

    // Leaf node
    if children_right[node_index] < 0 {
        for i in 1..=unique_depth {
            let w = unwound_path_sum(unique_path, unique_depth, i);
            let el = &unique_path[i];
            let phi_offset = el.feature_index as usize * num_outputs as usize;
            let values_offset = node_index * num_outputs as usize;
            let scale = w * (el.one_fraction - el.zero_fraction) * condition_fraction;

            for j in 0..num_outputs as usize {
                phi[phi_offset + j] += scale * values[values_offset + j];
            }
        }
    } else {
        // Internal node - find which branch is "hot"
        let threshold_type = threshold_types[node_index];
        let hot_index = if x_missing[split_index] {
            children_default[node_index] as usize
        } else if threshold_type == 0 && x[split_index] <= thresholds[node_index] {
            children_left[node_index] as usize
        } else if threshold_type == 1 && category_in_threshold(thresholds[node_index], x[split_index]) {
            children_left[node_index] as usize
        } else {
            children_right[node_index] as usize
        };

        let cold_index = if hot_index == children_left[node_index] as usize {
            children_right[node_index] as usize
        } else {
            children_left[node_index] as usize
        };

        let w = node_sample_weight[node_index];
        let hot_zero_fraction = node_sample_weight[hot_index] / w;
        let cold_zero_fraction = node_sample_weight[cold_index] / w;
        let mut incoming_zero_fraction = 1.0;
        let mut incoming_one_fraction = 1.0;

        // See if we have already split on this feature
        let mut path_index = 0;
        while path_index <= unique_depth {
            if unique_path[path_index].feature_index as usize == split_index {
                break;
            }
            path_index += 1;
        }

        let mut current_unique_depth = unique_depth;
        if path_index != unique_depth + 1 {
            incoming_zero_fraction = unique_path[path_index].zero_fraction;
            incoming_one_fraction = unique_path[path_index].one_fraction;
            unwind_path(unique_path, unique_depth, path_index);
            current_unique_depth -= 1;
        }

        // Divide up the condition_fraction among the recursive calls
        let mut hot_condition_fraction = condition_fraction;
        let mut cold_condition_fraction = condition_fraction;

        if condition > 0 && split_index == condition_feature as usize {
            cold_condition_fraction = 0.0;
            current_unique_depth -= 1;
        } else if condition < 0 && split_index == condition_feature as usize {
            hot_condition_fraction *= hot_zero_fraction;
            cold_condition_fraction *= cold_zero_fraction;
            current_unique_depth -= 1;
        }

        // Recurse down hot path
        tree_shap_recursive(
            num_outputs,
            children_left,
            children_right,
            children_default,
            features,
            thresholds,
            threshold_types,
            values,
            node_sample_weight,
            x,
            x_missing,
            phi,
            hot_index,
            current_unique_depth + 1,
            parent_unique_path,
            hot_zero_fraction * incoming_zero_fraction,
            incoming_one_fraction,
            split_index as i32,
            condition,
            condition_feature,
            hot_condition_fraction,
        );

        // Recurse down cold path
        tree_shap_recursive(
            num_outputs,
            children_left,
            children_right,
            children_default,
            features,
            thresholds,
            threshold_types,
            values,
            node_sample_weight,
            x,
            x_missing,
            phi,
            cold_index,
            current_unique_depth + 1,
            parent_unique_path,
            cold_zero_fraction * incoming_zero_fraction,
            0.0,
            split_index as i32,
            condition,
            condition_feature,
            cold_condition_fraction,
        );
    }
}

/// Compute SHAP values for a single tree
pub fn tree_shap(
    tree: &TreeEnsemble,
    data: &ExplanationDataset,
    out_contribs: &mut [TFloat],
    condition: i32,
    condition_feature: u32,
) {
    // Update the reference value with the expected value of the tree's predictions
    if condition == 0 {
        for j in 0..tree.num_outputs as usize {
            out_contribs[data.M as usize * tree.num_outputs as usize + j] += tree.values[j];
        }
    }

    // Pre-allocate space for the unique path data
    let maxd = (tree.max_depth + 2) as usize;
    let mut unique_path_data = vec![PathElement::new(-1, 0.0, 0.0, 0.0); (maxd * (maxd + 1)) / 2];

    tree_shap_recursive(
        tree.num_outputs,
        &tree.children_left,
        &tree.children_right,
        &tree.children_default,
        &tree.features,
        &tree.thresholds,
        &tree.thresholds_types,
        &tree.values,
        &tree.node_sample_weights,
        &data.X,
        &data.X_missing,
        out_contribs,
        0,
        0,
        &mut unique_path_data,
        1.0,
        1.0,
        -1,
        condition,
        condition_feature,
        1.0,
    );
}

/// Tree prediction helper
#[inline]
pub fn tree_predict<'a>(
    tree_idx: usize,
    trees: &'a TreeEnsemble,
    x: &[TFloat],
    x_missing: &[bool],
) -> &'a [TFloat] {
    let offset = tree_idx * trees.max_nodes as usize;
    let mut node = 0;

    loop {
        let pos = offset + node;
        let feature = trees.features[pos] as usize;

        // We hit a leaf, return a pointer to the values
        if trees.is_leaf(pos) {
            return &trees.values[pos * trees.num_outputs as usize..];
        }

        // Otherwise we are at an internal node and need to recurse
        node = if x_missing[feature] {
            trees.children_default[pos] as usize
        } else if trees.thresholds_types[pos] == 0 && x[feature] <= trees.thresholds[pos] {
            trees.children_left[pos] as usize
        } else if trees.thresholds_types[pos] == 1 && category_in_threshold(trees.thresholds[pos], x[feature]) {
            trees.children_left[pos] as usize
        } else {
            trees.children_right[pos] as usize
        };
    }
}

/// Dense tree prediction with optional transformation
pub fn dense_tree_predict(
    out: &mut [TFloat],
    trees: &TreeEnsemble,
    data: &ExplanationDataset,
    model_transform: u32,
) {
    let transform = get_transform(model_transform);

    for i in 0..data.num_X as usize {
        let x_start = i * data.M as usize;
        let x = &data.X[x_start..x_start + data.M as usize];
        let x_missing = &data.X_missing[x_start..x_start + data.M as usize];
        let row_out = &mut out[i * trees.num_outputs as usize..(i + 1) * trees.num_outputs as usize];

        // Add the base offset
        for k in 0..trees.num_outputs as usize {
            row_out[k] += trees.base_offset[k];
        }

        // Add the leaf values from each tree
        for j in 0..trees.tree_limit as usize {
            let leaf_value = tree_predict(j, trees, x, x_missing);
            for k in 0..trees.num_outputs as usize {
                row_out[k] += leaf_value[k];
            }
        }

        // Apply any needed transform
        if let Some(transform_fn) = transform {
            let y_i = if data.y.is_empty() { 0.0 } else { data.y[i] };
            for k in 0..trees.num_outputs as usize {
                row_out[k] = transform_fn(row_out[k], y_i);
            }
        }
    }
}

/// Dense tree path dependent SHAP computation (multi-threaded)
pub fn dense_tree_path_dependent(
    trees: &TreeEnsemble,
    data: &ExplanationDataset,
    out_contribs: &mut [TFloat],
    _model_transform: u32,
) {
    // Calculate chunk size for each sample's output
    let chunk_size = (data.M as usize + 1) * trees.num_outputs as usize;

    // Build explanation for each sample IN PARALLEL
    out_contribs
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(i, instance_out_contribs)| {
            // Create instance view for this sample
            let instance = ExplanationDataset {
                X: data.X[i * data.M as usize..(i + 1) * data.M as usize].to_vec(),
                X_missing: data.X_missing[i * data.M as usize..(i + 1) * data.M as usize].to_vec(),
                y: if data.y.is_empty() { vec![] } else { vec![data.y[i]] },
                R: data.R.clone(),
                R_missing: data.R_missing.clone(),
                num_X: 1,
                M: data.M,
                num_R: data.num_R,
            };

            // Aggregate the effect of explaining each tree
            for j in 0..trees.tree_limit as usize {
                // Create a single-tree ensemble for this tree
                let tree_ensemble = TreeEnsemble {
                    children_left: trees.children_left[j * trees.max_nodes as usize..(j + 1) * trees.max_nodes as usize].to_vec(),
                    children_right: trees.children_right[j * trees.max_nodes as usize..(j + 1) * trees.max_nodes as usize].to_vec(),
                    children_default: trees.children_default[j * trees.max_nodes as usize..(j + 1) * trees.max_nodes as usize].to_vec(),
                    features: trees.features[j * trees.max_nodes as usize..(j + 1) * trees.max_nodes as usize].to_vec(),
                    thresholds: trees.thresholds[j * trees.max_nodes as usize..(j + 1) * trees.max_nodes as usize].to_vec(),
                    thresholds_types: trees.thresholds_types[j * trees.max_nodes as usize..(j + 1) * trees.max_nodes as usize].to_vec(),
                    values: trees.values[j * trees.max_nodes as usize * trees.num_outputs as usize..(j + 1) * trees.max_nodes as usize * trees.num_outputs as usize].to_vec(),
                    node_sample_weights: trees.node_sample_weights[j * trees.max_nodes as usize..(j + 1) * trees.max_nodes as usize].to_vec(),
                    max_depth: trees.max_depth,
                    tree_limit: 1,
                    base_offset: trees.base_offset.clone(),
                    max_nodes: trees.max_nodes,
                    num_outputs: trees.num_outputs,
                };
                tree_shap(&tree_ensemble, &instance, instance_out_contribs, 0, 0);
            }

            // Apply the base offset to the bias term
            for j in 0..trees.num_outputs as usize {
                instance_out_contribs[data.M as usize * trees.num_outputs as usize + j] += trees.base_offset[j];
            }
        });
}

/// Main entry point for dense tree SHAP computation
pub fn dense_tree_shap(
    trees: &TreeEnsemble,
    data: &ExplanationDataset,
    out_contribs: &mut [TFloat],
    feature_dependence: u32,
    model_transform: u32,
    _interactions: bool,
) {
    match feature_dependence {
        feature_dependence::TREE_PATH_DEPENDENT => {
            dense_tree_path_dependent(trees, data, out_contribs, model_transform);
        }
        _ => {
            eprintln!("Feature dependence mode {} not yet implemented in Rust", feature_dependence);
        }
    }
}

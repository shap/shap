use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyArrayMethods, PyUntypedArrayMethods};

mod types;
mod tree_shap;

use types::*;

/// Simple test function to verify Rust-Python binding works
#[pyfunction]
fn test_add(a: f64, b: f64) -> f64 {
    a + b
}

/// Dense tree SHAP computation - Python interface
/// Matches the signature expected by Python's TreeExplainer
#[pyfunction]
#[pyo3(signature = (children_left, children_right, children_default, features, thresholds, threshold_types, values, node_sample_weights, max_depth, X, X_missing, y, R, R_missing, tree_limit, base_offset, out_contribs, feature_perturbation, output_transform, _less_than_or_equal))]
fn dense_tree_shap<'py>(
    _py: Python<'py>,
    children_left: PyReadonlyArray1<i32>,
    children_right: PyReadonlyArray1<i32>,
    children_default: PyReadonlyArray1<i32>,
    features: PyReadonlyArray1<i32>,
    thresholds: PyReadonlyArray1<f64>,
    threshold_types: PyReadonlyArray1<i32>,
    values: PyReadonlyArray1<f64>,
    node_sample_weights: PyReadonlyArray1<f64>,
    max_depth: u32,
    X: PyReadonlyArray2<f64>,
    X_missing: PyReadonlyArray2<bool>,
    y: PyReadonlyArray1<f64>,
    R: PyReadonlyArray2<f64>,
    R_missing: PyReadonlyArray2<bool>,
    tree_limit: u32,
    base_offset: PyReadonlyArray1<f64>,
    out_contribs: &Bound<'py, PyArray2<f64>>,
    feature_perturbation: u32,
    output_transform: u32,
    _less_than_or_equal: bool,
) -> PyResult<()> {
    // Determine max_nodes and num_outputs from array shapes
    let num_outputs = base_offset.len() as u32;
    let total_nodes = children_left.len();
    let max_nodes = if tree_limit > 0 {
        (total_nodes / tree_limit as usize) as u32
    } else {
        total_nodes as u32
    };

    // Convert numpy arrays to Rust vectors
    let trees = TreeEnsemble {
        children_left: children_left.as_array().to_vec(),
        children_right: children_right.as_array().to_vec(),
        children_default: children_default.as_array().to_vec(),
        features: features.as_array().to_vec(),
        thresholds: thresholds.as_array().to_vec(),
        thresholds_types: threshold_types.as_array().to_vec(),
        values: values.as_array().to_vec(),
        node_sample_weights: node_sample_weights.as_array().to_vec(),
        max_depth,
        tree_limit,
        base_offset: base_offset.as_array().to_vec(),
        max_nodes,
        num_outputs,
    };

    let x_shape = X.shape();
    let num_X = x_shape[0] as u32;
    let M = x_shape[1] as u32;

    let r_shape = R.shape();
    let num_R = r_shape[0] as u32;

    let data = ExplanationDataset {
        X: X.as_array().to_owned().into_raw_vec_and_offset().0,
        X_missing: X_missing.as_array().to_owned().into_raw_vec_and_offset().0,
        y: y.as_array().to_vec(),
        R: R.as_array().to_owned().into_raw_vec_and_offset().0,
        R_missing: R_missing.as_array().to_owned().into_raw_vec_and_offset().0,
        num_X,
        M,
        num_R,
    };

    // Get mutable access to output array
    let mut out_slice = unsafe { out_contribs.as_slice_mut()? };

    // Call the Rust tree SHAP implementation
    tree_shap::dense_tree_shap(
        &trees,
        &data,
        out_slice,
        feature_perturbation,
        output_transform,
        false, // interactions not yet supported
    );

    Ok(())
}

/// Dense tree prediction - Python interface
#[pyfunction]
#[pyo3(signature = (children_left, children_right, children_default, features, thresholds, threshold_types, values, max_depth, tree_limit, base_offset, output_transform, X, X_missing, y, out_predictions))]
fn dense_tree_predict<'py>(
    _py: Python<'py>,
    children_left: PyReadonlyArray1<i32>,
    children_right: PyReadonlyArray1<i32>,
    children_default: PyReadonlyArray1<i32>,
    features: PyReadonlyArray1<i32>,
    thresholds: PyReadonlyArray1<f64>,
    threshold_types: PyReadonlyArray1<i32>,
    values: PyReadonlyArray1<f64>,
    max_depth: u32,
    tree_limit: u32,
    base_offset: PyReadonlyArray1<f64>,
    output_transform: u32,
    X: PyReadonlyArray2<f64>,
    X_missing: PyReadonlyArray2<bool>,
    y: PyReadonlyArray1<f64>,
    out_predictions: &Bound<'py, PyArray1<f64>>,
) -> PyResult<()> {
    let num_outputs = base_offset.len() as u32;
    let total_nodes = children_left.len();
    let max_nodes = if tree_limit > 0 {
        (total_nodes / tree_limit as usize) as u32
    } else {
        total_nodes as u32
    };

    let trees = TreeEnsemble {
        children_left: children_left.as_array().to_vec(),
        children_right: children_right.as_array().to_vec(),
        children_default: children_default.as_array().to_vec(),
        features: features.as_array().to_vec(),
        thresholds: thresholds.as_array().to_vec(),
        thresholds_types: threshold_types.as_array().to_vec(),
        values: values.as_array().to_vec(),
        node_sample_weights: vec![1.0; total_nodes], // Not used in prediction
        max_depth,
        tree_limit,
        base_offset: base_offset.as_array().to_vec(),
        max_nodes,
        num_outputs,
    };

    let x_shape = X.shape();
    let num_X = x_shape[0] as u32;
    let M = x_shape[1] as u32;

    let data = ExplanationDataset {
        X: X.as_array().to_owned().into_raw_vec_and_offset().0,
        X_missing: X_missing.as_array().to_owned().into_raw_vec_and_offset().0,
        y: y.as_array().to_vec(),
        R: vec![],
        R_missing: vec![],
        num_X,
        M,
        num_R: 0,
    };

    let mut out_slice = unsafe { out_predictions.as_slice_mut()? };
    tree_shap::dense_tree_predict(out_slice, &trees, &data, output_transform);

    Ok(())
}

/// Dense tree Saabas approximation - Python interface
#[pyfunction]
#[pyo3(signature = (children_left, children_right, children_default, features, thresholds, threshold_types, values, max_depth, tree_limit, base_offset, output_transform, X, X_missing, y, out_contribs))]
fn dense_tree_saabas<'py>(
    _py: Python<'py>,
    children_left: PyReadonlyArray1<i32>,
    children_right: PyReadonlyArray1<i32>,
    children_default: PyReadonlyArray1<i32>,
    features: PyReadonlyArray1<i32>,
    thresholds: PyReadonlyArray1<f64>,
    threshold_types: PyReadonlyArray1<i32>,
    values: PyReadonlyArray1<f64>,
    max_depth: u32,
    tree_limit: u32,
    base_offset: PyReadonlyArray1<f64>,
    output_transform: u32,
    X: PyReadonlyArray2<f64>,
    X_missing: PyReadonlyArray2<bool>,
    y: PyReadonlyArray1<f64>,
    out_contribs: &Bound<'py, PyArray2<f64>>,
) -> PyResult<()> {
    // Saabas is a simpler approximation - for now just use tree_path_dependent
    // TODO: Implement proper Saabas algorithm
    let num_outputs = base_offset.len() as u32;
    let total_nodes = children_left.len();
    let max_nodes = if tree_limit > 0 {
        (total_nodes / tree_limit as usize) as u32
    } else {
        total_nodes as u32
    };

    let trees = TreeEnsemble {
        children_left: children_left.as_array().to_vec(),
        children_right: children_right.as_array().to_vec(),
        children_default: children_default.as_array().to_vec(),
        features: features.as_array().to_vec(),
        thresholds: thresholds.as_array().to_vec(),
        thresholds_types: threshold_types.as_array().to_vec(),
        values: values.as_array().to_vec(),
        node_sample_weights: vec![1.0; total_nodes],
        max_depth,
        tree_limit,
        base_offset: base_offset.as_array().to_vec(),
        max_nodes,
        num_outputs,
    };

    let x_shape = X.shape();
    let num_X = x_shape[0] as u32;
    let M = x_shape[1] as u32;

    let data = ExplanationDataset {
        X: X.as_array().to_owned().into_raw_vec_and_offset().0,
        X_missing: X_missing.as_array().to_owned().into_raw_vec_and_offset().0,
        y: y.as_array().to_vec(),
        R: vec![],
        R_missing: vec![],
        num_X,
        M,
        num_R: 0,
    };

    let mut out_slice = unsafe { out_contribs.as_slice_mut()? };

    // Use tree_path_dependent as approximation for now
    tree_shap::dense_tree_shap(
        &trees,
        &data,
        out_slice,
        1, // TREE_PATH_DEPENDENT
        output_transform,
        false,
    );

    Ok(())
}

/// Update node sample weights (for background data)
#[pyfunction]
#[pyo3(signature = (_children_left, _children_right, _children_default, _features, _thresholds, _threshold_types, _values, _node_sample_weights, _max_depth, _tree_limit, _R, _R_missing))]
fn dense_tree_update_weights<'py>(
    _py: Python<'py>,
    _children_left: PyReadonlyArray1<i32>,
    _children_right: PyReadonlyArray1<i32>,
    _children_default: PyReadonlyArray1<i32>,
    _features: PyReadonlyArray1<i32>,
    _thresholds: PyReadonlyArray1<f64>,
    _threshold_types: PyReadonlyArray1<i32>,
    _values: PyReadonlyArray1<f64>,
    _node_sample_weights: &Bound<'py, PyArray1<f64>>,
    _max_depth: u32,
    _tree_limit: u32,
    _R: PyReadonlyArray2<f64>,
    _R_missing: PyReadonlyArray2<bool>,
) -> PyResult<()> {
    // TODO: Implement proper weight update
    // For now, this is a placeholder
    Ok(())
}

/// Compute expected values
#[pyfunction]
#[pyo3(signature = (_children_left, _children_right, _children_default, _features, _thresholds, _threshold_types, _values, _node_sample_weights, max_depth, _tree_limit, base_offset, _num_outputs, _R, _R_missing, out_expected_values))]
fn compute_expectations<'py>(
    _py: Python<'py>,
    _children_left: PyReadonlyArray1<i32>,
    _children_right: PyReadonlyArray1<i32>,
    _children_default: PyReadonlyArray1<i32>,
    _features: PyReadonlyArray1<i32>,
    _thresholds: PyReadonlyArray1<f64>,
    _threshold_types: PyReadonlyArray1<i32>,
    _values: PyReadonlyArray1<f64>,
    _node_sample_weights: PyReadonlyArray1<f64>,
    max_depth: u32,
    _tree_limit: u32,
    base_offset: PyReadonlyArray1<f64>,
    _num_outputs: u32,
    _R: PyReadonlyArray2<f64>,
    _R_missing: PyReadonlyArray2<bool>,
    out_expected_values: &Bound<'py, PyArray1<f64>>,
) -> PyResult<u32> {
    // TODO: Implement proper expectation computation
    // For now, just return base offset
    let out_slice = unsafe { out_expected_values.as_slice_mut()? };
    let base = base_offset.as_array();
    for (i, &val) in base.iter().enumerate() {
        out_slice[i] = val;
    }
    Ok(max_depth)
}

/// Python module for Tree SHAP (Rust implementation)
#[pymodule]
fn _cext(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(test_add, m)?)?;
    m.add_function(wrap_pyfunction!(dense_tree_shap, m)?)?;
    m.add_function(wrap_pyfunction!(dense_tree_predict, m)?)?;
    m.add_function(wrap_pyfunction!(dense_tree_saabas, m)?)?;
    // TODO: Re-enable these when fully implemented
    // m.add_function(wrap_pyfunction!(dense_tree_update_weights, m)?)?;
    // m.add_function(wrap_pyfunction!(compute_expectations, m)?)?;
    Ok(())
}

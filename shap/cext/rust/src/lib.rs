use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyArrayMethods, PyUntypedArrayMethods};

mod types;
mod tree_shap;

use types::*;
use tree_shap::*;

/// Simple test function to verify Rust-Python binding works
#[pyfunction]
fn test_add(a: f64, b: f64) -> f64 {
    a + b
}

/// Dense tree SHAP computation - Python interface
#[pyfunction]
#[pyo3(signature = (children_left, children_right, children_default, features, thresholds, threshold_types, values, node_sample_weights, max_depth, tree_limit, base_offset, max_nodes, num_outputs, X, X_missing, out_contribs, feature_dependence=1, model_transform=0, interactions=false))]
fn py_dense_tree_shap<'py>(
    py: Python<'py>,
    children_left: PyReadonlyArray1<i32>,
    children_right: PyReadonlyArray1<i32>,
    children_default: PyReadonlyArray1<i32>,
    features: PyReadonlyArray1<i32>,
    thresholds: PyReadonlyArray1<f64>,
    threshold_types: PyReadonlyArray1<i32>,
    values: PyReadonlyArray1<f64>,
    node_sample_weights: PyReadonlyArray1<f64>,
    max_depth: u32,
    tree_limit: u32,
    base_offset: PyReadonlyArray1<f64>,
    max_nodes: u32,
    num_outputs: u32,
    X: PyReadonlyArray2<f64>,
    X_missing: PyReadonlyArray2<bool>,
    out_contribs: &Bound<'py, PyArray2<f64>>,
    feature_dependence: u32,
    model_transform: u32,
    interactions: bool,
) -> PyResult<()> {
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

    let data = ExplanationDataset {
        X: X.as_array().to_owned().into_raw_vec(),
        X_missing: X_missing.as_array().to_owned().into_raw_vec(),
        y: vec![],
        R: vec![],
        R_missing: vec![],
        num_X,
        M,
        num_R: 0,
    };

    // Get mutable access to output array
    let mut out_slice = unsafe { out_contribs.as_slice_mut()? };

    // Call the Rust tree SHAP implementation
    dense_tree_shap(
        &trees,
        &data,
        out_slice,
        feature_dependence,
        model_transform,
        interactions,
    );

    Ok(())
}

/// Dense tree prediction - Python interface
#[pyfunction]
#[pyo3(signature = (children_left, children_right, children_default, features, thresholds, threshold_types, values, node_sample_weights, max_depth, tree_limit, base_offset, max_nodes, num_outputs, X, X_missing, model_transform=0))]
fn py_dense_tree_predict<'py>(
    py: Python<'py>,
    children_left: PyReadonlyArray1<i32>,
    children_right: PyReadonlyArray1<i32>,
    children_default: PyReadonlyArray1<i32>,
    features: PyReadonlyArray1<i32>,
    thresholds: PyReadonlyArray1<f64>,
    threshold_types: PyReadonlyArray1<i32>,
    values: PyReadonlyArray1<f64>,
    node_sample_weights: PyReadonlyArray1<f64>,
    max_depth: u32,
    tree_limit: u32,
    base_offset: PyReadonlyArray1<f64>,
    max_nodes: u32,
    num_outputs: u32,
    X: PyReadonlyArray2<f64>,
    X_missing: PyReadonlyArray2<bool>,
    model_transform: u32,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
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

    let data = ExplanationDataset {
        X: X.as_array().to_owned().into_raw_vec(),
        X_missing: X_missing.as_array().to_owned().into_raw_vec(),
        y: vec![],
        R: vec![],
        R_missing: vec![],
        num_X,
        M,
        num_R: 0,
    };

    let mut out = vec![0.0; (num_X * num_outputs) as usize];
    dense_tree_predict(&mut out, &trees, &data, model_transform);

    Ok(PyArray1::from_vec_bound(py, out))
}

/// Python module for Tree SHAP GPU acceleration
#[pymodule]
fn _cext_gpu(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(test_add, m)?)?;
    m.add_function(wrap_pyfunction!(py_dense_tree_shap, m)?)?;
    m.add_function(wrap_pyfunction!(py_dense_tree_predict, m)?)?;
    Ok(())
}

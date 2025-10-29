use pyo3::prelude::*;

/// Simple test function to verify Rust-Python binding works
#[pyfunction]
fn test_add(a: f64, b: f64) -> f64 {
    a + b
}

/// Python module for Tree SHAP GPU acceleration
#[pymodule]
fn _cext_gpu(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(test_add, m)?)?;
    Ok(())
}

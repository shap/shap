#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include "tree_shap.h"
#include <iostream>

static PyObject *_cext_dense_tree_shap(PyObject *self, PyObject *args);
static PyObject *_cext_dense_tree_predict(PyObject *self, PyObject *args);
static PyObject *_cext_dense_tree_update_weights(PyObject *self, PyObject *args);
static PyObject *_cext_dense_tree_saabas(PyObject *self, PyObject *args);
static PyObject *_cext_compute_expectations(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"dense_tree_shap", _cext_dense_tree_shap, METH_VARARGS, "C implementation of Tree SHAP for dense."},
    {"dense_tree_predict", _cext_dense_tree_predict, METH_VARARGS, "C implementation of tree predictions."},
    {"dense_tree_update_weights", _cext_dense_tree_update_weights, METH_VARARGS, "C implementation of tree node weight compuatations."},
    {"dense_tree_saabas", _cext_dense_tree_saabas, METH_VARARGS, "C implementation of Saabas (rough fast approximation to Tree SHAP)."},
    {"compute_expectations", _cext_compute_expectations, METH_VARARGS, "Compute expectations of internal nodes."},
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_cext",
    "This module provides an interface for a fast Tree SHAP implementation.",
    -1,
    module_methods,
    NULL,
    NULL,
    NULL,
    NULL
};
#endif

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit__cext(void)
#else
PyMODINIT_FUNC init_cext(void)
#endif
{
    #if PY_MAJOR_VERSION >= 3
        PyObject *module = PyModule_Create(&moduledef);
        if (!module) return NULL;
    #else
        PyObject *module = Py_InitModule("_cext", module_methods);
        if (!module) return;
    #endif

    /* Load `numpy` functionality. */
    import_array();

    #if PY_MAJOR_VERSION >= 3
        return module;
    #endif
}

static PyObject *_cext_compute_expectations(PyObject *self, PyObject *args)
{
    PyObject *children_left_obj;
    PyObject *children_right_obj;
    PyObject *node_sample_weight_obj;
    PyObject *values_obj;
    
    /* Parse the input tuple */
    if (!PyArg_ParseTuple(
        args, "OOOO", &children_left_obj, &children_right_obj, &node_sample_weight_obj, &values_obj
    )) return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyArrayObject *children_left_array = (PyArrayObject*)PyArray_FROM_OTF(children_left_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *children_right_array = (PyArrayObject*)PyArray_FROM_OTF(children_right_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *node_sample_weight_array = (PyArrayObject*)PyArray_FROM_OTF(node_sample_weight_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *values_array = (PyArrayObject*)PyArray_FROM_OTF(values_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);

    /* If that didn't work, throw an exception. */
    if (children_left_array == NULL || children_right_array == NULL ||
        values_array == NULL || node_sample_weight_array == NULL) {
        Py_XDECREF(children_left_array);
        Py_XDECREF(children_right_array);
        //PyArray_ResolveWritebackIfCopy(values_array);
        Py_XDECREF(values_array);
        Py_XDECREF(node_sample_weight_array);
        return NULL;
    }

    TreeEnsemble tree;

    // number of outputs
    tree.num_outputs = PyArray_DIM(values_array, 1);

    /* Get pointers to the data as C-types. */
    tree.children_left = (int*)PyArray_DATA(children_left_array);
    tree.children_right = (int*)PyArray_DATA(children_right_array);
    tree.values = (tfloat*)PyArray_DATA(values_array);
    tree.node_sample_weights = (tfloat*)PyArray_DATA(node_sample_weight_array);

    const int max_depth = compute_expectations(tree);

    // clean up the created python objects
    Py_XDECREF(children_left_array);
    Py_XDECREF(children_right_array);
    //PyArray_ResolveWritebackIfCopy(values_array);
    Py_XDECREF(values_array);
    Py_XDECREF(node_sample_weight_array);

    PyObject *ret = Py_BuildValue("i", max_depth);
    return ret;
}


static PyObject *_cext_dense_tree_shap(PyObject *self, PyObject *args)
{
    PyObject *children_left_obj;
    PyObject *children_right_obj;
    PyObject *children_default_obj;
    PyObject *features_obj;
    PyObject *thresholds_obj;
    PyObject *values_obj;
    PyObject *node_sample_weights_obj;
    int max_depth;
    PyObject *X_obj;
    PyObject *X_missing_obj;
    PyObject *y_obj;
    PyObject *R_obj;
    PyObject *R_missing_obj;
    int tree_limit;
    PyObject *out_contribs_obj;
    int feature_dependence;
    int model_output;
    double base_offset;
    bool interactions;
    bool model_stack;
  
    /* Parse the input tuple */
    if (!PyArg_ParseTuple(
        args, "OOOOOOOiOOOOOidOiibb", &children_left_obj, &children_right_obj, &children_default_obj,
        &features_obj, &thresholds_obj, &values_obj, &node_sample_weights_obj,
        &max_depth, &X_obj, &X_missing_obj, &y_obj, &R_obj, &R_missing_obj, &tree_limit, &base_offset,
        &out_contribs_obj, &feature_dependence, &model_output, &interactions, &model_stack
    )) return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyArrayObject *children_left_array = (PyArrayObject*)PyArray_FROM_OTF(children_left_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *children_right_array = (PyArrayObject*)PyArray_FROM_OTF(children_right_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *children_default_array = (PyArrayObject*)PyArray_FROM_OTF(children_default_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *features_array = (PyArrayObject*)PyArray_FROM_OTF(features_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *thresholds_array = (PyArrayObject*)PyArray_FROM_OTF(thresholds_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *values_array = (PyArrayObject*)PyArray_FROM_OTF(values_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *node_sample_weights_array = (PyArrayObject*)PyArray_FROM_OTF(node_sample_weights_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *X_array = (PyArrayObject*)PyArray_FROM_OTF(X_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *X_missing_array = (PyArrayObject*)PyArray_FROM_OTF(X_missing_obj, NPY_BOOL, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *y_array = NULL;
    if (y_obj != Py_None) y_array = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *R_array = NULL;
    if (R_obj != Py_None) R_array = (PyArrayObject*)PyArray_FROM_OTF(R_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *R_missing_array = NULL;
    if (R_missing_obj != Py_None) R_missing_array = (PyArrayObject*)PyArray_FROM_OTF(R_missing_obj, NPY_BOOL, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *out_contribs_array = (PyArrayObject*)PyArray_FROM_OTF(out_contribs_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);

    /* If that didn't work, throw an exception. Note that R and y are optional. */
    if (children_left_array == NULL || children_right_array == NULL ||
        children_default_array == NULL || features_array == NULL || thresholds_array == NULL ||
        values_array == NULL || node_sample_weights_array == NULL || X_array == NULL ||
        X_missing_array == NULL || out_contribs_array == NULL) {
        Py_XDECREF(children_left_array);
        Py_XDECREF(children_right_array);
        Py_XDECREF(children_default_array);
        Py_XDECREF(features_array);
        Py_XDECREF(thresholds_array);
        Py_XDECREF(values_array);
        Py_XDECREF(node_sample_weights_array);
        Py_XDECREF(X_array);
        Py_XDECREF(X_missing_array);
        if (y_array != NULL) Py_XDECREF(y_array);
        if (R_array != NULL) Py_XDECREF(R_array);
        if (R_missing_array != NULL) Py_XDECREF(R_missing_array);
        //PyArray_ResolveWritebackIfCopy(out_contribs_array);
        Py_XDECREF(out_contribs_array);
        return NULL;
    }

    const unsigned num_X = PyArray_DIM(X_array, 0);
    const unsigned M = PyArray_DIM(X_array, 1);
    const unsigned max_nodes = PyArray_DIM(values_array, 1);
    const unsigned num_outputs = PyArray_DIM(values_array, 2);
    unsigned num_R = 0;
    if (R_array != NULL) num_R = PyArray_DIM(R_array, 0);

    // Get pointers to the data as C-types
    int *children_left = (int*)PyArray_DATA(children_left_array);
    int *children_right = (int*)PyArray_DATA(children_right_array);
    int *children_default = (int*)PyArray_DATA(children_default_array);
    int *features = (int*)PyArray_DATA(features_array);
    tfloat *thresholds = (tfloat*)PyArray_DATA(thresholds_array);
    tfloat *values = (tfloat*)PyArray_DATA(values_array);
    tfloat *node_sample_weights = (tfloat*)PyArray_DATA(node_sample_weights_array);
    tfloat *X = (tfloat*)PyArray_DATA(X_array);
    bool *X_missing = (bool*)PyArray_DATA(X_missing_array);
    tfloat *y = NULL;
    if (y_array != NULL) y = (tfloat*)PyArray_DATA(y_array);
    tfloat *R = NULL;
    if (R_array != NULL) R = (tfloat*)PyArray_DATA(R_array);
    bool *R_missing = NULL;
    if (R_missing_array != NULL) R_missing = (bool*)PyArray_DATA(R_missing_array);
    tfloat *out_contribs = (tfloat*)PyArray_DATA(out_contribs_array);

    // these are just a wrapper objects for all the pointers and numbers associated with
    // the ensemble tree model and the datset we are explaing
    TreeEnsemble trees = TreeEnsemble(
        children_left, children_right, children_default, features, thresholds, values,
        node_sample_weights, max_depth, tree_limit, base_offset,
        max_nodes, num_outputs
    );
    ExplanationDataset data = ExplanationDataset(X, X_missing, y, R, R_missing, num_X, M, num_R);

    dense_tree_shap(trees, data, out_contribs, feature_dependence, model_output, interactions, model_stack);

    // retrieve return value before python cleanup of objects
    tfloat ret_value = (double)values[0];

    // clean up the created python objects 
    Py_XDECREF(children_left_array);
    Py_XDECREF(children_right_array);
    Py_XDECREF(children_default_array);
    Py_XDECREF(features_array);
    Py_XDECREF(thresholds_array);
    Py_XDECREF(values_array);
    Py_XDECREF(node_sample_weights_array);
    Py_XDECREF(X_array);
    Py_XDECREF(X_missing_array);
    if (y_array != NULL) Py_XDECREF(y_array);
    if (R_array != NULL) Py_XDECREF(R_array);
    if (R_missing_array != NULL) Py_XDECREF(R_missing_array);
    //PyArray_ResolveWritebackIfCopy(out_contribs_array);
    Py_XDECREF(out_contribs_array);

    /* Build the output tuple */
    PyObject *ret = Py_BuildValue("d", ret_value);
    return ret;
}


static PyObject *_cext_dense_tree_predict(PyObject *self, PyObject *args)
{
    PyObject *children_left_obj;
    PyObject *children_right_obj;
    PyObject *children_default_obj;
    PyObject *features_obj;
    PyObject *thresholds_obj;
    PyObject *values_obj;
    int max_depth;
    int tree_limit;
    double base_offset;
    int model_output;
    PyObject *X_obj;
    PyObject *X_missing_obj;
    PyObject *y_obj;
    PyObject *out_pred_obj;
    
    
  
    /* Parse the input tuple */
    if (!PyArg_ParseTuple(
        args, "OOOOOOiidiOOOO", &children_left_obj, &children_right_obj, &children_default_obj,
        &features_obj, &thresholds_obj, &values_obj, &max_depth, &tree_limit, &base_offset, &model_output,
        &X_obj, &X_missing_obj, &y_obj, &out_pred_obj
    )) return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyArrayObject *children_left_array = (PyArrayObject*)PyArray_FROM_OTF(children_left_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *children_right_array = (PyArrayObject*)PyArray_FROM_OTF(children_right_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *children_default_array = (PyArrayObject*)PyArray_FROM_OTF(children_default_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *features_array = (PyArrayObject*)PyArray_FROM_OTF(features_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *thresholds_array = (PyArrayObject*)PyArray_FROM_OTF(thresholds_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *values_array = (PyArrayObject*)PyArray_FROM_OTF(values_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *X_array = (PyArrayObject*)PyArray_FROM_OTF(X_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *X_missing_array = (PyArrayObject*)PyArray_FROM_OTF(X_missing_obj, NPY_BOOL, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *y_array = NULL;
    if (y_obj != Py_None) y_array = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *out_pred_array = (PyArrayObject*)PyArray_FROM_OTF(out_pred_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);

    /* If that didn't work, throw an exception. Note that R and y are optional. */
    if (children_left_array == NULL || children_right_array == NULL ||
        children_default_array == NULL || features_array == NULL || thresholds_array == NULL ||
        values_array == NULL || X_array == NULL ||
        X_missing_array == NULL || out_pred_array == NULL) {
        Py_XDECREF(children_left_array);
        Py_XDECREF(children_right_array);
        Py_XDECREF(children_default_array);
        Py_XDECREF(features_array);
        Py_XDECREF(thresholds_array);
        Py_XDECREF(values_array);
        Py_XDECREF(X_array);
        Py_XDECREF(X_missing_array);
        if (y_array != NULL) Py_XDECREF(y_array);
        //PyArray_ResolveWritebackIfCopy(out_pred_array);
        Py_XDECREF(out_pred_array);
        return NULL;
    }

    const unsigned num_X = PyArray_DIM(X_array, 0);
    const unsigned M = PyArray_DIM(X_array, 1);
    const unsigned max_nodes = PyArray_DIM(values_array, 1);
    const unsigned num_outputs = PyArray_DIM(values_array, 2);

    // Get pointers to the data as C-types
    int *children_left = (int*)PyArray_DATA(children_left_array);
    int *children_right = (int*)PyArray_DATA(children_right_array);
    int *children_default = (int*)PyArray_DATA(children_default_array);
    int *features = (int*)PyArray_DATA(features_array);
    tfloat *thresholds = (tfloat*)PyArray_DATA(thresholds_array);
    tfloat *values = (tfloat*)PyArray_DATA(values_array);
    tfloat *X = (tfloat*)PyArray_DATA(X_array);
    bool *X_missing = (bool*)PyArray_DATA(X_missing_array);
    tfloat *y = NULL;
    if (y_array != NULL) y = (tfloat*)PyArray_DATA(y_array);
    tfloat *out_pred = (tfloat*)PyArray_DATA(out_pred_array);

    // these are just wrapper objects for all the pointers and numbers associated with
    // the ensemble tree model and the datset we are explaing
    TreeEnsemble trees = TreeEnsemble(
        children_left, children_right, children_default, features, thresholds, values,
        NULL, max_depth, tree_limit, base_offset,
        max_nodes, num_outputs
    );
    ExplanationDataset data = ExplanationDataset(X, X_missing, y, NULL, NULL, num_X, M, 0);

    dense_tree_predict(out_pred, trees, data, model_output);

    // clean up the created python objects 
    Py_XDECREF(children_left_array);
    Py_XDECREF(children_right_array);
    Py_XDECREF(children_default_array);
    Py_XDECREF(features_array);
    Py_XDECREF(thresholds_array);
    Py_XDECREF(values_array);
    Py_XDECREF(X_array);
    Py_XDECREF(X_missing_array);
    if (y_array != NULL) Py_XDECREF(y_array);
    //PyArray_ResolveWritebackIfCopy(out_pred_array);
    Py_XDECREF(out_pred_array);

    /* Build the output tuple */
    PyObject *ret = Py_BuildValue("d", (double)values[0]);
    return ret;
}


static PyObject *_cext_dense_tree_update_weights(PyObject *self, PyObject *args)
{
    PyObject *children_left_obj;
    PyObject *children_right_obj;
    PyObject *children_default_obj;
    PyObject *features_obj;
    PyObject *thresholds_obj;
    PyObject *values_obj;
    int tree_limit;
    PyObject *node_sample_weight_obj;
    PyObject *X_obj;
    PyObject *X_missing_obj;
  
    /* Parse the input tuple */
    if (!PyArg_ParseTuple(
        args, "OOOOOOiOOO", &children_left_obj, &children_right_obj, &children_default_obj,
        &features_obj, &thresholds_obj, &values_obj, &tree_limit, &node_sample_weight_obj, &X_obj, &X_missing_obj
    )) return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyArrayObject *children_left_array = (PyArrayObject*)PyArray_FROM_OTF(children_left_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *children_right_array = (PyArrayObject*)PyArray_FROM_OTF(children_right_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *children_default_array = (PyArrayObject*)PyArray_FROM_OTF(children_default_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *features_array = (PyArrayObject*)PyArray_FROM_OTF(features_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *thresholds_array = (PyArrayObject*)PyArray_FROM_OTF(thresholds_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *values_array = (PyArrayObject*)PyArray_FROM_OTF(values_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *node_sample_weight_array = (PyArrayObject*)PyArray_FROM_OTF(node_sample_weight_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
    PyArrayObject *X_array = (PyArrayObject*)PyArray_FROM_OTF(X_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *X_missing_array = (PyArrayObject*)PyArray_FROM_OTF(X_missing_obj, NPY_BOOL, NPY_ARRAY_IN_ARRAY);

    /* If that didn't work, throw an exception. */
    if (children_left_array == NULL || children_right_array == NULL ||
        children_default_array == NULL || features_array == NULL || thresholds_array == NULL ||
        values_array == NULL || node_sample_weight_array == NULL || X_array == NULL ||
        X_missing_array == NULL) {
        Py_XDECREF(children_left_array);
        Py_XDECREF(children_right_array);
        Py_XDECREF(children_default_array);
        Py_XDECREF(features_array);
        Py_XDECREF(thresholds_array);
        Py_XDECREF(values_array);
        //PyArray_ResolveWritebackIfCopy(node_sample_weight_array);
        Py_XDECREF(node_sample_weight_array);
        Py_XDECREF(X_array);
        Py_XDECREF(X_missing_array);
        std::cerr << "Found a NULL input array in _cext_dense_tree_update_weights!\n";
        return NULL;
    }

    const unsigned num_X = PyArray_DIM(X_array, 0);
    const unsigned M = PyArray_DIM(X_array, 1);
    const unsigned max_nodes = PyArray_DIM(values_array, 1);

    // Get pointers to the data as C-types
    int *children_left = (int*)PyArray_DATA(children_left_array);
    int *children_right = (int*)PyArray_DATA(children_right_array);
    int *children_default = (int*)PyArray_DATA(children_default_array);
    int *features = (int*)PyArray_DATA(features_array);
    tfloat *thresholds = (tfloat*)PyArray_DATA(thresholds_array);
    tfloat *values = (tfloat*)PyArray_DATA(values_array);
    tfloat *node_sample_weight = (tfloat*)PyArray_DATA(node_sample_weight_array);
    tfloat *X = (tfloat*)PyArray_DATA(X_array);
    bool *X_missing = (bool*)PyArray_DATA(X_missing_array);

    // these are just wrapper objects for all the pointers and numbers associated with
    // the ensemble tree model and the datset we are explaing
    TreeEnsemble trees = TreeEnsemble(
        children_left, children_right, children_default, features, thresholds, values,
        node_sample_weight, 0, tree_limit, 0, max_nodes, 0
    );
    ExplanationDataset data = ExplanationDataset(X, X_missing, NULL, NULL, NULL, num_X, M, 0);

    dense_tree_update_weights(trees, data);

    // clean up the created python objects 
    Py_XDECREF(children_left_array);
    Py_XDECREF(children_right_array);
    Py_XDECREF(children_default_array);
    Py_XDECREF(features_array);
    Py_XDECREF(thresholds_array);
    Py_XDECREF(values_array);
    //PyArray_ResolveWritebackIfCopy(node_sample_weight_array);
    Py_XDECREF(node_sample_weight_array);
    Py_XDECREF(X_array);
    Py_XDECREF(X_missing_array);

    /* Build the output tuple */
    PyObject *ret = Py_BuildValue("d", 1);
    return ret;
}


static PyObject *_cext_dense_tree_saabas(PyObject *self, PyObject *args)
{
    PyObject *children_left_obj;
    PyObject *children_right_obj;
    PyObject *children_default_obj;
    PyObject *features_obj;
    PyObject *thresholds_obj;
    PyObject *values_obj;
    int max_depth;
    int tree_limit;
    double base_offset;
    int model_output;
    PyObject *X_obj;
    PyObject *X_missing_obj;
    PyObject *y_obj;
    PyObject *out_pred_obj;
    
  
    /* Parse the input tuple */
    if (!PyArg_ParseTuple(
        args, "OOOOOOiidiOOOO", &children_left_obj, &children_right_obj, &children_default_obj,
        &features_obj, &thresholds_obj, &values_obj, &max_depth, &tree_limit, &base_offset, &model_output,
        &X_obj, &X_missing_obj, &y_obj, &out_pred_obj
    )) return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyArrayObject *children_left_array = (PyArrayObject*)PyArray_FROM_OTF(children_left_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *children_right_array = (PyArrayObject*)PyArray_FROM_OTF(children_right_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *children_default_array = (PyArrayObject*)PyArray_FROM_OTF(children_default_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *features_array = (PyArrayObject*)PyArray_FROM_OTF(features_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *thresholds_array = (PyArrayObject*)PyArray_FROM_OTF(thresholds_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *values_array = (PyArrayObject*)PyArray_FROM_OTF(values_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *X_array = (PyArrayObject*)PyArray_FROM_OTF(X_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *X_missing_array = (PyArrayObject*)PyArray_FROM_OTF(X_missing_obj, NPY_BOOL, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *y_array = NULL;
    if (y_obj != Py_None) y_array = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *out_pred_array = (PyArrayObject*)PyArray_FROM_OTF(out_pred_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    /* If that didn't work, throw an exception. Note that R and y are optional. */
    if (children_left_array == NULL || children_right_array == NULL ||
        children_default_array == NULL || features_array == NULL || thresholds_array == NULL ||
        values_array == NULL || X_array == NULL ||
        X_missing_array == NULL || out_pred_array == NULL) {
        Py_XDECREF(children_left_array);
        Py_XDECREF(children_right_array);
        Py_XDECREF(children_default_array);
        Py_XDECREF(features_array);
        Py_XDECREF(thresholds_array);
        Py_XDECREF(values_array);
        Py_XDECREF(X_array);
        Py_XDECREF(X_missing_array);
        if (y_array != NULL) Py_XDECREF(y_array);
        //PyArray_ResolveWritebackIfCopy(out_pred_array);
        Py_XDECREF(out_pred_array);
        return NULL;
    }

    const unsigned num_X = PyArray_DIM(X_array, 0);
    const unsigned M = PyArray_DIM(X_array, 1);
    const unsigned max_nodes = PyArray_DIM(values_array, 1);
    const unsigned num_outputs = PyArray_DIM(values_array, 2);

    // Get pointers to the data as C-types
    int *children_left = (int*)PyArray_DATA(children_left_array);
    int *children_right = (int*)PyArray_DATA(children_right_array);
    int *children_default = (int*)PyArray_DATA(children_default_array);
    int *features = (int*)PyArray_DATA(features_array);
    tfloat *thresholds = (tfloat*)PyArray_DATA(thresholds_array);
    tfloat *values = (tfloat*)PyArray_DATA(values_array);
    tfloat *X = (tfloat*)PyArray_DATA(X_array);
    bool *X_missing = (bool*)PyArray_DATA(X_missing_array);
    tfloat *y = NULL;
    if (y_array != NULL) y = (tfloat*)PyArray_DATA(y_array);
    tfloat *out_pred = (tfloat*)PyArray_DATA(out_pred_array);

    // these are just wrapper objects for all the pointers and numbers associated with
    // the ensemble tree model and the datset we are explaing
    TreeEnsemble trees = TreeEnsemble(
        children_left, children_right, children_default, features, thresholds, values,
        NULL, max_depth, tree_limit, base_offset,
        max_nodes, num_outputs
    );
    ExplanationDataset data = ExplanationDataset(X, X_missing, y, NULL, NULL, num_X, M, 0);

    dense_tree_saabas(out_pred, trees, data);

    // clean up the created python objects 
    Py_XDECREF(children_left_array);
    Py_XDECREF(children_right_array);
    Py_XDECREF(children_default_array);
    Py_XDECREF(features_array);
    Py_XDECREF(thresholds_array);
    Py_XDECREF(values_array);
    Py_XDECREF(X_array);
    Py_XDECREF(X_missing_array);
    if (y_array != NULL) Py_XDECREF(y_array);
    //PyArray_ResolveWritebackIfCopy(out_pred_array);
    Py_XDECREF(out_pred_array);

    /* Build the output tuple */
    PyObject *ret = Py_BuildValue("d", (double)values[0]);
    return ret;
}
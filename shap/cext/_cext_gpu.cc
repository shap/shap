#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include "tree_shap.h"
#include <iostream>

static PyObject *_cext_dense_tree_shap(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"dense_tree_shap", _cext_dense_tree_shap, METH_VARARGS, "C implementation of Tree SHAP for dense."},
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_cext_gpu",
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
PyMODINIT_FUNC PyInit__cext_gpu(void)
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

void dense_tree_shap_gpu(const TreeEnsemble& trees, const ExplanationDataset &data, tfloat *out_contribs,
                     const int feature_dependence, unsigned model_transform, bool interactions);

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
    PyObject *base_offset_obj;
    bool interactions;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(
        args, "OOOOOOOiOOOOOiOOiib", &children_left_obj, &children_right_obj, &children_default_obj,
        &features_obj, &thresholds_obj, &values_obj, &node_sample_weights_obj,
        &max_depth, &X_obj, &X_missing_obj, &y_obj, &R_obj, &R_missing_obj, &tree_limit, &base_offset_obj,
        &out_contribs_obj, &feature_dependence, &model_output, &interactions
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
    PyArrayObject *base_offset_array = (PyArrayObject*)PyArray_FROM_OTF(base_offset_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);

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
        Py_XDECREF(base_offset_array);
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
    tfloat *base_offset = (tfloat*)PyArray_DATA(base_offset_array);

    // these are just a wrapper objects for all the pointers and numbers associated with
    // the ensemble tree model and the dataset we are explaining
    TreeEnsemble trees = TreeEnsemble(
        children_left, children_right, children_default, features, thresholds, values,
        node_sample_weights, max_depth, tree_limit, base_offset,
        max_nodes, num_outputs
    );
    ExplanationDataset data = ExplanationDataset(X, X_missing, y, R, R_missing, num_X, M, num_R);

    dense_tree_shap_gpu(trees, data, out_contribs, feature_dependence, model_output, interactions);


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
    Py_XDECREF(base_offset_array);

    /* Build the output tuple */
    PyObject *ret = Py_BuildValue("d", ret_value);
    return ret;
}

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include "tree_shap.h"
#include <iostream>

typedef double tfloat;

// Have a treeshapdependent and a treeshapindependent
static PyObject *_cext_dense_tree_shap(PyObject *self, PyObject *args);
static PyObject *_cext_compute_expectations(PyObject *self, PyObject *args);
//static PyObject *_cext_tree_shap_indep(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"dense_tree_shap", _cext_dense_tree_shap, METH_VARARGS, "C implementation of Tree SHAP for dense."},
    {"compute_expectations", _cext_compute_expectations, METH_VARARGS, "Compute expectations of internal nodes."},
    //{"tree_shap_indep", _cext_tree_shap_indep, METH_VARARGS, "C implementation of Independent Tree SHAP."},
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
    PyArrayObject *values_array = (PyArrayObject*)PyArray_FROM_OTF(values_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    /* If that didn't work, throw an exception. */
    if (children_left_array == NULL || children_right_array == NULL ||
        values_array == NULL || node_sample_weight_array == NULL) {
        Py_XDECREF(children_left_array);
        Py_XDECREF(children_right_array);
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
    bool less_than_or_equal;
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
  
  /* Parse the input tuple */
    if (!PyArg_ParseTuple(
        args, "OOOOOOObiOOOOOidOii", &children_left_obj, &children_right_obj, &children_default_obj,
        &features_obj, &thresholds_obj, &values_obj, &node_sample_weights_obj, &less_than_or_equal,
        &max_depth, &X_obj, &X_missing_obj, &y_obj, &R_obj, &R_missing_obj, &tree_limit, &base_offset,
        &out_contribs_obj, &feature_dependence, &model_output
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
    PyArrayObject *y_array = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *R_array = (PyArrayObject*)PyArray_FROM_OTF(R_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *R_missing_array = (PyArrayObject*)PyArray_FROM_OTF(R_missing_obj, NPY_BOOL, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *out_contribs_array = (PyArrayObject*)PyArray_FROM_OTF(out_contribs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

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
        Py_XDECREF(y_array);
        Py_XDECREF(R_array);
        Py_XDECREF(R_missing_array);
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

    //const int max_depth = compute_expectations(children_left, children_right, node_sample_weight, values, 0, 0);

    // unsigned i;
    // std::cout << "int children_left[] = {";
    // for (i = 0; i < max_nodes*tree_limit-1; ++i) {
    //   std::cout << children_left[i] << ", ";
    // } 
    // std::cout << children_left[i] << "};\n";

    // std::cout << "int children_right[] = {";
    // for (i = 0; i < max_nodes*tree_limit-1; ++i) {
    //   std::cout << children_right[i] << ", ";
    // } 
    // std::cout << children_right[i] << "};\n";

    // std::cout << "int children_default[] = {";
    // for (i = 0; i < max_nodes*tree_limit-1; ++i) {
    //   std::cout << children_default[i] << ", ";
    // } 
    // std::cout << children_default[i] << "};\n";

    // std::cout << "int features[] = {";
    // for (i = 0; i < max_nodes*tree_limit-1; ++i) {
    //   std::cout << features[i] << ", ";
    // } 
    // std::cout << features[i] << "};\n";

    // std::cout << "tfloat thresholds[] = {";
    // for (i = 0; i < max_nodes*tree_limit-1; ++i) {
    //   std::cout << thresholds[i] << ", ";
    // } 
    // std::cout << thresholds[i] << "};\n";

    // std::cout << "tfloat values[] = {";
    // for (i = 0; i < max_nodes*num_outputs*tree_limit-1; ++i) {
    //   std::cout << values[i] << ", ";
    // } 
    // std::cout << values[i] << "};\n";

    // std::cout << "tfloat node_sample_weights[] = {";
    // for (i = 0; i < max_nodes*tree_limit-1; ++i) {
    //   std::cout << node_sample_weights[i] << ", ";
    // } 
    // std::cout << node_sample_weights[i] << "};\n";

    // std::cout << "tfloat X[] = {";
    // for (i = 0; i < num_X * M - 1; ++i) {
    //   std::cout << X[i] << ", ";
    // } 
    // std::cout << X[i] << "};\n";

    // std::cout << "bool X_missing[] = {";
    // for (i = 0; i < num_X * M - 1; ++i) {
    //   std::cout << X_missing[i] << ", ";
    // } 
    // std::cout << X_missing[i] << "};\n";

    // std::cout << "tfloat R[] = {";
    // for (i = 0; i < num_R * M - 1; ++i) {
    //   std::cout << R[i] << ", ";
    // } 
    // std::cout << R[i] << "};\n";

    // std::cout << "bool R_missing[] = {";
    // for (i = 0; i < num_R * M - 1; ++i) {
    //   std::cout << R_missing[i] << ", ";
    // } 
    // std::cout << R_missing[i] << "};\n";

    // std::cout << "tfloat *y = NULL;\n";

    // std::cout << "tfloat out_contribs[] = {";
    // for (i = 0; i < M; ++i) {
    //   std::cout << out_contribs[i] << ", ";
    // } 
    // std::cout << out_contribs[i] << "};\n";

    // std::cout << "unsigned num_X = " << num_X << ";\n";
    // std::cout << "unsigned M = " << M << ";\n";
    // std::cout << "unsigned max_nodes = " << max_nodes << ";\n";
    // std::cout << "unsigned num_outputs = " << num_outputs << ";\n";
    // std::cout << "unsigned num_R = " << num_R << ";\n";
    // std::cout << "unsigned max_depth = " << max_depth << ";\n";
    // std::cout << "bool less_than_or_equal = " << less_than_or_equal << ";\n";
    // std::cout << "int tree_limit = " << tree_limit << ";\n";
    // std::cout << "int feature_dependence = " << feature_dependence << ";\n";
    // std::cout << "int model_output = " << model_output << ";\n";
    // std::cout << "double base_offset = " << base_offset << ";\n";

    // these are just a wrapper objects for all the pointers and numbers associated with
    // the ensemble tree model and the datset we are explaing
    TreeEnsemble trees = TreeEnsemble(
        children_left, children_right, children_default, features, thresholds, values,
        node_sample_weights, less_than_or_equal, max_depth, tree_limit, base_offset,
        max_nodes, num_outputs
    );
    ExplanationDataset data = ExplanationDataset(X, X_missing, y, R, R_missing, num_X, M, num_R);

    std::cout << "In cext75!" << std::endl;
    dense_tree_shap(trees, data, out_contribs, feature_dependence, model_output);
    std::cout << "past dense_tree_shap" << std::endl;

    // tree_shap(
    //   M, num_outputs, max_depth, children_left, children_right, children_default, features,
    //   thresholds, values, node_sample_weight, x, x_missing, out_contribs,
    //   condition, condition_feature, less_than_or_equal
    // );

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
    Py_XDECREF(y_array);
    Py_XDECREF(R_array);
    Py_XDECREF(R_missing_array);
    Py_XDECREF(out_contribs_array);

    /* Build the output tuple */
    PyObject *ret = Py_BuildValue("d", (double)values[0]);
    return ret;
}

// static PyObject *_cext_tree_shap_indep(PyObject *self, PyObject *args)
// {
//   int max_depth;
//   PyObject *children_left_obj;
//   PyObject *children_right_obj;
//   PyObject *children_default_obj;
//   PyObject *features_obj;
//   PyObject *thresholds_obj;
//   PyObject *values_obj;
//   PyObject *x_obj;
//   PyObject *x_missing_obj;
//   PyObject *r_obj;
//   PyObject *r_missing_obj;
//   PyObject *out_contribs_obj;

//   /* Parse the input tuple */
//   if (!PyArg_ParseTuple(
//     args, "iOOOOOOOOOOO", &max_depth, &children_left_obj, &children_right_obj,
//     &children_default_obj, &features_obj, &thresholds_obj, &values_obj, 
//     &x_obj, &x_missing_obj, &r_obj, &r_missing_obj, &out_contribs_obj
//   )) return NULL;

//   /* Interpret the input objects as numpy arrays. */
//   PyArrayObject *children_left_array = (PyArrayObject*)PyArray_FROM_OTF(children_left_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
//   PyArrayObject *children_right_array = (PyArrayObject*)PyArray_FROM_OTF(children_right_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
//   PyArrayObject *children_default_array = (PyArrayObject*)PyArray_FROM_OTF(children_default_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
//   PyArrayObject *features_array = (PyArrayObject*)PyArray_FROM_OTF(features_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
//   PyArrayObject *thresholds_array = (PyArrayObject*)PyArray_FROM_OTF(thresholds_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
//   PyArrayObject *values_array = (PyArrayObject*)PyArray_FROM_OTF(values_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
//   PyArrayObject *x_array = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
//   PyArrayObject *x_missing_array = (PyArrayObject*)PyArray_FROM_OTF(x_missing_obj, NPY_BOOL, NPY_ARRAY_IN_ARRAY);
//   PyArrayObject *r_array = (PyArrayObject*)PyArray_FROM_OTF(r_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
//   PyArrayObject *r_missing_array = (PyArrayObject*)PyArray_FROM_OTF(r_missing_obj, NPY_BOOL, NPY_ARRAY_IN_ARRAY);
//   PyArrayObject *out_contribs_array = (PyArrayObject*)PyArray_FROM_OTF(out_contribs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

//   /* If that didn't work, throw an exception. */
//   if (children_left_array == NULL || children_right_array == NULL || 
//       children_default_array == NULL || features_array == NULL || 
//       thresholds_array == NULL || values_array == NULL || x_array == NULL || 
//       x_missing_array == NULL || r_array == NULL || r_missing_array == NULL || 
//       out_contribs_array == NULL) {
//     Py_XDECREF(children_left_array);
//     Py_XDECREF(children_right_array);
//     Py_XDECREF(children_default_array);
//     Py_XDECREF(features_array);
//     Py_XDECREF(thresholds_array);
//     Py_XDECREF(values_array);
//     Py_XDECREF(x_array);
//     Py_XDECREF(x_missing_array);
//     Py_XDECREF(r_array);
//     Py_XDECREF(r_missing_array);
//     Py_XDECREF(out_contribs_array);
//     return NULL;
//   }

//   // number of features
//   const unsigned num_feats = PyArray_DIM(x_array, 0);

//   // number of nodes
//   const unsigned num_nodes = PyArray_DIM(features_array, 0);

//   // Get pointers to the data as C-types
//   int *children_left = (int*)PyArray_DATA(children_left_array);
//   int *children_right = (int*)PyArray_DATA(children_right_array);
//   int *children_default = (int*)PyArray_DATA(children_default_array);
//   int *features = (int*)PyArray_DATA(features_array);
//   tfloat *thresholds = (tfloat*)PyArray_DATA(thresholds_array);
//   tfloat *values = (tfloat*)PyArray_DATA(values_array);
//   tfloat *x = (tfloat*)PyArray_DATA(x_array);
//   bool *x_missing = (bool*)PyArray_DATA(x_missing_array);
//   tfloat *r = (tfloat*)PyArray_DATA(r_array);
//   bool *r_missing = (bool*)PyArray_DATA(r_missing_array);
//   tfloat *out_contribs = (tfloat*)PyArray_DATA(out_contribs_array);
    
//   // Preallocating things    
//   Node *mytree = new Node[num_nodes];
//   for (unsigned i = 0; i < num_nodes; ++i) {
//     mytree[i].cl = children_left[i];
//     mytree[i].cr = children_right[i];
//     mytree[i].cd = children_default[i];
//     if (i == 0) {
//       mytree[i].pnode = 0;
//     }
//     if (children_left[i] >= 0) {
//       mytree[children_left[i]].pnode = i;
//       mytree[children_left[i]].pfeat = features[i];
//     }
//     if (children_right[i] >= 0) {
//       mytree[children_right[i]].pnode = i;
//       mytree[children_right[i]].pfeat = features[i];
//     }

//     mytree[i].thres = thresholds[i];
//     mytree[i].value = values[i];
//     mytree[i].feat = features[i];
//   }
    
//   float *pos_lst = new float[num_nodes];
//   float *neg_lst = new float[num_nodes];
//   int *node_stack = new int[(unsigned) max_depth];
//   signed short *feat_hist = new signed short[num_feats];
//   float *memoized_weights = new float[30*30];
//   for (int n = 0; n < 30; ++n) {
//     for (int m = 0; m < 30; ++m) {
//       memoized_weights[n+30*m] = calc_weight(n, m);
//     }
//   }

//   tree_shap_indep(
//       max_depth, num_feats, num_nodes, x, x_missing, r, r_missing, 
//       out_contribs, pos_lst, neg_lst, feat_hist, memoized_weights, 
//       node_stack, mytree
//   );
//   delete[] mytree;
//   delete[] pos_lst;
//   delete[] neg_lst;
//   delete[] node_stack;
//   delete[] feat_hist;
//   delete[] memoized_weights;

//   // clean up the created python objects
//   Py_XDECREF(children_left_array);
//   Py_XDECREF(children_right_array);
//   Py_XDECREF(children_default_array);
//   Py_XDECREF(features_array);
//   Py_XDECREF(thresholds_array);
//   Py_XDECREF(values_array);
//   Py_XDECREF(x_array);
//   Py_XDECREF(x_missing_array);
//   Py_XDECREF(r_array);
//   Py_XDECREF(r_missing_array);
//   Py_XDECREF(out_contribs_array);

//   /* Build the output tuple */
//   PyObject *ret = Py_BuildValue("d", (double)values[0]);
//   return ret;
// }

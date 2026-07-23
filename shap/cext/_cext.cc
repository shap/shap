#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>

#include <optional>
#include <string>

#include "tree_shap.h"

namespace nb = nanobind;
using namespace nb::literals;

template <typename T, typename... Args>
using Array = nb::ndarray<T, nb::c_contig, Args...>;

using IntArray = Array<const int>;
using DoubleArray = Array<const double>;
using BoolArray = Array<const bool>;
using MutableDoubleArray = Array<double>;

using IntArray1D = Array<const int, nb::ndim<1>>;
using DoubleArray1D = Array<const double, nb::ndim<1>>;
using MutableDoubleArray1D = Array<double, nb::ndim<1>>;
using DoubleArray2D = Array<const double, nb::ndim<2>>;
using MutableDoubleArray2D = Array<double, nb::ndim<2>>;
using DoubleArray3D = Array<const double, nb::ndim<3>>;

int compute_expectations_wrapper(
    const IntArray1D &children_left,
    const IntArray1D &children_right,
    const DoubleArray1D &node_sample_weight,
    MutableDoubleArray2D &values
) {
    TreeEnsemble tree;
    tree.num_outputs = values.shape(1);
    tree.children_left = const_cast<int *>(children_left.data());
    tree.children_right = const_cast<int *>(children_right.data());
    tree.values = values.data();
    tree.node_sample_weights = const_cast<double *>(node_sample_weight.data());

    return compute_expectations(tree);
}

double dense_tree_shap_wrapper(
    const IntArray &children_left,
    const IntArray &children_right,
    const IntArray &children_default,
    const IntArray &features,
    const DoubleArray &thresholds,
    const IntArray &threshold_types,
    const DoubleArray3D &values,
    const DoubleArray &node_sample_weights,
    int max_depth,
    const DoubleArray2D &X,
    const BoolArray &X_missing,
    const std::optional<DoubleArray> &y,
    const std::optional<DoubleArray2D> &R,
    const std::optional<BoolArray> &R_missing,
    int tree_limit,
    const DoubleArray1D &base_offset,
    MutableDoubleArray &out_contribs,
    int feature_dependence,
    int model_output,
    bool interactions
) {
    const unsigned num_X = X.shape(0);
    const unsigned M = X.shape(1);
    const unsigned max_nodes = values.shape(1);
    const unsigned num_outputs = values.shape(2);
    const unsigned num_R = R ? R->shape(0) : 0;

    TreeEnsemble trees(
        const_cast<int *>(children_left.data()),
        const_cast<int *>(children_right.data()),
        const_cast<int *>(children_default.data()),
        const_cast<int *>(features.data()),
        const_cast<double *>(thresholds.data()),
        const_cast<int *>(threshold_types.data()),
        const_cast<double *>(values.data()),
        const_cast<double *>(node_sample_weights.data()),
        max_depth,
        tree_limit,
        const_cast<double *>(base_offset.data()),
        max_nodes,
        num_outputs
    );
    ExplanationDataset data(
        const_cast<double *>(X.data()),
        const_cast<bool *>(X_missing.data()),
        y ? const_cast<double *>(y->data()) : nullptr,
        R ? const_cast<double *>(R->data()) : nullptr,
        R_missing ? const_cast<bool *>(R_missing->data()) : nullptr,
        num_X,
        M,
        num_R
    );

    dense_tree_shap(
        trees, data, out_contribs.data(), feature_dependence, model_output, interactions
    );
    return values.data()[0];
}

double dense_tree_predict_wrapper(
    const IntArray &children_left,
    const IntArray &children_right,
    const IntArray &children_default,
    const IntArray &features,
    const DoubleArray &thresholds,
    const IntArray &threshold_types,
    const DoubleArray3D &values,
    int max_depth,
    int tree_limit,
    const DoubleArray1D &base_offset,
    int model_output,
    const DoubleArray2D &X,
    const BoolArray &X_missing,
    const std::optional<DoubleArray> &y,
    MutableDoubleArray &out_pred
) {
    const unsigned num_X = X.shape(0);
    const unsigned M = X.shape(1);
    const unsigned max_nodes = values.shape(1);
    const unsigned num_outputs = values.shape(2);

    if (base_offset.shape(0) != num_outputs) {
        throw nb::value_error(
            (
                "The passed base_offset array does not have the same number of outputs "
                "as the values array: " +
                std::to_string(base_offset.shape(0)) + " vs. " +
                std::to_string(num_outputs)
            ).c_str()
        );
    }

    TreeEnsemble trees(
        const_cast<int *>(children_left.data()),
        const_cast<int *>(children_right.data()),
        const_cast<int *>(children_default.data()),
        const_cast<int *>(features.data()),
        const_cast<double *>(thresholds.data()),
        const_cast<int *>(threshold_types.data()),
        const_cast<double *>(values.data()),
        nullptr,
        max_depth,
        tree_limit,
        const_cast<double *>(base_offset.data()),
        max_nodes,
        num_outputs
    );
    ExplanationDataset data(
        const_cast<double *>(X.data()),
        const_cast<bool *>(X_missing.data()),
        y ? const_cast<double *>(y->data()) : nullptr,
        nullptr,
        nullptr,
        num_X,
        M,
        0
    );

    dense_tree_predict(out_pred.data(), trees, data, model_output);
    return values.data()[0];
}

double dense_tree_update_weights_wrapper(
    const IntArray1D &children_left,
    const IntArray1D &children_right,
    const IntArray1D &children_default,
    const IntArray1D &features,
    const DoubleArray1D &thresholds,
    const IntArray1D &threshold_types,
    const DoubleArray2D &values,
    int tree_limit,
    MutableDoubleArray1D &node_sample_weight,
    const DoubleArray2D &X,
    const BoolArray &X_missing
) {
    const unsigned num_X = X.shape(0);
    const unsigned M = X.shape(1);
    const unsigned max_nodes = values.shape(0);

    TreeEnsemble trees(
        const_cast<int *>(children_left.data()),
        const_cast<int *>(children_right.data()),
        const_cast<int *>(children_default.data()),
        const_cast<int *>(features.data()),
        const_cast<double *>(thresholds.data()),
        const_cast<int *>(threshold_types.data()),
        const_cast<double *>(values.data()),
        node_sample_weight.data(),
        0,
        tree_limit,
        nullptr,
        max_nodes,
        0
    );
    ExplanationDataset data(
        const_cast<double *>(X.data()),
        const_cast<bool *>(X_missing.data()),
        nullptr,
        nullptr,
        nullptr,
        num_X,
        M,
        0
    );

    dense_tree_update_weights(trees, data);
    return 1.0;
}

double dense_tree_saabas_wrapper(
    const IntArray &children_left,
    const IntArray &children_right,
    const IntArray &children_default,
    const IntArray &features,
    const DoubleArray &thresholds,
    const IntArray &threshold_types,
    const DoubleArray3D &values,
    int max_depth,
    int tree_limit,
    const DoubleArray1D &base_offset,
    int model_output,
    const DoubleArray2D &X,
    const BoolArray &X_missing,
    const std::optional<DoubleArray> &y,
    MutableDoubleArray &out_pred
) {
    const unsigned num_X = X.shape(0);
    const unsigned M = X.shape(1);
    const unsigned max_nodes = values.shape(1);
    const unsigned num_outputs = values.shape(2);

    TreeEnsemble trees(
        const_cast<int *>(children_left.data()),
        const_cast<int *>(children_right.data()),
        const_cast<int *>(children_default.data()),
        const_cast<int *>(features.data()),
        const_cast<double *>(thresholds.data()),
        const_cast<int *>(threshold_types.data()),
        const_cast<double *>(values.data()),
        nullptr,
        max_depth,
        tree_limit,
        const_cast<double *>(base_offset.data()),
        max_nodes,
        num_outputs
    );
    ExplanationDataset data(
        const_cast<double *>(X.data()),
        const_cast<bool *>(X_missing.data()),
        y ? const_cast<double *>(y->data()) : nullptr,
        nullptr,
        nullptr,
        num_X,
        M,
        0
    );

    dense_tree_saabas(out_pred.data(), trees, data);
    return values.data()[0];
}

NB_MODULE(_cext, m) {
    m.doc() = "Fast Tree SHAP implementation.";

    m.def(
        "dense_tree_shap",
        &dense_tree_shap_wrapper,
        "children_left"_a,
        "children_right"_a,
        "children_default"_a,
        "features"_a,
        "thresholds"_a,
        "threshold_types"_a,
        "values"_a,
        "node_sample_weights"_a,
        "max_depth"_a,
        "X"_a,
        "X_missing"_a,
        "y"_a,
        "R"_a,
        "R_missing"_a,
        "tree_limit"_a,
        "base_offset"_a,
        "out_contribs"_a,
        "feature_dependence"_a,
        "model_output"_a,
        "interactions"_a,
        "C implementation of Tree SHAP for dense trees."
    );
    m.def(
        "dense_tree_predict",
        &dense_tree_predict_wrapper,
        "children_left"_a,
        "children_right"_a,
        "children_default"_a,
        "features"_a,
        "thresholds"_a,
        "threshold_types"_a,
        "values"_a,
        "max_depth"_a,
        "tree_limit"_a,
        "base_offset"_a,
        "model_output"_a,
        "X"_a,
        "X_missing"_a,
        "y"_a,
        "out_pred"_a,
        "C implementation of tree predictions."
    );
    m.def(
        "dense_tree_update_weights",
        &dense_tree_update_weights_wrapper,
        "children_left"_a,
        "children_right"_a,
        "children_default"_a,
        "features"_a,
        "thresholds"_a,
        "threshold_types"_a,
        "values"_a,
        "tree_limit"_a,
        "node_sample_weight"_a,
        "X"_a,
        "X_missing"_a,
        "C implementation of tree node weight computations."
    );
    m.def(
        "dense_tree_saabas",
        &dense_tree_saabas_wrapper,
        "children_left"_a,
        "children_right"_a,
        "children_default"_a,
        "features"_a,
        "thresholds"_a,
        "threshold_types"_a,
        "values"_a,
        "max_depth"_a,
        "tree_limit"_a,
        "base_offset"_a,
        "model_output"_a,
        "X"_a,
        "X_missing"_a,
        "y"_a,
        "out_pred"_a,
        "C implementation of Saabas (rough fast approximation to Tree SHAP)."
    );
    m.def(
        "compute_expectations",
        &compute_expectations_wrapper,
        "children_left"_a,
        "children_right"_a,
        "node_sample_weight"_a,
        "values"_a,
        "Compute expectations of internal nodes."
    );
}

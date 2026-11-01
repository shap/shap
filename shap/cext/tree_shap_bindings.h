#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <initializer_list>
#include <limits>
#include <string>
#include "tree_shap.h"

namespace shap_bindings {
namespace nb = nanobind;
using namespace nb::literals;

// Inputs may be converted to contiguous storage. Outputs must never be converted:
// writes to a temporary would be lost when the binding returns.
template <typename T>
using Array = nb::ndarray<nb::numpy, T, nb::c_contig, nb::device::cpu>;
using IntArray = Array<const int>;
using FloatArray = Array<const double>;
using BoolArray = Array<const bool>;
using OutputArray = Array<double>;

template <typename T>
void require_shape(const Array<T> &array, std::initializer_list<size_t> shape,
                   const char *name) {
    if (array.ndim() != shape.size() ||
        !std::equal(shape.begin(), shape.end(), array.shape_ptr())) {
        throw nb::value_error((std::string(name) + " has an incompatible shape").c_str());
    }
}

// The algorithm's legacy structs use mutable pointers even for read-only inputs.
// Keep that adaptation here instead of requiring writable Python inputs.
template <typename T>
T *input_data(const Array<const T> &array) {
    return array.is_valid() ? const_cast<T *>(array.data()) : nullptr;
}

inline unsigned dimension(size_t value) {
    if (value > std::numeric_limits<unsigned>::max()) {
        throw nb::value_error("Array dimension exceeds the tree implementation's limit");
    }
    return static_cast<unsigned>(value);
}

inline TreeEnsemble make_tree(
    IntArray children_left, IntArray children_right, IntArray children_default,
    IntArray features, FloatArray thresholds, IntArray threshold_types,
    FloatArray values, FloatArray weights, int max_depth, int tree_limit,
    FloatArray base_offset, bool single_tree = false) {
    const size_t rank = single_tree ? 2 : 3;
    if (values.ndim() != rank || values.size() == 0) {
        throw nb::value_error("values must be a nonempty tree array with the expected rank");
    }
    const size_t num_trees = single_tree ? 1 : values.shape(0);
    const size_t max_nodes = values.shape(rank - 2);
    const size_t num_outputs = values.shape(rank - 1);
    if (max_depth < 0 || tree_limit < 0 || static_cast<size_t>(tree_limit) > num_trees) {
        throw nb::value_error("Invalid max_depth or tree_limit");
    }
    auto check_nodes = [&](const auto &array, const char *name) {
        if (single_tree) require_shape(array, {max_nodes}, name);
        else require_shape(array, {num_trees, max_nodes}, name);
    };
    check_nodes(children_left, "children_left");
    check_nodes(children_right, "children_right");
    check_nodes(children_default, "children_default");
    check_nodes(features, "features");
    check_nodes(thresholds, "thresholds");
    check_nodes(threshold_types, "threshold_types");
    if (weights.is_valid()) check_nodes(weights, "node_sample_weights");
    if (base_offset.is_valid()) require_shape(base_offset, {num_outputs}, "base_offset");
    return TreeEnsemble(
        input_data(children_left), input_data(children_right), input_data(children_default),
        input_data(features), input_data(thresholds), input_data(threshold_types),
        input_data(values), input_data(weights), max_depth, tree_limit, input_data(base_offset),
        dimension(max_nodes), dimension(num_outputs));
}

inline ExplanationDataset make_data(FloatArray X, BoolArray X_missing,
                                    FloatArray y = {}, FloatArray R = {}, BoolArray R_missing = {}) {
    if (X.ndim() != 2) throw nb::value_error("X must have two dimensions");
    const size_t rows = X.shape(0), columns = X.shape(1);
    require_shape(X_missing, {rows, columns}, "X_missing");
    if (y.is_valid()) require_shape(y, {rows}, "y");
    if (R.is_valid() != R_missing.is_valid()) {
        throw nb::value_error("R and R_missing must both be arrays or both be None");
    }
    size_t background_rows = 0;
    if (R.is_valid()) {
        if (R.ndim() != 2 || R.shape(1) != columns) {
            throw nb::value_error("R must have the same number of columns as X");
        }
        background_rows = R.shape(0);
        require_shape(R_missing, {background_rows, columns}, "R_missing");
    }
    return ExplanationDataset(input_data(X), input_data(X_missing), input_data(y),
                              input_data(R), input_data(R_missing), dimension(rows),
                              dimension(columns), dimension(background_rows));
}

inline void check_contributions(OutputArray out, const TreeEnsemble &trees,
                                const ExplanationDataset &data, bool interactions) {
    const size_t features = static_cast<size_t>(data.M) + 1;
    if (interactions) {
        require_shape(out, {data.num_X, features, features, trees.num_outputs}, "out_contribs");
    } else {
        require_shape(out, {data.num_X, features, trees.num_outputs}, "out_contribs");
    }
}

// Both backends expose the same Python signature and buffer contract.
template <typename Function>
void bind_dense_tree_shap(nb::module_ &m, Function implementation) {
    m.def("dense_tree_shap", [implementation](
        IntArray children_left, IntArray children_right, IntArray children_default,
        IntArray features, FloatArray thresholds, IntArray threshold_types,
        FloatArray values, FloatArray node_sample_weights, int max_depth,
        FloatArray X, BoolArray X_missing, FloatArray y, FloatArray R, BoolArray R_missing,
        int tree_limit, FloatArray base_offset, OutputArray out_contribs,
        int feature_dependence, int model_output, bool interactions) {
        auto trees = make_tree(children_left, children_right, children_default, features,
                               thresholds, threshold_types, values, node_sample_weights,
                               max_depth, tree_limit, base_offset);
        auto data = make_data(X, X_missing, y, R, R_missing);
        check_contributions(out_contribs, trees, data, interactions);
        if (feature_dependence == FEATURE_DEPENDENCE::independent && data.num_R == 0) {
            throw nb::value_error("Independent Tree SHAP requires nonempty background data");
        }
        if ((model_output == MODEL_TRANSFORM::logistic_nlogloss ||
             model_output == MODEL_TRANSFORM::squared_loss) && !y.is_valid()) {
            throw nb::value_error("Explaining loss requires labels y");
        }
        // Retain the GIL: the CPU implementation calls Python's stderr API.
        implementation(trees, data, out_contribs.data(), feature_dependence, model_output, interactions);
        return values.data()[0];
    }, "children_left"_a, "children_right"_a, "children_default"_a, "features"_a,
       "thresholds"_a, "threshold_types"_a, "values"_a, "node_sample_weights"_a,
       "max_depth"_a, "X"_a, "X_missing"_a, "y"_a.none(), "R"_a.none(),
       "R_missing"_a.none(), "tree_limit"_a, "base_offset"_a,
       "out_contribs"_a.noconvert(), "feature_dependence"_a, "model_output"_a,
       "interactions"_a, "Compute Tree SHAP values for a dense ensemble.");
}
}  // namespace shap_bindings

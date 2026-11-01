#include "tree_shap_bindings.h"

using namespace shap_bindings;

NB_MODULE(_cext, m) {
    m.doc() = "Bindings for the dense Tree SHAP implementation.";
    bind_dense_tree_shap(m, &dense_tree_shap);

    m.def("compute_expectations", [](IntArray children_left, IntArray children_right,
                                      FloatArray node_sample_weight, OutputArray values) {
        if (values.ndim() != 2 || values.size() == 0) {
            throw nb::value_error("values must be a nonempty two-dimensional array");
        }
        const size_t nodes = values.shape(0);
        require_shape(children_left, {nodes}, "children_left");
        require_shape(children_right, {nodes}, "children_right");
        require_shape(node_sample_weight, {nodes}, "node_sample_weight");
        TreeEnsemble tree;
        tree.num_outputs = dimension(values.shape(1));
        tree.children_left = input_data(children_left);
        tree.children_right = input_data(children_right);
        tree.values = values.data();
        tree.node_sample_weights = input_data(node_sample_weight);
        return compute_expectations(tree);
    }, "children_left"_a, "children_right"_a, "node_sample_weight"_a,
       "values"_a.noconvert(), "Compute expectations of internal nodes in place.");

    m.def("dense_tree_predict", [](
        IntArray children_left, IntArray children_right, IntArray children_default,
        IntArray features, FloatArray thresholds, IntArray threshold_types,
        FloatArray values, int max_depth, int tree_limit, FloatArray base_offset,
        int model_output, FloatArray X, BoolArray X_missing, FloatArray y, OutputArray out_pred) {
        auto trees = make_tree(children_left, children_right, children_default, features,
                               thresholds, threshold_types, values, {}, max_depth, tree_limit, base_offset);
        auto data = make_data(X, X_missing, y);
        require_shape(out_pred, {data.num_X, trees.num_outputs}, "out_pred");
        dense_tree_predict(out_pred.data(), trees, data, model_output);
        return values.data()[0];
    }, "children_left"_a, "children_right"_a, "children_default"_a, "features"_a,
       "thresholds"_a, "threshold_types"_a, "values"_a, "max_depth"_a, "tree_limit"_a,
       "base_offset"_a, "model_output"_a, "X"_a, "X_missing"_a, "y"_a.none(),
       "out_pred"_a.noconvert(), "Predict from a dense tree ensemble.");

    m.def("dense_tree_update_weights", [](
        IntArray children_left, IntArray children_right, IntArray children_default,
        IntArray features, FloatArray thresholds, IntArray threshold_types,
        FloatArray values, int tree_limit, OutputArray node_sample_weight,
        FloatArray X, BoolArray X_missing) {
        auto trees = make_tree(children_left, children_right, children_default, features,
                               thresholds, threshold_types, values, FloatArray(node_sample_weight),
                               0, tree_limit, {}, values.ndim() == 2);
        auto data = make_data(X, X_missing);
        dense_tree_update_weights(trees, data);
        return 1.0;
    }, "children_left"_a, "children_right"_a, "children_default"_a, "features"_a,
       "thresholds"_a, "threshold_types"_a, "values"_a, "tree_limit"_a,
       "node_sample_weight"_a.noconvert(), "X"_a, "X_missing"_a,
       "Update node sample weights in place.");

    m.def("dense_tree_saabas", [](
        IntArray children_left, IntArray children_right, IntArray children_default,
        IntArray features, FloatArray thresholds, IntArray threshold_types,
        FloatArray values, int max_depth, int tree_limit, FloatArray base_offset,
        int model_output, FloatArray X, BoolArray X_missing, FloatArray y, OutputArray out_pred) {
        auto trees = make_tree(children_left, children_right, children_default, features,
                               thresholds, threshold_types, values, {}, max_depth, tree_limit, base_offset);
        auto data = make_data(X, X_missing, y);
        check_contributions(out_pred, trees, data, false);
        dense_tree_saabas(out_pred.data(), trees, data);
        return values.data()[0];
    }, "children_left"_a, "children_right"_a, "children_default"_a, "features"_a,
       "thresholds"_a, "threshold_types"_a, "values"_a, "max_depth"_a, "tree_limit"_a,
       "base_offset"_a, "model_output"_a, "X"_a, "X_missing"_a, "y"_a.none(),
       "out_pred"_a.noconvert(), "Compute the Saabas approximation for a dense ensemble.");
}

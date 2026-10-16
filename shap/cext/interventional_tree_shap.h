#pragma once

#include <algorithm>
#include <vector>

// TreeEnsemble, ExplanationDataset and tree_predict are defined by tree_shap.h.
/**
 * Interventional Tree SHAP values with an empirical background distribution.
 *
 * For a foreground row x and a background row r, every coalition takes each
 * feature from either x or r. Splits where x and r agree are followed directly,
 * and a feature that already has a source on the current path follows that
 * source, so only distinct disagreeing features become players. At a reachable
 * leaf with a foreground and b background assignments, the coalition polynomial
 * is t^a (1-t)^b. Integrating its derivative gives each foreground feature
 * 1 / (a * C(a+b, a)) and each background feature -1 / (b * C(a+b, b)) of the
 * leaf value. Other features are dummies for that leaf. The leaf reached with
 * every assignment from the background (a = 0) contributes to the baseline.
 * Unlike quadrature, these coefficients are exact and need no cover statistics.
 */
namespace shap_interventional {

// Per-player coefficients indexed by (a, b), already divided by a or b.
struct Coefficients {
    unsigned size;
    std::vector<double> positive, negative;

    explicit Coefficients(unsigned max_players)
        : size(max_players + 1), positive(size * size, 0), negative(size * size, 0) {
        for (unsigned a = 0; a < size; ++a) {
            for (unsigned b = 0; a + b < size; ++b) {
                // 1 / C(a+b, a) by a short multiplicative recurrence, avoiding overflow.
                unsigned small = std::min(a, b), large = std::max(a, b);
                double inverse = 1;
                for (unsigned i = 1; i <= small; ++i) inverse *= static_cast<double>(i) / (large + i);
                if (a > 0) positive[a * size + b] = inverse / a;
                if (b > 0) negative[a * size + b] = -inverse / b;
            }
        }
    }
};

inline bool GoesLeft(const TreeEnsemble &tree, unsigned node, const double *x, const bool *missing) {
    unsigned feature = tree.features[node];
    if (missing[feature]) return tree.children_default[node] == tree.children_left[node];
    if (tree.threshold_types[node] == 1) return category_in_threshold(tree.thresholds[node], x[feature]);
    return x[feature] <= tree.thresholds[node];
}

template <bool SingleOutput> struct Runner {
    const Coefficients &coefficients;
    unsigned features;
    const TreeEnsemble *tree = nullptr;
    const double *x = nullptr, *r = nullptr;
    const bool *x_missing = nullptr, *r_missing = nullptr;
    double *phi = nullptr;
    // +1 when a feature is taken from x on the current path, -1 from r, 0 if unassigned.
    // Every assignment is undone while unwinding, so this is shared across trees and pairs.
    std::vector<signed char> source;
    std::vector<unsigned> path;

    Runner(const Coefficients &coefficients, unsigned features, unsigned max_depth)
        : coefficients(coefficients), features(features), source(features, 0) {
        path.reserve(max_depth);
    }

    unsigned NumOutputs() const { return SingleOutput ? 1 : tree->num_outputs; }

    void Leaf(unsigned node, unsigned a, unsigned b) {
        const double *values = tree->values + static_cast<size_t>(node) * NumOutputs();
        size_t index = static_cast<size_t>(a) * coefficients.size + b;
        double positive = coefficients.positive[index], negative = coefficients.negative[index];
        for (unsigned o = 0; o < NumOutputs(); ++o) {
            double value = values[o];
            if (a == 0) phi[static_cast<size_t>(features) * NumOutputs() + o] += value;
            for (auto feature : path) {
                phi[static_cast<size_t>(feature) * NumOutputs() + o] +=
                    value * (source[feature] > 0 ? positive : negative);
            }
        }
    }

    // Finish the foreground subtree before visiting its background sibling.
    void Visit(unsigned node, unsigned a, unsigned b) {
        while (!tree->is_leaf(node)) {
            unsigned feature = tree->features[node];
            bool x_left = GoesLeft(*tree, node, x, x_missing);
            bool r_left = GoesLeft(*tree, node, r, r_missing);
            if (x_left != r_left && source[feature] == 0) {
                unsigned left = tree->children_left[node], right = tree->children_right[node];
                path.push_back(feature);
                source[feature] = 1;
                Visit(x_left ? left : right, a + 1, b);
                source[feature] = -1;
                Visit(r_left ? left : right, a, b + 1);
                source[feature] = 0;
                path.pop_back();
                return;
            }
            // Agreement, or a repeated split following its feature's assigned source.
            bool left = x_left == r_left || source[feature] > 0 ? x_left : r_left;
            node = left ? tree->children_left[node] : tree->children_right[node];
        }
        Leaf(node, a, b);
    }
};

// Size the coefficient table from the trees rather than trusting max_depth metadata.
inline unsigned Depth(const TreeEnsemble &tree, unsigned node) {
    if (tree.is_leaf(node)) return 0;
    return 1 + std::max(Depth(tree, tree.children_left[node]), Depth(tree, tree.children_right[node]));
}

template <bool SingleOutput>
inline void ExplainRows(const TreeEnsemble &trees, const std::vector<TreeEnsemble> &views,
                        const ExplanationDataset &data, double *phi, transform_f transform) {
    const unsigned outputs = trees.num_outputs;
    const size_t stride = static_cast<size_t>(data.M + 1) * outputs;
    unsigned depth = 0;
    for (const auto &view : views) depth = std::max(depth, Depth(view, 0));
    Coefficients coefficients(std::min(depth, data.M));
    Runner<SingleOutput> runner(coefficients, data.M, depth);
    // Non-linear transforms rescale each reference's attributions by its own margins.
    std::vector<double> pair(transform != NULL ? stride : 0);
    std::vector<double> margin_x(outputs), margin_r(outputs);
    auto margins = [&](const double *row, const bool *missing, std::vector<double> &margin) {
        for (unsigned o = 0; o < outputs; ++o) margin[o] = trees.base_offset[o];
        for (unsigned t = 0; t < trees.tree_limit; ++t) {
            const double *values = tree_predict(t, trees, row, missing);
            for (unsigned o = 0; o < outputs; ++o) margin[o] += values[o];
        }
    };

    for (unsigned i = 0; i < data.num_X; ++i) {
        double *out = phi + i * stride;
        runner.x = data.X + static_cast<size_t>(i) * data.M;
        runner.x_missing = data.X_missing + static_cast<size_t>(i) * data.M;
        const double y = data.y == NULL ? 0 : data.y[i];
        if (transform != NULL) margins(runner.x, runner.x_missing, margin_x);

        for (unsigned j = 0; j < data.num_R; ++j) {
            runner.r = data.R + static_cast<size_t>(j) * data.M;
            runner.r_missing = data.R_missing + static_cast<size_t>(j) * data.M;
            if (transform != NULL) std::fill(pair.begin(), pair.end(), 0);
            runner.phi = transform != NULL ? pair.data() : out;
            for (const auto &view : views) {
                runner.tree = &view;
                runner.Visit(0, 0, 0);
            }
            if (transform == NULL) continue;

            margins(runner.r, runner.r_missing, margin_r);
            for (unsigned o = 0; o < outputs; ++o) {
                double rescale = 1;
                if (margin_x[o] != margin_r[o]) {
                    rescale = ((*transform)(margin_x[o], y) - (*transform)(margin_r[o], y)) /
                              (margin_x[o] - margin_r[o]);
                }
                for (unsigned k = 0; k < data.M; ++k) out[k * outputs + o] += pair[k * outputs + o] * rescale;
                out[data.M * outputs + o] +=
                    (*transform)(trees.base_offset[o] + pair[data.M * outputs + o], 0);
            }
        }

        for (size_t k = 0; k < stride; ++k) out[k] /= data.num_R;
        if (transform == NULL) {
            for (unsigned o = 0; o < outputs; ++o) out[data.M * outputs + o] += trees.base_offset[o];
        }
    }
}

// Matches dense_independent's output layout, transforms and reference averaging,
// computing every output in a single traversal per row/reference pair.
inline void Explain(const TreeEnsemble &trees, const ExplanationDataset &data, double *phi,
                    transform_f transform) {
    std::vector<TreeEnsemble> views(trees.tree_limit);
    for (unsigned t = 0; t < trees.tree_limit; ++t) trees.get_tree(views[t], t);
    if (trees.num_outputs == 1) ExplainRows<true>(trees, views, data, phi, transform);
    else ExplainRows<false>(trees, views, data, phi, transform);
}
}  // namespace shap_interventional

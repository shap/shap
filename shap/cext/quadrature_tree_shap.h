/**
 * Copyright 2017-2026, XGBoost Contributors
 * Copyright 2026, SHAP contributors
 *
 * Adapted from XGBoost's QuadratureTreeShapRunner and quadrature rule at
 * 51b132efe13cb39e6cb424394980ac4610cbe8e9.
 *
 * This header is included after SHAP's TreeEnsemble/ExplanationDataset definitions.
 */
#ifndef SHAP_QUADRATURE_TREE_SHAP_H
#define SHAP_QUADRATURE_TREE_SHAP_H

#include <vector>

namespace shap_quadrature {

struct Rule {
    std::vector<double> nodes, weights;
    explicit Rule(unsigned n) : nodes(n), weights(n) {
        constexpr double pi = 3.141592653589793238462643383279502884;
        for (unsigned i = 0; i < n; ++i) {
            double x = std::cos(pi * (i + 0.75) / (n + 0.5));
            double derivative = 0;
            for (unsigned iter = 0; iter < 64; ++iter) {
                double p0 = 1, p1 = x;
                for (unsigned k = 2; k <= n; ++k) {
                    double p = ((2.0 * k - 1) * x * p1 - (k - 1.0) * p0) / k;
                    p0 = p1;
                    p1 = p;
                }
                derivative = n * (x * p1 - p0) / (x * x - 1);
                double dx = p1 / derivative;
                x -= dx;
                if (std::abs(dx) < 1e-15) break;
            }
            // Re-evaluate at the converged root, as in the upstream rule.
            double p0 = 1, p1 = x;
            for (unsigned k = 2; k <= n; ++k) {
                double p = ((2.0 * k - 1) * x * p1 - (k - 1.0) * p0) / k;
                p0 = p1;
                p1 = p;
            }
            derivative = n * (x * p1 - p0) / (x * x - 1);
            nodes[n - 1 - i] = 0.5 * (x + 1);
            weights[n - 1 - i] = 1 / ((1 - x * x) * derivative * derivative);
        }
    }
};

// With m distinct path features the additive integrand has degree m - 1.
// A ceil(m / 2)-point Gauss-Legendre rule integrates it exactly. The pairwise
// integrand has lower degree; its diagonal still needs the additive rule.
inline unsigned CountPathFeatures(const TreeEnsemble &tree, unsigned node,
                                  std::vector<unsigned> &counts, unsigned unique) {
    if (tree.is_leaf(node)) return unique;
    auto feature = tree.features[node];
    if (counts[feature]++ == 0) ++unique;
    auto left = CountPathFeatures(tree, tree.children_left[node], counts, unique);
    auto right = CountPathFeatures(tree, tree.children_right[node], counts, unique);
    --counts[feature];
    return std::max(left, right);
}

inline unsigned RuleSize(const TreeEnsemble &tree, unsigned num_features) {
    std::vector<unsigned> counts(num_features, 0);
    return std::max(1u, (CountPathFeatures(tree, 0, counts, 0) + 1) / 2);
}

inline double BranchWeight(double cover, double parent_cover) {
    // Preserve even very small positive probabilities in SHAP's double-precision
    // model. Upstream's float32 floor would change valid rare-branch attributions.
    if (parent_cover <= 0) return 0.5;
    return cover > 0 ? cover / parent_cover : 1e-12;
}

inline std::vector<double> CoverRatios(const TreeEnsemble &tree) {
    std::vector<double> ratios(tree.max_nodes, 1);
    for (unsigned node = 0; node < tree.max_nodes; ++node) {
        if (tree.is_leaf(node)) continue;
        auto left = tree.children_left[node], right = tree.children_right[node];
        ratios[left] = BranchWeight(tree.node_sample_weights[left], tree.node_sample_weights[node]);
        ratios[right] = BranchWeight(tree.node_sample_weights[right], tree.node_sample_weights[node]);
    }
    return ratios;
}

// Compute the baseline from leaves and covers, as upstream does. Imported trees
// can have stale internal values (in particular when a float32 loader was copied
// by the C extension), so tree.values[0] is not a reliable root expectation.
inline void RootMean(const TreeEnsemble &tree, const std::vector<double> &ratios,
                     unsigned node, double weight, std::vector<double> &mean) {
    if (tree.is_leaf(node)) {
        for (unsigned o = 0; o < tree.num_outputs; ++o) {
            mean[o] += weight * tree.values[static_cast<size_t>(node) * tree.num_outputs + o];
        }
        return;
    }
    auto left = tree.children_left[node], right = tree.children_right[node];
    RootMean(tree, ratios, left, weight * ratios[left], mean);
    RootMean(tree, ratios, right, weight * ratios[right], mean);
}

// Upstream's return-edge recurrence, with SHAP's split semantics and output layout.
// Fold w_prod into the basis (as the upstream GPU does), avoiding large basis
// products multiplied by vanishing leaf probabilities.
template <bool Interactions> struct Runner {
    const TreeEnsemble &tree;
    const Rule &rule;
    const std::vector<double> &ratios;
    const double *x;
    const bool *missing;
    unsigned features, n, outputs;
    size_t return_size, workspace_size;
    double *phi;
    std::vector<double> probability;
    std::vector<unsigned> path;
    std::vector<double> scratch;

    Runner(const TreeEnsemble &tree, const Rule &rule, const std::vector<double> &ratios,
           const double *x, const bool *missing, unsigned features, double *phi)
        : tree(tree), rule(rule), ratios(ratios), x(x), missing(missing), features(features),
          n(static_cast<unsigned>(rule.nodes.size())), outputs(tree.num_outputs),
          return_size(static_cast<size_t>(n) * outputs), workspace_size(n + 2 * return_size),
          phi(phi), probability(features, -1),
          scratch(n + return_size + static_cast<size_t>(tree.max_depth) * workspace_size) {
        path.reserve(tree.max_depth);
        std::fill(scratch.begin(), scratch.begin() + n, 1);
    }

    double *Value(unsigned feature) {
        return phi + (Interactions ? static_cast<size_t>(feature) * (features + 2) : feature)
                     * outputs;
    }

    // Each level owns a shared next basis and output-by-point right/edge buffers.
    // Siblings reuse the same level after returning; descendants use later levels.
    // Root basis and results are also reused across rows.
    double *Workspace(unsigned depth) {
        return scratch.data() + n + return_size + static_cast<size_t>(depth) * workspace_size;
    }

    void Visit(unsigned split, unsigned child, bool satisfies, unsigned depth,
               const double *basis, double *result) {
        unsigned feature = tree.features[split];
        double old = probability[feature];
        bool first = old < 0;
        double previous = first ? 1 : old;
        double weight = ratios[child];
        double enter = satisfies ? previous / weight : 0;
        double *next = Workspace(depth) + return_size;
        for (unsigned i = 0; i < n; ++i) {
            next[i] = basis[i] * weight * (1 + (enter - 1) * rule.nodes[i])
                      / (1 + (previous - 1) * rule.nodes[i]);
        }
        probability[feature] = enter;
        if (Interactions && first) path.push_back(feature);
        RunNode(child, depth + 1, next, result);
        double *edge = Workspace(depth) + return_size + n;
        // The child's basis is no longer needed. Reuse it for coefficients shared
        // by every output; keep points contiguous for each output's reduction.
        for (unsigned i = 0; i < n; ++i) {
            double t = rule.nodes[i];
            next[i] = (enter - 1) / (1 + (enter - 1) * t) -
                      (previous - 1) / (1 + (previous - 1) * t);
        }
        double *value = Value(feature);
        for (unsigned o = 0; o < outputs; ++o) {
            size_t offset = static_cast<size_t>(o) * n;
            double delta = 0;
            for (unsigned i = 0; i < n; ++i) {
                edge[offset + i] = rule.weights[i] * result[offset + i] * next[i];
                delta += edge[offset + i];
            }
            value[o] += delta;
        }
        if (Interactions) {
            for (auto partner : path) {
                if (partner == feature) continue;
                double alpha = probability[partner] - 1;
                for (unsigned i = 0; i < n; ++i) next[i] = 1 + alpha * rule.nodes[i];
                double *pair = phi + (static_cast<size_t>(feature) * (features + 1) + partner)
                                     * outputs;
                for (unsigned o = 0; o < outputs; ++o) {
                    size_t offset = static_cast<size_t>(o) * n;
                    double delta = 0;
                    for (unsigned i = 0; i < n; ++i) {
                        delta += edge[offset + i] * alpha / next[i];
                    }
                    pair[o] += delta;
                }
            }
            if (first) path.pop_back();
        }
        probability[feature] = old;
    }

    void RunNode(unsigned node, unsigned depth, const double *basis, double *result) {
        if (tree.is_leaf(node)) {
            const double *values = tree.values + static_cast<size_t>(node) * outputs;
            for (unsigned o = 0; o < outputs; ++o) {
                for (unsigned i = 0; i < n; ++i) {
                    result[static_cast<size_t>(o) * n + i] = basis[i] * values[o];
                }
            }
            return;
        }
        unsigned feature = tree.features[node];
        bool left = missing[feature] ? tree.children_default[node] == tree.children_left[node] :
            (tree.threshold_types[node] == 1 ? category_in_threshold(tree.thresholds[node], x[feature]) :
                                             x[feature] <= tree.thresholds[node]);
        double *right = Workspace(depth);
        Visit(node, tree.children_left[node], left, depth, basis, result);
        Visit(node, tree.children_right[node], !left, depth, basis, right);
        for (size_t i = 0; i < return_size; ++i) result[i] += right[i];
    }

    void Run(const std::vector<double> &root_mean) {
        RunNode(0, 0, scratch.data(), scratch.data() + n);
        double *baseline = Value(features);
        for (unsigned o = 0; o < outputs; ++o) baseline[o] += root_mean[o];
    }
};

inline void FinalizeInteractions(double *phi, unsigned features, unsigned outputs) {
    size_t columns = features + 1;
    for (unsigned output = 0; output < outputs; ++output) {
        for (unsigned i = 0; i < features; ++i) {
            for (unsigned j = i + 1; j < features; ++j) {
                double &a = phi[(i * columns + j) * outputs + output];
                double &b = phi[(j * columns + i) * outputs + output];
                a = b = 0.5 * (a + b);
            }
        }
        for (unsigned i = 0; i < features; ++i) {
            double &diag = phi[(i * columns + i) * outputs + output];
            for (unsigned j = 0; j < features; ++j) {
                if (i != j) diag -= phi[(i * columns + j) * outputs + output];
            }
        }
    }
}

template <bool Interactions>
inline void ExplainTree(const TreeEnsemble &tree, const Rule &rule,
                        const std::vector<double> &ratios, const std::vector<double> &mean,
                        const ExplanationDataset &data, double *phi, size_t stride) {
    Runner<Interactions> runner(tree, rule, ratios, nullptr, nullptr, data.M, nullptr);
    // Rewind restores the probability/path state. Reuse all workspace across rows,
    // and traverse once per row with shared path arithmetic for every output.
    for (unsigned row = 0; row < data.num_X; ++row) {
        runner.x = data.X + static_cast<size_t>(row) * data.M;
        runner.missing = data.X_missing + static_cast<size_t>(row) * data.M;
        runner.phi = phi + row * stride;
        runner.Run(mean);
    }
}

template <bool Interactions>
inline void Explain(const TreeEnsemble &trees, const ExplanationDataset &data, double *phi) {
    const size_t columns = data.M + 1;
    const size_t stride = columns * (Interactions ? columns : 1) * trees.num_outputs;
    // Prepare cover ratios and quadrature once per tree, not once per explained row.
    for (unsigned t = 0; t < trees.tree_limit; ++t) {
        TreeEnsemble tree;
        trees.get_tree(tree, t);
        Rule rule(RuleSize(tree, data.M));
        auto ratios = CoverRatios(tree);
        std::vector<double> mean(tree.num_outputs, 0);
        RootMean(tree, ratios, 0, 1, mean);
        ExplainTree<Interactions>(tree, rule, ratios, mean, data, phi, stride);
    }
    for (unsigned row = 0; row < data.num_X; ++row) {
        double *out = phi + row * stride;
        for (unsigned output = 0; output < trees.num_outputs; ++output) {
            out[stride - trees.num_outputs + output] += trees.base_offset[output];
        }
        if (Interactions) FinalizeInteractions(out, data.M, trees.num_outputs);
    }
}
}  // namespace shap_quadrature
#endif  // SHAP_QUADRATURE_TREE_SHAP_H

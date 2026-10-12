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
    unsigned features, output, n;
    double *phi;
    std::vector<double> probability;
    std::vector<unsigned> path;

    Runner(const TreeEnsemble &tree, const Rule &rule, const std::vector<double> &ratios,
           const double *x, const bool *missing, unsigned features, unsigned output, double *phi)
        : tree(tree), rule(rule), ratios(ratios), x(x), missing(missing), features(features),
          output(output), n(static_cast<unsigned>(rule.nodes.size())), phi(phi),
          probability(features, -1) { path.reserve(tree.max_depth); }

    double &Value(unsigned feature) {
        return phi[(Interactions ? static_cast<size_t>(feature) * (features + 2) : feature)
                   * tree.num_outputs + output];
    }

    void Visit(unsigned split, unsigned child, bool satisfies,
               const std::vector<double> &basis, std::vector<double> &result) {
        unsigned feature = tree.features[split];
        double old = probability[feature];
        bool first = old < 0;
        double previous = first ? 1 : old;
        double weight = ratios[child];
        double enter = satisfies ? previous / weight : 0;
        std::vector<double> next(n);
        for (unsigned i = 0; i < n; ++i) {
            next[i] = basis[i] * weight * (1 + (enter - 1) * rule.nodes[i])
                      / (1 + (previous - 1) * rule.nodes[i]);
        }
        probability[feature] = enter;
        if (Interactions && first) path.push_back(feature);
        RunNode(child, next, result);
        std::vector<double> edge(n);
        double delta = 0;
        for (unsigned i = 0; i < n; ++i) {
            double t = rule.nodes[i];
            edge[i] = rule.weights[i] * result[i] *
                ((enter - 1) / (1 + (enter - 1) * t) -
                 (previous - 1) / (1 + (previous - 1) * t));
            delta += edge[i];
        }
        Value(feature) += delta;
        if (Interactions) {
            for (auto partner : path) {
                if (partner == feature) continue;
                double alpha = probability[partner] - 1, pair = 0;
                for (unsigned i = 0; i < n; ++i) {
                    pair += edge[i] * alpha / (1 + alpha * rule.nodes[i]);
                }
                phi[(static_cast<size_t>(feature) * (features + 1) + partner)
                    * tree.num_outputs + output] += pair;
            }
            if (first) path.pop_back();
        }
        probability[feature] = old;
    }

    void RunNode(unsigned node, const std::vector<double> &basis, std::vector<double> &result) {
        if (tree.is_leaf(node)) {
            double value = tree.values[static_cast<size_t>(node) * tree.num_outputs + output];
            for (unsigned i = 0; i < n; ++i) result[i] = basis[i] * value;
            return;
        }
        unsigned feature = tree.features[node];
        bool left = missing[feature] ? tree.children_default[node] == tree.children_left[node] :
            (tree.threshold_types[node] == 1 ? category_in_threshold(tree.thresholds[node], x[feature]) :
                                             x[feature] <= tree.thresholds[node]);
        std::vector<double> right(n);
        Visit(node, tree.children_left[node], left, basis, result);
        Visit(node, tree.children_right[node], !left, basis, right);
        for (unsigned i = 0; i < n; ++i) result[i] += right[i];
    }

    void Run(double root_mean) {
        std::vector<double> basis(n), result(n);
        for (unsigned i = 0; i < n; ++i) basis[i] = 1;
        RunNode(0, basis, result);
        Value(features) += root_mean;
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
    for (unsigned output = 0; output < tree.num_outputs; ++output) {
        Runner<Interactions> runner(tree, rule, ratios, nullptr, nullptr, data.M, output, nullptr);
        // Rewind restores the probability/path state. Reuse the feature workspace
        // across rows instead of allocating O(num_features) storage for every row.
        for (unsigned row = 0; row < data.num_X; ++row) {
            runner.x = data.X + static_cast<size_t>(row) * data.M;
            runner.missing = data.X_missing + static_cast<size_t>(row) * data.M;
            runner.phi = phi + row * stride;
            runner.Run(mean[output]);
        }
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

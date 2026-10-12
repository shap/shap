/**
 * Fast recursive computation of SHAP values in trees.
 * See https://arxiv.org/abs/1802.03888 for details.
 *
 * Scott Lundberg, 2018 (independent algorithm courtesy of Hugh Chen 2018)
 */

#include <algorithm>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <cmath>
#include <ctime>
#include <vector>
using namespace std;

typedef double tfloat;
typedef tfloat (* transform_f)(const tfloat margin, const tfloat y);

namespace FEATURE_DEPENDENCE {
    const unsigned independent = 0;
    const unsigned tree_path_dependent = 1;
}

struct TreeEnsemble {
    int *children_left;
    int *children_right;
    int *children_default;
    int *features;
    tfloat *thresholds;
    int* threshold_types;
    tfloat *values;
    tfloat *node_sample_weights;
    unsigned max_depth;
    unsigned tree_limit;
    tfloat *base_offset;
    unsigned max_nodes;
    unsigned num_outputs;

    TreeEnsemble() {}
    TreeEnsemble(int *children_left, int *children_right, int *children_default, int *features,
                 tfloat *thresholds, int *threshold_types, tfloat *values, tfloat *node_sample_weights,
                 unsigned max_depth, unsigned tree_limit, tfloat *base_offset,
                 unsigned max_nodes, unsigned num_outputs) :
        children_left(children_left), children_right(children_right),
        children_default(children_default), features(features), thresholds(thresholds),
        threshold_types(threshold_types), values(values), node_sample_weights(node_sample_weights),
        max_depth(max_depth), tree_limit(tree_limit),
        base_offset(base_offset), max_nodes(max_nodes), num_outputs(num_outputs) {}

    void get_tree(TreeEnsemble &tree, const unsigned i) const {
        const unsigned d = i * max_nodes;

        tree.children_left = children_left + d;
        tree.children_right = children_right + d;
        tree.children_default = children_default + d;
        tree.features = features + d;
        tree.thresholds = thresholds + d;
        tree.threshold_types = threshold_types + d;
        tree.values = values + d * num_outputs;
        tree.node_sample_weights = node_sample_weights + d;
        tree.max_depth = max_depth;
        tree.tree_limit = 1;
        tree.base_offset = base_offset;
        tree.max_nodes = max_nodes;
        tree.num_outputs = num_outputs;
    }

    bool is_leaf(unsigned pos)const {
        return children_left[pos] < 0;
    }
};

struct ExplanationDataset {
    tfloat *X;
    bool *X_missing;
    tfloat *y;
    tfloat *R;
    bool *R_missing;
    unsigned num_X;
    unsigned M;
    unsigned num_R;

    ExplanationDataset() {}
    ExplanationDataset(tfloat *X, bool *X_missing, tfloat *y, tfloat *R, bool *R_missing, unsigned num_X,
                       unsigned M, unsigned num_R) :
        X(X), X_missing(X_missing), y(y), R(R), R_missing(R_missing), num_X(num_X), M(M), num_R(num_R) {}

    void get_x_instance(ExplanationDataset &instance, const unsigned i) const {
        instance.M = M;
        instance.X = X + i * M;
        instance.X_missing = X_missing + i * M;
        instance.num_X = 1;
    }
};


inline tfloat logistic_transform(const tfloat margin, const tfloat y) {
    return 1 / (1 + exp(-margin));
}

inline tfloat logistic_nlogloss_transform(const tfloat margin, const tfloat y) {
    return log(1 + exp(margin)) - y * margin; // y is in {0, 1}
}

inline tfloat squared_loss_transform(const tfloat margin, const tfloat y) {
    return (margin - y) * (margin - y);
}

namespace MODEL_TRANSFORM {
    const unsigned identity = 0;
    const unsigned logistic = 1;
    const unsigned logistic_nlogloss = 2;
    const unsigned squared_loss = 3;
}

inline transform_f get_transform(unsigned model_transform) {
    transform_f transform = NULL;
    switch (model_transform) {
        case MODEL_TRANSFORM::logistic:
            transform = logistic_transform;
            break;

        case MODEL_TRANSFORM::logistic_nlogloss:
            transform = logistic_nlogloss_transform;
            break;

        case MODEL_TRANSFORM::squared_loss:
            transform = squared_loss_transform;
            break;
    }

    return transform;
}

inline bool category_in_threshold(float threshold, float category) {
    int category_flag = (1 << (int(category) - 1));
    return (int(threshold) & category_flag) != 0;
}

inline tfloat *tree_predict(unsigned i, const TreeEnsemble &trees, const tfloat *x, const bool *x_missing) {
    const unsigned offset = i * trees.max_nodes;
    unsigned node = 0;
    while (true) {
        const unsigned pos = offset + node;
        const unsigned feature = trees.features[pos];

        // we hit a leaf so return a pointer to the values
        if (trees.is_leaf(pos)) {
            return trees.values + pos * trees.num_outputs;
        }

        // otherwise we are at an internal node and need to recurse
        if (x_missing[feature]) {
            node = trees.children_default[pos];
        } else if (trees.threshold_types[pos] == 0 && x[feature] <= trees.thresholds[pos]) {
            node = trees.children_left[pos];
        } else if (trees.threshold_types[pos] == 1 && category_in_threshold(trees.thresholds[pos], x[feature])) {
            node = trees.children_left[pos];
        } else {
            node = trees.children_right[pos];
        }
    }
}

inline void dense_tree_predict(tfloat *out, const TreeEnsemble &trees, const ExplanationDataset &data, unsigned model_transform) {
    tfloat *row_out = out;
    const tfloat *x = data.X;
    const bool *x_missing = data.X_missing;

    // see what transform (if any) we have
    transform_f transform = get_transform(model_transform);

    for (unsigned i = 0; i < data.num_X; ++i) {

        // add the base offset
        for (unsigned k = 0; k < trees.num_outputs; ++k) {
            row_out[k] += trees.base_offset[k];
        }

        // add the leaf values from each tree
        for (unsigned j = 0; j < trees.tree_limit; ++j) {
            const tfloat *leaf_value = tree_predict(j, trees, x, x_missing);

            for (unsigned k = 0; k < trees.num_outputs; ++k) {
                row_out[k] += leaf_value[k];
            }
        }

        // apply any needed transform
        if (transform != NULL) {
            const tfloat y_i = data.y == NULL ? 0 : data.y[i];
            for (unsigned k = 0; k < trees.num_outputs; ++k) {
                row_out[k] = transform(row_out[k], y_i);
            }
        }

        x += data.M;
        x_missing += data.M;
        row_out += trees.num_outputs;
    }
}

inline void tree_update_weights(unsigned i, TreeEnsemble &trees, const tfloat *x, const bool *x_missing) {
    const unsigned offset = i * trees.max_nodes;
    unsigned node = 0;
    while (true) {
        const unsigned pos = offset + node;
        const unsigned feature = trees.features[pos];

        // Record that a sample passed through this node
        trees.node_sample_weights[pos] += 1.0;

        // we hit a leaf so return a pointer to the values
        if (trees.children_left[pos] < 0) break;

        // otherwise we are at an internal node and need to recurse
        if (x_missing[feature]) {
            node = trees.children_default[pos];
        } else if (trees.threshold_types[pos] == 0 && x[feature] <= trees.thresholds[pos]) {
            node = trees.children_left[pos];
        } else if (trees.threshold_types[pos] == 1 && category_in_threshold(trees.thresholds[pos], x[feature])) {
            node = trees.children_left[pos];
        }
         else {
            node = trees.children_right[pos];
        }
    }
}

inline void dense_tree_update_weights(TreeEnsemble &trees, const ExplanationDataset &data) {
    const tfloat *x = data.X;
    const bool *x_missing = data.X_missing;

    for (unsigned i = 0; i < data.num_X; ++i) {

        // add the leaf values from each tree
        for (unsigned j = 0; j < trees.tree_limit; ++j) {
            tree_update_weights(j, trees, x, x_missing);
        }

        x += data.M;
        x_missing += data.M;
    }
}

inline void tree_saabas(tfloat *out, const TreeEnsemble &tree, const ExplanationDataset &data) {
    unsigned curr_node = 0;
    unsigned next_node = 0;
    while (true) {

        // we hit a leaf and are done
        if (tree.children_left[curr_node] < 0) return;

        // otherwise we are at an internal node and need to recurse
        const unsigned feature = tree.features[curr_node];
        if (data.X_missing[feature]) {
            next_node = tree.children_default[curr_node];
        } else if (data.X[feature] <= tree.thresholds[curr_node]) {
            next_node = tree.children_left[curr_node];
        } else {
            next_node = tree.children_right[curr_node];
        }

        // assign credit to this feature as the difference in values at the current node vs. the next node
        for (unsigned i = 0; i < tree.num_outputs; ++i) {
            out[feature * tree.num_outputs + i] += tree.values[next_node * tree.num_outputs + i] - tree.values[curr_node * tree.num_outputs + i];
        }

        curr_node = next_node;
    }
}

/**
 * This runs Tree SHAP with a per tree path conditional dependence assumption.
 */
inline void dense_tree_saabas(tfloat *out_contribs, const TreeEnsemble& trees, const ExplanationDataset &data) {
    tfloat *instance_out_contribs;
    TreeEnsemble tree;
    ExplanationDataset instance;

    // build explanation for each sample
    for (unsigned i = 0; i < data.num_X; ++i) {
        instance_out_contribs = out_contribs + i * (data.M + 1) * trees.num_outputs;
        data.get_x_instance(instance, i);

        // aggregate the effect of explaining each tree
        // (this works because of the linearity property of Shapley values)
        for (unsigned j = 0; j < trees.tree_limit; ++j) {
            trees.get_tree(tree, j);
            tree_saabas(instance_out_contribs, tree, instance);
        }

        // apply the base offset to the bias term
        for (unsigned j = 0; j < trees.num_outputs; ++j) {
            instance_out_contribs[data.M * trees.num_outputs + j] += trees.base_offset[j];
        }
    }
}


/**
 * Path-dependent Tree SHAP values and interactions.
 *
 * Original Tree SHAP: Lundberg, Erion, and Lee (2018),
 * "Consistent Individualized Feature Attribution for Tree Ensembles."
 * https://arxiv.org/abs/1802.03888
 *
 * This implementation uses Quadrature-TreeSHAP: Wettenstein, Mitchell, and Yu (2026),
 * "Quadrature-TreeSHAP: Depth-Independent TreeSHAP and Shapley Interactions."
 * https://arxiv.org/abs/2605.04497
 * Quadrature order is chosen for exact polynomial integration in double precision.
 */
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
template <bool Interactions, bool SingleOutput = false> struct Runner {
    const TreeEnsemble &tree;
    const Rule &rule;
    const std::vector<double> &ratios;
    const double *x;
    const bool *missing;
    unsigned features, n;
    size_t return_size, workspace_size;
    double *phi;
    std::vector<double> probability;
    std::vector<unsigned> path;
    std::vector<double> scratch;

    Runner(const TreeEnsemble &tree, const Rule &rule, const std::vector<double> &ratios,
           const double *x, const bool *missing, unsigned features, double *phi)
        : tree(tree), rule(rule), ratios(ratios), x(x), missing(missing), features(features),
          n(static_cast<unsigned>(rule.nodes.size())),
          return_size(static_cast<size_t>(n) * NumOutputs()), workspace_size(n + 2 * return_size),
          phi(phi), probability(features, -1),
          scratch(n + return_size + static_cast<size_t>(tree.max_depth) * workspace_size) {
        path.reserve(tree.max_depth);
        std::fill(scratch.begin(), scratch.begin() + n, 1);
    }

    // A fixed scalar extent lets the compiler remove output loops and indexing.
    // Otherwise the vector leaf output count is only known at runtime.
    unsigned NumOutputs() const { return SingleOutput ? 1 : tree.num_outputs; }

    double *Value(unsigned feature) {
        return phi + (Interactions ? static_cast<size_t>(feature) * (features + 2) : feature)
                     * NumOutputs();
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
        auto coefficient = [&](unsigned i) {
            double t = rule.nodes[i];
            return (enter - 1) / (1 + (enter - 1) * t) -
                   (previous - 1) / (1 + (previous - 1) * t);
        };
        // Vector leaves reuse coefficients across outputs. Scalar leaves compute
        // them directly in the reduction, avoiding a separate pass and buffer read.
        // The child's basis is no longer needed, so next can hold the coefficients.
        if (!SingleOutput) {
            for (unsigned i = 0; i < n; ++i) next[i] = coefficient(i);
        }
        double *value = Value(feature);
        for (unsigned o = 0; o < NumOutputs(); ++o) {
            size_t offset = static_cast<size_t>(o) * n;
            double delta = 0;
            for (unsigned i = 0; i < n; ++i) {
                double factor = SingleOutput ? coefficient(i) : next[i];
                edge[offset + i] = rule.weights[i] * result[offset + i] * factor;
                delta += edge[offset + i];
            }
            value[o] += delta;
        }
        if (Interactions) {
            for (auto partner : path) {
                if (partner == feature) continue;
                double alpha = probability[partner] - 1;
                auto denominator = [&](unsigned i) { return 1 + alpha * rule.nodes[i]; };
                if (!SingleOutput) {
                    for (unsigned i = 0; i < n; ++i) next[i] = denominator(i);
                }
                double *pair = phi + (static_cast<size_t>(feature) * (features + 1) + partner)
                                     * NumOutputs();
                for (unsigned o = 0; o < NumOutputs(); ++o) {
                    size_t offset = static_cast<size_t>(o) * n;
                    double delta = 0;
                    for (unsigned i = 0; i < n; ++i) {
                        double divisor = SingleOutput ? denominator(i) : next[i];
                        delta += edge[offset + i] * alpha / divisor;
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
            const double *values = tree.values + static_cast<size_t>(node) * NumOutputs();
            for (unsigned o = 0; o < NumOutputs(); ++o) {
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
        for (unsigned o = 0; o < NumOutputs(); ++o) baseline[o] += root_mean[o];
    }
};

inline void FinalizeInteractions(double *phi, unsigned features, unsigned outputs,
                                 const std::vector<unsigned> &active_features) {
    size_t columns = features + 1;
    for (unsigned output = 0; output < outputs; ++output) {
        for (size_t a_idx = 0; a_idx < active_features.size(); ++a_idx) {
            unsigned i = active_features[a_idx];
            for (size_t b_idx = a_idx + 1; b_idx < active_features.size(); ++b_idx) {
                unsigned j = active_features[b_idx];
                double &a = phi[(i * columns + j) * outputs + output];
                double &b = phi[(j * columns + i) * outputs + output];
                a = b = 0.5 * (a + b);
            }
        }
        for (unsigned i : active_features) {
            double &diag = phi[(i * columns + i) * outputs + output];
            for (unsigned j : active_features) {
                if (i != j) diag -= phi[(i * columns + j) * outputs + output];
            }
        }
    }
}

template <bool Interactions, bool SingleOutput = false>
inline void ExplainTree(const TreeEnsemble &tree, const Rule &rule,
                        const std::vector<double> &ratios, const std::vector<double> &mean,
                        const ExplanationDataset &data, double *phi, size_t stride) {
    Runner<Interactions, SingleOutput> runner(tree, rule, ratios, nullptr, nullptr, data.M, nullptr);
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
    // Unused features have zero interactions; avoid scanning their matrix entries.
    std::vector<bool> used_features(Interactions ? data.M : 0, false);
    // Prepare cover ratios and quadrature once per tree, not once per explained row.
    for (unsigned t = 0; t < trees.tree_limit; ++t) {
        TreeEnsemble tree;
        trees.get_tree(tree, t);
        if (Interactions) {
            for (unsigned node = 0; node < tree.max_nodes; ++node) {
                if (!tree.is_leaf(node)) used_features[tree.features[node]] = true;
            }
        }
        Rule rule(RuleSize(tree, data.M));
        auto ratios = CoverRatios(tree);
        std::vector<double> mean(tree.num_outputs, 0);
        RootMean(tree, ratios, 0, 1, mean);
        // Dispatch once per tree; quadrature point counts remain runtime-sized.
        if (tree.num_outputs == 1) {
            ExplainTree<Interactions, true>(tree, rule, ratios, mean, data, phi, stride);
        } else {
            ExplainTree<Interactions, false>(tree, rule, ratios, mean, data, phi, stride);
        }
    }
    std::vector<unsigned> active_features;
    if (Interactions) {
        for (unsigned feature = 0; feature < data.M; ++feature) {
            if (used_features[feature]) active_features.push_back(feature);
        }
    }
    for (unsigned row = 0; row < data.num_X; ++row) {
        double *out = phi + row * stride;
        for (unsigned output = 0; output < trees.num_outputs; ++output) {
            out[stride - trees.num_outputs + output] += trees.base_offset[output];
        }
        if (Interactions) FinalizeInteractions(out, data.M, trees.num_outputs, active_features);
    }
}
}  // namespace shap_quadrature

inline int compute_expectations(TreeEnsemble &tree, int i = 0, int depth = 0) {
    unsigned max_depth = 0;

    if (tree.children_right[i] >= 0) {
        const unsigned li = tree.children_left[i];
        const unsigned ri = tree.children_right[i];
        const unsigned depth_left = compute_expectations(tree, li, depth + 1);
        const unsigned depth_right = compute_expectations(tree, ri, depth + 1);
        const tfloat left_weight = tree.node_sample_weights[li];
        const tfloat right_weight = tree.node_sample_weights[ri];
        const unsigned li_offset = li * tree.num_outputs;
        const unsigned ri_offset = ri * tree.num_outputs;
        const unsigned i_offset = i * tree.num_outputs;
        for (unsigned j = 0; j < tree.num_outputs; ++j) {
            if ((left_weight == 0) && (right_weight == 0)) {
                tree.values[i_offset + j] = 0.0;
            } else {
                const tfloat v = (left_weight * tree.values[li_offset + j] + right_weight * tree.values[ri_offset + j]) / (left_weight + right_weight);
                tree.values[i_offset + j] = v;
            }
        }
        max_depth = std::max(depth_left, depth_right) + 1;
    }

    if (depth == 0) tree.max_depth = max_depth;

    return max_depth;
}

// Independent Tree SHAP functions below here
// ------------------------------------------
struct Node {
    int cl, cr, cd, pnode; // was int16; node indices can exceed 2^15 - 1, see GH#3486
    long feat, pfeat;
    float thres, value;
    char from_flag;
};

#define FROM_NEITHER 0
#define FROM_X_NOT_R 1
#define FROM_R_NOT_X 2

// https://www.geeksforgeeks.org/space-and-time-efficient-binomial-coefficient/
inline int bin_coeff(int n, int k) {
    int res = 1;
    if (k > n - k)
        k = n - k;
    for (int i = 0; i < k; ++i) {
        res *= (n - i);
        res /= (i + 1);
    }
    return res;
}

// note this only handles single output models, so multi-output models get explained using multiple passes
inline void tree_shap_indep(const unsigned max_depth, const unsigned num_feats,
                            const unsigned num_nodes, const tfloat *x,
                            const bool *x_missing, const tfloat *r,
                            const bool *r_missing, tfloat *out_contribs,
                            float *pos_lst, float *neg_lst, signed short *feat_hist,
                            float *memoized_weights, int *node_stack, Node *mytree) {

//     const bool DEBUG = true;
//     ofstream myfile;
//     if (DEBUG) {
//       myfile.open ("/homes/gws/hughchen/shap/out.txt",fstream::app);
//       myfile << "Entering tree_shap_indep\n";
//     }
    int ns_ctr = 0;
    std::fill_n(feat_hist, num_feats, 0);
    int node = 0, cl, cr, cd, pnode;
    long feat, pfeat = -1;
    int next_xnode = -1, next_rnode = -1;
    int next_node = -1, from_child = -1;
    float thres, pos_x = 0, neg_x = 0, pos_r = 0, neg_r = 0;
    char from_flag;
    unsigned M = 0, N = 0;

    Node curr_node = mytree[node];
    feat = curr_node.feat;
    thres = curr_node.thres;
    cl = curr_node.cl;
    cr = curr_node.cr;
    cd = curr_node.cd;

    // short circuit when this is a stump tree (with no splits)
    if (cl < 0) {
        out_contribs[num_feats] += curr_node.value;
        return;
    }

//     if (DEBUG) {
//       myfile << "\nNode: " << node << "\n";
//       myfile << "x[feat]: " << x[feat] << ", r[feat]: " << r[feat] << "\n";
//       myfile << "thres: " << thres << "\n";
//     }

    if (x_missing[feat]) {
        next_xnode = cd;
    } else if (x[feat] > thres) {
        next_xnode = cr;
    } else if (x[feat] <= thres) {
        next_xnode = cl;
    }

    if (r_missing[feat]) {
        next_rnode = cd;
    } else if (r[feat] > thres) {
        next_rnode = cr;
    } else if (r[feat] <= thres) {
        next_rnode = cl;
    }

    if (next_xnode != next_rnode) {
        mytree[next_xnode].from_flag = FROM_X_NOT_R;
        mytree[next_rnode].from_flag = FROM_R_NOT_X;
    } else {
        mytree[next_xnode].from_flag = FROM_NEITHER;
    }

    // Check if x and r go the same way
    if (next_xnode == next_rnode) {
        next_node = next_xnode;
    }

    // If not, go left
    if (next_node < 0) {
        next_node = cl;
        if (next_rnode == next_node) { // rpath
            N = N+1;
            feat_hist[feat] -= 1;
        } else if (next_xnode == next_node) { // xpath
            M = M+1;
            N = N+1;
            feat_hist[feat] += 1;
        }
    }
    node_stack[ns_ctr] = node;
    ns_ctr += 1;
    while (true) {
        node = next_node;
        curr_node = mytree[node];
        feat = curr_node.feat;
        thres = curr_node.thres;
        cl = curr_node.cl;
        cr = curr_node.cr;
        cd = curr_node.cd;
        pnode = curr_node.pnode;
        pfeat = curr_node.pfeat;
        from_flag = curr_node.from_flag;



//         if (DEBUG) {
//           myfile << "\nNode: " << node << "\n";
//           myfile << "N: " << N << ", M: " << M << "\n";
//           myfile << "from_flag==FROM_X_NOT_R: " << (from_flag==FROM_X_NOT_R) << "\n";
//           myfile << "from_flag==FROM_R_NOT_X: " << (from_flag==FROM_R_NOT_X) << "\n";
//           myfile << "from_flag==FROM_NEITHER: " << (from_flag==FROM_NEITHER) << "\n";
//           myfile << "feat_hist[feat]: " << feat_hist[feat] << "\n";
//         }

        // At a leaf
        if (cl < 0) {
            //      if (DEBUG) {
            //        myfile << "At a leaf\n";
            //      }

            if (M == 0) {
              out_contribs[num_feats] += mytree[node].value;
            }

            // Currently assuming a single output
            if (N != 0) {
                if (M != 0) {
                    pos_lst[node] = mytree[node].value * memoized_weights[N + max_depth * (M-1)];
                }
                if (M != N) {
                    neg_lst[node] = -mytree[node].value * memoized_weights[N + max_depth * M];
                }
            }
//             if (DEBUG) {
//               myfile << "pos_lst[node]: " << pos_lst[node] << "\n";
//               myfile << "neg_lst[node]: " << neg_lst[node] << "\n";
//             }
            // Pop from node_stack
            ns_ctr -= 1;
            next_node = node_stack[ns_ctr];
            from_child = node;
            // Unwind
            if (feat_hist[pfeat] > 0) {
                feat_hist[pfeat] -= 1;
            } else if (feat_hist[pfeat] < 0) {
                feat_hist[pfeat] += 1;
            }
            if (feat_hist[pfeat] == 0) {
                if (from_flag == FROM_X_NOT_R) {
                    N = N-1;
                    M = M-1;
                } else if (from_flag == FROM_R_NOT_X) {
                    N = N-1;
                }
            }
            continue;
        }

        const bool x_right = x[feat] > thres;
        const bool r_right = r[feat] > thres;

        if (x_missing[feat]) {
            next_xnode = cd;
        } else if (x_right) {
            next_xnode = cr;
        } else if (!x_right) {
            next_xnode = cl;
        }

        if (r_missing[feat]) {
            next_rnode = cd;
        } else if (r_right) {
            next_rnode = cr;
        } else if (!r_right) {
            next_rnode = cl;
        }

        if (next_xnode >= 0) {
          if (next_xnode != next_rnode) {
              mytree[next_xnode].from_flag = FROM_X_NOT_R;
              mytree[next_rnode].from_flag = FROM_R_NOT_X;
          } else {
              mytree[next_xnode].from_flag = FROM_NEITHER;
          }
        }

        // Arriving at node from parent
        if (from_child == -1) {
            //      if (DEBUG) {
            //        myfile << "Arriving at node from parent\n";
            //      }
            node_stack[ns_ctr] = node;
            ns_ctr += 1;
            next_node = -1;

            //      if (DEBUG) {
            //        myfile << "feat_hist[feat]" << feat_hist[feat] << "\n";
            //      }
            // Feature is set upstream
            if (feat_hist[feat] > 0) {
                next_node = next_xnode;
                feat_hist[feat] += 1;
            } else if (feat_hist[feat] < 0) {
                next_node = next_rnode;
                feat_hist[feat] -= 1;
            }

            // x and r go the same way
            if (next_node < 0) {
                if (next_xnode == next_rnode) {
                    next_node = next_xnode;
                }
            }

            // Go down one path
            if (next_node >= 0) {
                continue;
            }

            // Go down both paths, but go left first
            next_node = cl;
            if (next_rnode == next_node) {
                N = N+1;
                feat_hist[feat] -= 1;
            } else if (next_xnode == next_node) {
                M = M+1;
                N = N+1;
                feat_hist[feat] += 1;
            }
            from_child = -1;
            continue;
        }

        // Arriving at node from child
        if (from_child != -1) {
//             if (DEBUG) {
//               myfile << "Arriving at node from child\n";
//             }
            next_node = -1;
            // Check if we should unroll immediately
            if ((next_rnode == next_xnode) || (feat_hist[feat] != 0)) {
                next_node = pnode;
            }

            // Came from a single path, so unroll
            if (next_node >= 0) {
//                 if (DEBUG) {
//                   myfile << "Came from a single path, so unroll\n";
//                 }
                // At the root node
                if (node == 0) {
                    break;
                }
                // Update and unroll
                pos_lst[node] = pos_lst[from_child];
                neg_lst[node] = neg_lst[from_child];

//                 if (DEBUG) {
//                   myfile << "pos_lst[node]: " << pos_lst[node] << "\n";
//                   myfile << "neg_lst[node]: " << neg_lst[node] << "\n";
//                 }
                from_child = node;
                ns_ctr -= 1;

                // Unwind
                if (feat_hist[pfeat] > 0) {
                    feat_hist[pfeat] -= 1;
                } else if (feat_hist[pfeat] < 0) {
                    feat_hist[pfeat] += 1;
                }
                if (feat_hist[pfeat] == 0) {
                    if (from_flag == FROM_X_NOT_R) {
                        N = N-1;
                        M = M-1;
                    } else if (from_flag == FROM_R_NOT_X) {
                        N = N-1;
                    }
                }
                continue;
                // Go right - Arriving from the left child
            } else if (from_child == cl) {
//                 if (DEBUG) {
//                   myfile << "Go right - Arriving from the left child\n";
//                 }
                node_stack[ns_ctr] = node;
                ns_ctr += 1;
                next_node = cr;
                if (next_xnode == next_node) {
                    M = M+1;
                    N = N+1;
                    feat_hist[feat] += 1;
                } else if (next_rnode == next_node) {
                    N = N+1;
                    feat_hist[feat] -= 1;
                }
                from_child = -1;
                continue;
                // Compute stuff and unroll - Arriving from the right child
            } else if (from_child == cr) {
//                 if (DEBUG) {
//                   myfile << "Compute stuff and unroll - Arriving from the right child\n";
//                 }
                pos_x = 0;
                neg_x = 0;
                pos_r = 0;
                neg_r = 0;
                if ((next_xnode == cr) && (next_rnode == cl)) {
                    pos_x = pos_lst[cr];
                    neg_x = neg_lst[cr];
                    pos_r = pos_lst[cl];
                    neg_r = neg_lst[cl];
                } else if ((next_xnode == cl) && (next_rnode == cr)) {
                    pos_x = pos_lst[cl];
                    neg_x = neg_lst[cl];
                    pos_r = pos_lst[cr];
                    neg_r = neg_lst[cr];
                }
                // out_contribs needs to have been initialized as all zeros
                // if (pos_x + neg_r != 0) {
                //   std::cout << "val " << pos_x + neg_r << "\n";
                // }
                out_contribs[feat] += pos_x + neg_r;
                pos_lst[node] = pos_x + pos_r;
                neg_lst[node] = neg_x + neg_r;

//                 if (DEBUG) {
//                   myfile << "out_contribs[feat]: " << out_contribs[feat] << "\n";
//                   myfile << "pos_lst[node]: " << pos_lst[node] << "\n";
//                   myfile << "neg_lst[node]: " << neg_lst[node] << "\n";
//                 }

                // Check if at root
                if (node == 0) {
                    break;
                }

                // Pop
                ns_ctr -= 1;
                next_node = node_stack[ns_ctr];
                from_child = node;

                // Unwind
                if (feat_hist[pfeat] > 0) {
                    feat_hist[pfeat] -= 1;
                } else if (feat_hist[pfeat] < 0) {
                    feat_hist[pfeat] += 1;
                }
                if (feat_hist[pfeat] == 0) {
                    if (from_flag == FROM_X_NOT_R) {
                        N = N-1;
                        M = M-1;
                    } else if (from_flag == FROM_R_NOT_X) {
                        N = N-1;
                    }
                }
                continue;
            }
        }
    }
    //  if (DEBUG) {
    //    myfile.close();
    //  }
}


inline void print_progress_bar(tfloat &last_print, tfloat start_time, unsigned i, unsigned total_count) {
    const tfloat elapsed_seconds = difftime(time(NULL), start_time);

    if (elapsed_seconds > 10 && elapsed_seconds - last_print > 0.5) {
        const tfloat fraction = static_cast<tfloat>(i) / total_count;
        const double total_seconds = elapsed_seconds / fraction;
        last_print = elapsed_seconds;

        PySys_WriteStderr(
            "\r%3.0f%%|%.*s%.*s| %d/%d [%02d:%02d<%02d:%02d]       ",
            fraction * 100, int(0.5 + fraction*20), "===================",
            20-int(0.5 + fraction*20), "                   ",
            i, total_count,
            int(elapsed_seconds/60), int(elapsed_seconds) % 60,
            int((total_seconds - elapsed_seconds)/60), int(total_seconds - elapsed_seconds) % 60
        );

        // Get handle to python stderr file and flush it (https://mail.python.org/pipermail/python-list/2004-November/294912.html)
        PyObject *pyStderr = PySys_GetObject("stderr");
        if (pyStderr) {
            PyObject *result = PyObject_CallMethod(pyStderr, "flush", NULL);
            Py_XDECREF(result);
        }
    }
}

/**
 * Runs Tree SHAP with feature independence assumptions on dense data.
 */
inline void dense_independent(const TreeEnsemble& trees, const ExplanationDataset &data,
                       tfloat *out_contribs, tfloat transform(const tfloat, const tfloat)) {

    // reformat the trees for faster access
    Node *node_trees = new Node[trees.tree_limit * trees.max_nodes];
    for (unsigned i = 0; i < trees.tree_limit; ++i) {
        Node *node_tree = node_trees + i * trees.max_nodes;
        for (unsigned j = 0; j < trees.max_nodes; ++j) {
            const unsigned en_ind = i * trees.max_nodes + j;
            node_tree[j].cl = trees.children_left[en_ind];
            node_tree[j].cr = trees.children_right[en_ind];
            node_tree[j].cd = trees.children_default[en_ind];
            if (j == 0) {
                node_tree[j].pnode = 0;
            }
            if (trees.children_left[en_ind] >= 0) { // relies on all unused entries having negative values in them
                node_tree[trees.children_left[en_ind]].pnode = j;
                node_tree[trees.children_left[en_ind]].pfeat = trees.features[en_ind];
            }
            if (trees.children_right[en_ind] >= 0) { // relies on all unused entries having negative values in them
                node_tree[trees.children_right[en_ind]].pnode = j;
                node_tree[trees.children_right[en_ind]].pfeat = trees.features[en_ind];
            }

            node_tree[j].thres = trees.thresholds[en_ind];
            node_tree[j].feat = trees.features[en_ind];
        }
    }

    // preallocate arrays needed by the algorithm
    float *pos_lst = new float[trees.max_nodes];
    float *neg_lst = new float[trees.max_nodes];
    int *node_stack = new int[(unsigned) trees.max_depth];
    signed short *feat_hist = new signed short[data.M];
    tfloat *tmp_out_contribs = new tfloat[(data.M + 1)];

    // precompute all the weight coefficients
    float *memoized_weights = new float[(trees.max_depth+1) * (trees.max_depth+1)];
    for (unsigned n = 0; n <= trees.max_depth; ++n) {
        for (unsigned m = 0; m <= trees.max_depth; ++m) {
            memoized_weights[n + trees.max_depth * m] = 1.0 / (n * bin_coeff(n-1, m));
        }
    }

    // compute the explanations for each sample
    tfloat *instance_out_contribs;
    tfloat rescale_factor = 1.0;
    tfloat margin_x = 0;
    tfloat margin_r = 0;
    time_t start_time = time(NULL);
    tfloat last_print = 0;
    for (unsigned oind = 0; oind < trees.num_outputs; ++oind) {
        // set the values in the reformatted tree to the current output index
        for (unsigned i = 0; i < trees.tree_limit; ++i) {
            Node *node_tree = node_trees + i * trees.max_nodes;
            for (unsigned j = 0; j < trees.max_nodes; ++j) {
                const unsigned en_ind = i * trees.max_nodes + j;
                node_tree[j].value = trees.values[en_ind * trees.num_outputs + oind];
            }
        }

        // loop over all the samples
        for (unsigned i = 0; i < data.num_X; ++i) {
            const tfloat *x = data.X + i * data.M;
            const bool *x_missing = data.X_missing + i * data.M;
            instance_out_contribs = out_contribs + i * (data.M + 1) * trees.num_outputs;
            const tfloat y_i = data.y == NULL ? 0 : data.y[i];

            print_progress_bar(last_print, start_time, oind * data.num_X + i, data.num_X * trees.num_outputs);

            // compute the model's margin output for x
            if (transform != NULL) {
                margin_x = trees.base_offset[oind];
                for (unsigned k = 0; k < trees.tree_limit; ++k) {
                    margin_x += tree_predict(k, trees, x, x_missing)[oind];
                }
            }

            for (unsigned j = 0; j < data.num_R; ++j) {
                const tfloat *r = data.R + j * data.M;
                const bool *r_missing = data.R_missing + j * data.M;
                std::fill_n(tmp_out_contribs, (data.M + 1), 0);

                // compute the model's margin output for r
                if (transform != NULL) {
                    margin_r = trees.base_offset[oind];
                    for (unsigned k = 0; k < trees.tree_limit; ++k) {
                        margin_r += tree_predict(k, trees, r, r_missing)[oind];
                    }
                }

                for (unsigned k = 0; k < trees.tree_limit; ++k) {
                    tree_shap_indep(
                        trees.max_depth, data.M, trees.max_nodes, x, x_missing, r, r_missing,
                        tmp_out_contribs, pos_lst, neg_lst, feat_hist, memoized_weights,
                        node_stack, node_trees + k * trees.max_nodes
                    );
                }

                // compute the rescale factor
                if (transform != NULL) {
                    if (margin_x == margin_r) {
                        rescale_factor = 1.0;
                    } else {
                        rescale_factor = (*transform)(margin_x, y_i) - (*transform)(margin_r, y_i);
                        rescale_factor /= margin_x - margin_r;
                    }
                }

                // add the effect of the current reference to our running total
                // this is where we can do per reference scaling for non-linear transformations
                for (unsigned k = 0; k < data.M; ++k) {
                    instance_out_contribs[k * trees.num_outputs + oind] += tmp_out_contribs[k] * rescale_factor;
                }

                // Add the base offset
                if (transform != NULL) {
                    instance_out_contribs[data.M * trees.num_outputs + oind] += (*transform)(trees.base_offset[oind] + tmp_out_contribs[data.M], 0);
                } else {
                    instance_out_contribs[data.M * trees.num_outputs + oind] += trees.base_offset[oind] + tmp_out_contribs[data.M];
                }
            }

            // average the results over all the references.
            for (unsigned j = 0; j < (data.M + 1); ++j) {
                instance_out_contribs[j * trees.num_outputs + oind] /= data.num_R;
            }

            // apply the base offset to the bias term
            // for (unsigned j = 0; j < trees.num_outputs; ++j) {
            //     instance_out_contribs[data.M * trees.num_outputs + j] += (*transform)(trees.base_offset[j], 0);
            // }
        }
    }

    delete[] tmp_out_contribs;
    delete[] node_trees;
    delete[] pos_lst;
    delete[] neg_lst;
    delete[] node_stack;
    delete[] feat_hist;
    delete[] memoized_weights;
}


/**
 * The main method for computing Tree SHAP on models using dense data.
 */
inline void dense_tree_shap(const TreeEnsemble& trees, const ExplanationDataset &data, tfloat *out_contribs,
                     const int feature_dependence, unsigned model_transform, bool interactions) {

    // see what transform (if any) we have
    transform_f transform = get_transform(model_transform);

    // dispatch to the correct algorithm handler
    switch (feature_dependence) {
        case FEATURE_DEPENDENCE::independent:
            if (interactions) {
                std::cerr << "FEATURE_DEPENDENCE::independent does not support interactions!\n";
            } else dense_independent(trees, data, out_contribs, transform);
            return;

        case FEATURE_DEPENDENCE::tree_path_dependent:
            if (interactions) shap_quadrature::Explain<true>(trees, data, out_contribs);
            else shap_quadrature::Explain<false>(trees, data, out_contribs);
            return;
    }
}

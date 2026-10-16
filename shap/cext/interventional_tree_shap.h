#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
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

// Per-player shares indexed by (a, b), already divided by a or b.
class Coefficients {
public:
    struct Share {
        double positive = 0, negative = 0;
    };

    explicit Coefficients(std::size_t max_players)
        : size_(max_players + 1), shares_(size_ * size_) {
        for (std::size_t a = 0; a < size_; ++a) {
            for (std::size_t b = 0; a + b < size_; ++b) {
                // 1 / C(a+b, a) by a short multiplicative recurrence, avoiding overflow.
                auto [small, large] = std::minmax(a, b);
                double inverse = 1;
                for (std::size_t i = 1; i <= small; ++i) inverse *= static_cast<double>(i) / (large + i);
                auto &share = shares_[a * size_ + b];
                if (a > 0) share.positive = inverse / a;
                if (b > 0) share.negative = -inverse / b;
            }
        }
    }

    std::size_t MaxPlayers() const { return size_ - 1; }
    const Share &operator()(std::size_t a, std::size_t b) const { return shares_[a * size_ + b]; }

private:
    std::size_t size_;
    std::vector<Share> shares_;
};

inline bool GoesLeft(const TreeEnsemble &tree, unsigned node, const double *x, const bool *missing) {
    unsigned feature = tree.features[node];
    if (missing[feature]) return tree.children_default[node] == tree.children_left[node];
    if (tree.threshold_types[node] == 1) return category_in_threshold(tree.thresholds[node], x[feature]);
    return x[feature] <= tree.thresholds[node];
}

// One explained or background row, with its missing-value mask.
struct Row {
    const double *values;
    const bool *missing;
};

inline Row ForegroundRow(const ExplanationDataset &data, std::size_t i) {
    return {data.X + i * data.M, data.X_missing + i * data.M};
}

inline Row BackgroundRow(const ExplanationDataset &data, std::size_t j) {
    return {data.R + j * data.M, data.R_missing + j * data.M};
}

// Where a feature on the current path takes its value from.
enum class Source : std::int8_t { unassigned, foreground, background };

template <bool SingleOutput> class Traversal {
public:
    Traversal(const Coefficients &coefficients, std::size_t features, std::size_t outputs)
        : coefficients_(coefficients), features_(features), outputs_(outputs),
          source_(features, Source::unassigned),
          // A root slot, then foreground and background child slots for every player level.
          scratch_((1 + 2 * coefficients.MaxPlayers()) * 2 * outputs) {}

    // Add the attributions of one tree for (x, r) into phi, laid out as (features + 1, outputs).
    void Run(const TreeEnsemble &tree, Row x, Row r, double *phi) {
        tree_ = &tree;
        x_ = x;
        r_ = r;
        phi_ = phi;
        Visit(0, 0, 0, Slot(0));
    }

private:
    // A subtree's summed shares for each foreground player and each background player.
    struct Shares {
        double *positive, *negative;
    };

    std::size_t NumOutputs() const { return SingleOutput ? 1 : outputs_; }

    Shares Slot(std::size_t slot) {
        double *begin = scratch_.data() + slot * 2 * NumOutputs();
        return {begin, begin + NumOutputs()};
    }

    double *Phi(std::size_t feature) { return phi_ + feature * NumOutputs(); }

    // Every foreground player of a leaf receives the same share, as does every
    // background player. Sum these shares over a subtree and credit each player once
    // at its split, keeping the work per pair linear in the reachable nodes.
    void Leaf(unsigned node, std::size_t a, std::size_t b, Shares out) {
        const double *values = tree_->values + node * NumOutputs();
        const auto &share = coefficients_(a, b);
        double *baseline = Phi(features_);
        for (std::size_t o = 0; o < NumOutputs(); ++o) {
            out.positive[o] = values[o] * share.positive;
            out.negative[o] = values[o] * share.negative;
            if (a == 0) baseline[o] += values[o];
        }
    }

    // Finish the foreground subtree before visiting its background sibling.
    void Visit(unsigned node, std::size_t a, std::size_t b, Shares out) {
        while (!tree_->is_leaf(node)) {
            unsigned feature = tree_->features[node];
            bool x_left = GoesLeft(*tree_, node, x_.values, x_.missing);
            bool r_left = GoesLeft(*tree_, node, r_.values, r_.missing);
            Source &source = source_[feature];
            if (x_left != r_left && source == Source::unassigned) {
                unsigned left = tree_->children_left[node], right = tree_->children_right[node];
                std::size_t players = a + b;
                Shares x_shares = Slot(1 + 2 * players), r_shares = Slot(2 + 2 * players);
                source = Source::foreground;
                Visit(x_left ? left : right, a + 1, b, x_shares);
                source = Source::background;
                Visit(r_left ? left : right, a, b + 1, r_shares);
                source = Source::unassigned;
                double *value = Phi(feature);
                for (std::size_t o = 0; o < NumOutputs(); ++o) {
                    value[o] += x_shares.positive[o] + r_shares.negative[o];
                    out.positive[o] = x_shares.positive[o] + r_shares.positive[o];
                    out.negative[o] = x_shares.negative[o] + r_shares.negative[o];
                }
                return;
            }
            // Agreement, or a repeated split following its feature's assigned source.
            bool go_left = x_left == r_left || source == Source::foreground ? x_left : r_left;
            node = go_left ? tree_->children_left[node] : tree_->children_right[node];
        }
        Leaf(node, a, b, out);
    }

    const Coefficients &coefficients_;
    std::size_t features_, outputs_;
    const TreeEnsemble *tree_ = nullptr;
    Row x_{}, r_{};
    double *phi_ = nullptr;
    // Every assignment is undone while unwinding, so this is shared across trees and pairs.
    std::vector<Source> source_;
    std::vector<double> scratch_;
};

// Size the coefficient table from the trees rather than trusting max_depth metadata.
inline std::size_t Depth(const TreeEnsemble &tree, unsigned node) {
    if (tree.is_leaf(node)) return 0;
    return 1 + std::max(Depth(tree, tree.children_left[node]), Depth(tree, tree.children_right[node]));
}

template <bool SingleOutput>
inline void ExplainRows(const TreeEnsemble &trees, const std::vector<TreeEnsemble> &views,
                        const ExplanationDataset &data, double *phi, transform_f transform) {
    const std::size_t outputs = trees.num_outputs, features = data.M;
    const std::size_t stride = (features + 1) * outputs;
    std::size_t depth = 0;
    for (const auto &view : views) depth = std::max(depth, Depth(view, 0));
    Coefficients coefficients(std::min<std::size_t>(depth, features));
    Traversal<SingleOutput> traversal(coefficients, features, outputs);

    // Non-linear transforms rescale each reference's attributions by its own margins,
    // so they are accumulated per reference before being combined.
    std::vector<double> pairs(transform ? data.num_R * stride : 0);
    std::vector<double> margin_x(outputs), margin_r(outputs);
    auto margins = [&](Row row, std::vector<double> &margin) {
        std::copy_n(trees.base_offset, outputs, margin.begin());
        for (unsigned t = 0; t < trees.tree_limit; ++t) {
            const double *values = tree_predict(t, trees, row.values, row.missing);
            std::transform(margin.begin(), margin.end(), values, margin.begin(), std::plus<>());
        }
    };

    for (std::size_t i = 0; i < data.num_X; ++i) {
        double *out = phi + i * stride;
        const Row x = ForegroundRow(data, i);
        const double y = data.y ? data.y[i] : 0;
        if (transform) margins(x, margin_x);

        // Visit every reference within a tree, keeping that tree's nodes in cache
        // instead of streaming the whole ensemble once per reference.
        std::fill(pairs.begin(), pairs.end(), 0);
        for (const auto &view : views) {
            for (std::size_t j = 0; j < data.num_R; ++j) {
                traversal.Run(view, x, BackgroundRow(data, j), transform ? pairs.data() + j * stride : out);
            }
        }

        if (transform) {
            for (std::size_t j = 0; j < data.num_R; ++j) {
                const double *pair = pairs.data() + j * stride;
                margins(BackgroundRow(data, j), margin_r);
                for (std::size_t o = 0; o < outputs; ++o) {
                    double rescale = 1;
                    if (margin_x[o] != margin_r[o]) {
                        rescale = (transform(margin_x[o], y) - transform(margin_r[o], y)) /
                                  (margin_x[o] - margin_r[o]);
                    }
                    for (std::size_t k = 0; k < features; ++k) out[k * outputs + o] += pair[k * outputs + o] * rescale;
                    out[features * outputs + o] += transform(trees.base_offset[o] + pair[features * outputs + o], 0);
                }
            }
        }

        std::transform(out, out + stride, out, [&](double value) { return value / data.num_R; });
        if (!transform) {
            double *baseline = out + features * outputs;
            std::transform(baseline, baseline + outputs, trees.base_offset, baseline, std::plus<>());
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

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <utility>
#include <vector>

// TreeEnsemble, ExplanationDataset and tree_predict are defined by tree_shap.h.
/**
 * Interventional Tree SHAP values with an empirical background distribution.
 *
 * Baseline SHAP explains a foreground row x against a single background row r:
 * every coalition takes each feature from either x or r. Splits where x and r
 * agree are followed directly, and a feature that already has a source on the
 * current path follows that source, so only distinct disagreeing features
 * become players. At a reachable leaf with a foreground and b background
 * assignments, the coalition polynomial is t^a (1-t)^b. Integrating its
 * derivative gives each foreground feature 1 / (a * C(a+b, a)) and each
 * background feature -1 / (b * C(a+b, b)) of the leaf value. Other features are
 * dummies for that leaf. The leaf reached with every assignment from the
 * background (a = 0) contributes to the baseline. Unlike quadrature, these
 * coefficients are exact and need no cover statistics.
 *
 * Interventional (background) SHAP is the average of baseline SHAP over every
 * background row.
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
        for (std::size_t large = 0; large < size_; ++large) {
            // 1 / C(small+large, small), one factor below 1 per step in small, avoiding overflow.
            double inverse = 1;
            for (std::size_t small = 0; small <= large && small + large < size_; ++small) {
                if (small > 0) inverse *= static_cast<double>(small) / (large + small);
                // Symmetric in a and b, so it fills both orientations.
                for (auto [a, b] : {std::pair{small, large}, std::pair{large, small}}) {
                    auto &share = shares_[a * size_ + b];
                    if (a > 0) share.positive = inverse / a;
                    if (b > 0) share.negative = -inverse / b;
                }
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

inline std::size_t Leaves(const TreeEnsemble &tree, unsigned node) {
    if (tree.is_leaf(node)) return 1;
    return Leaves(tree, tree.children_left[node]) + Leaves(tree, tree.children_right[node]);
}

// The most distinct features on any root-to-leaf path, with uses counting each
// feature's splits on the current path. Repeated features share a pattern bit.
inline std::size_t PathFeatures(const TreeEnsemble &tree, unsigned node, std::vector<unsigned> &uses) {
    if (tree.is_leaf(node)) return 0;
    unsigned feature = tree.features[node];
    std::size_t added = uses[feature]++ == 0;
    std::size_t below = std::max(PathFeatures(tree, tree.children_left[node], uses),
                                 PathFeatures(tree, tree.children_right[node], uses));
    --uses[feature];
    return added + below;
}

/**
 * Decision-pattern algorithm, after WOODELF's background SHAP:
 * https://ron-wettenstein.github.io/TreeBranchMarks/benchmarks/resources/mn_background_explanation.html
 *
 * For a leaf, give each distinct feature on its path a bit, set for a row that satisfies
 * all of that feature's splits on the path. A foreground pattern c and background
 * pattern b reach the leaf only if c | b covers every bit. Then the foreground players
 * are exactly the zero bits of b and the background players the zero bits of c, so a
 * pair's shares depend only on the two popcounts. Rows with equal patterns share all
 * work, costing unique foreground x unique background patterns per leaf instead of
 * foreground x background rows. Attributions are summed over references, so this
 * applies to raw outputs only.
 */
// Sixteen bits bound the dense pattern lookup table to 2^16 slots.
using Pattern = std::uint16_t;
constexpr std::size_t kMaxPatternBits = 16;
static_assert(kMaxPatternBits <= 8 * sizeof(Pattern));

template <bool SingleOutput> class Patterns {
public:
    Patterns(const Coefficients &coefficients, const ExplanationDataset &data, std::size_t outputs,
             std::size_t depth)
        : coefficients_(coefficients), data_(data), outputs_(outputs), bit_of_feature_(data.M, -1),
          foreground_((depth + 1) * data.num_X), background_((depth + 1) * data.num_R),
          x_left_(depth * data.num_X), r_left_(depth * data.num_R), row_slot_(data.num_X),
          slot_(std::size_t(1) << kMaxPatternBits, -1) {
        for (std::size_t i = 0; i < data.num_X; ++i) x_.push_back(ForegroundRow(data, i));
        for (std::size_t j = 0; j < data.num_R; ++j) r_.push_back(BackgroundRow(data, j));
    }

    // Add one tree's attributions, summed rather than averaged over references, into phi.
    void Run(const TreeEnsemble &tree, double *phi) {
        tree_ = &tree;
        phi_ = phi;
        Visit(0, 0, 0);
    }

private:
    std::size_t NumOutputs() const { return SingleOutput ? 1 : outputs_; }
    std::size_t Stride() const { return (data_.M + 1) * NumOutputs(); }

    // Patterns of every row at a depth; depth zero is the root, where no bits are in use.
    Pattern *Foreground(std::size_t depth) { return foreground_.data() + depth * data_.num_X; }
    Pattern *Background(std::size_t depth) { return background_.data() + depth * data_.num_R; }

    // Each row's child pattern: a new feature takes the condition as its bit, and a
    // repeated feature must also satisfy it.
    static void Descend(const Pattern *parent, const std::uint8_t *left, bool to_left, Pattern bit, bool repeated,
                        Pattern *child, std::size_t rows) {
        for (std::size_t i = 0; i < rows; ++i) {
            bool satisfied = left[i] == to_left;
            child[i] = repeated ? (satisfied ? parent[i] : parent[i] & ~bit) : (satisfied ? parent[i] | bit : parent[i]);
        }
    }

    void Visit(unsigned node, std::size_t depth, std::size_t bits) {
        if (tree_->is_leaf(node)) return Leaf(node, depth, bits);
        unsigned feature = tree_->features[node];
        std::uint8_t *x_left = x_left_.data() + depth * data_.num_X, *r_left = r_left_.data() + depth * data_.num_R;
        for (std::size_t i = 0; i < data_.num_X; ++i) x_left[i] = GoesLeft(*tree_, node, x_[i].values, x_[i].missing);
        for (std::size_t j = 0; j < data_.num_R; ++j) r_left[j] = GoesLeft(*tree_, node, r_[j].values, r_[j].missing);
        int &bit_index = bit_of_feature_[feature];
        bool repeated = bit_index >= 0;
        if (!repeated) {
            bit_index = static_cast<int>(bits);
            feature_of_bit_.push_back(feature);
        }
        Pattern bit = Pattern(1) << bit_index;
        std::size_t child_bits = repeated ? bits : bits + 1;
        for (bool to_left : {true, false}) {
            Descend(Foreground(depth), x_left, to_left, bit, repeated, Foreground(depth + 1), data_.num_X);
            Descend(Background(depth), r_left, to_left, bit, repeated, Background(depth + 1), data_.num_R);
            Visit(to_left ? tree_->children_left[node] : tree_->children_right[node], depth + 1, child_bits);
        }
        if (!repeated) {
            bit_index = -1;
            feature_of_bit_.pop_back();
        }
    }

    // Give each distinct pattern a slot, recording the slot of every row in slots.
    std::size_t Unique(const Pattern *patterns, std::size_t rows, std::vector<Pattern> &unique,
                       std::vector<double> *counts, std::int32_t *slots) {
        unique.clear();
        if (counts) counts->clear();
        for (std::size_t i = 0; i < rows; ++i) {
            std::int32_t &slot = slot_[patterns[i]];
            if (slot < 0) {
                slot = static_cast<std::int32_t>(unique.size());
                unique.push_back(patterns[i]);
                if (counts) counts->push_back(0);
            }
            if (counts) (*counts)[slot] += 1;
            if (slots) slots[i] = slot;
        }
        for (auto pattern : unique) slot_[pattern] = -1;
        return unique.size();
    }

    void Leaf(unsigned node, std::size_t depth, std::size_t bits) {
        const double *values = tree_->values + node * NumOutputs();
        const Pattern full = static_cast<Pattern>((std::uint64_t(1) << bits) - 1);
        Unique(Background(depth), data_.num_R, r_unique_, &r_counts_, nullptr);
        Unique(Foreground(depth), data_.num_X, x_unique_, nullptr, row_slot_.data());

        // The baseline leaf is reached by references satisfying the whole path.
        double baseline = 0;
        for (std::size_t b = 0; b < r_unique_.size(); ++b) {
            if (r_unique_[b] == full) baseline = r_counts_[b];
        }

        // For a foreground pattern c and a reference pattern b, each feature on the
        // path takes one of four roles from its bits c_d and b_d:
        //   1. c_d = 1, b_d = 0: foreground player, gaining the pair's positive share.
        //   2. c_d = 0, b_d = 1: background player, gaining the pair's negative share.
        //   3. c_d = 1, b_d = 1: dummy, since either source reaches the leaf.
        //   4. c_d = 0, b_d = 0: the leaf is unreachable, so the pair is skipped.
        // For each foreground pattern, sum every feature's shares over the references
        // that reach the leaf.
        shares_.assign(x_unique_.size() * bits, 0);
        positive_by_bit_.resize(bits);
        for (std::size_t c = 0; c < x_unique_.size(); ++c) {
            Pattern x_pattern = x_unique_[c];
            std::size_t negatives = bits - Popcount(x_pattern);
            double positive = 0, negative = 0;
            std::fill(positive_by_bit_.begin(), positive_by_bit_.end(), 0);
            for (std::size_t b = 0; b < r_unique_.size(); ++b) {
                Pattern r_pattern = r_unique_[b];
                // Rule 4: some feature has c_d = 0 and b_d = 0.
                if ((x_pattern | r_pattern) != full) continue;
                const auto &share = coefficients_(bits - Popcount(r_pattern), negatives);
                double positive_share = r_counts_[b] * share.positive;
                positive += positive_share;
                negative += r_counts_[b] * share.negative;
                // Rule 3: foreground bits that this reference also satisfies are dummies for it.
                for (Pattern both = x_pattern & r_pattern; both; both &= both - 1) {
                    positive_by_bit_[Ctz(both)] += positive_share;
                }
            }
            // Rules 1 and 2. Every reaching reference has b_d = 1 where c_d = 0.
            double *share = shares_.data() + c * bits;
            for (std::size_t d = 0; d < bits; ++d) {
                share[d] = x_pattern >> d & 1 ? positive - positive_by_bit_[d] : negative;
            }
        }

        for (std::size_t i = 0; i < data_.num_X; ++i) {
            double *out = phi_ + i * Stride();
            const double *share = shares_.data() + row_slot_[i] * bits;
            for (std::size_t d = 0; d < bits; ++d) {
                if (share[d] == 0) continue;
                double *value = out + feature_of_bit_[d] * NumOutputs();
                for (std::size_t o = 0; o < NumOutputs(); ++o) value[o] += values[o] * share[d];
            }
            double *bias = out + data_.M * NumOutputs();
            for (std::size_t o = 0; o < NumOutputs(); ++o) bias[o] += values[o] * baseline;
        }
    }

    static std::size_t Popcount(Pattern pattern) {
        std::size_t count = 0;
        for (; pattern; pattern &= pattern - 1) ++count;
        return count;
    }

    static std::size_t Ctz(Pattern pattern) {
        std::size_t index = 0;
        for (; !(pattern & 1); pattern >>= 1) ++index;
        return index;
    }

    const Coefficients &coefficients_;
    const ExplanationDataset &data_;
    std::size_t outputs_;
    const TreeEnsemble *tree_ = nullptr;
    double *phi_ = nullptr;
    std::vector<Row> x_, r_;
    std::vector<int> bit_of_feature_;
    std::vector<unsigned> feature_of_bit_;
    std::vector<Pattern> foreground_, background_;
    std::vector<std::uint8_t> x_left_, r_left_;
    std::vector<std::int32_t> row_slot_, slot_;
    std::vector<Pattern> x_unique_, r_unique_;
    std::vector<double> r_counts_, shares_, positive_by_bit_;
};

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
    std::vector<double> margin_x(outputs), margins_r(transform ? data.num_R * outputs : 0);
    auto margins = [&](Row row, double *margin) {
        std::copy_n(trees.base_offset, outputs, margin);
        for (unsigned t = 0; t < trees.tree_limit; ++t) {
            const double *values = tree_predict(t, trees, row.values, row.missing);
            std::transform(margin, margin + outputs, values, margin, std::plus<>());
        }
    };
    // Background margins do not depend on the foreground row, so predict them once.
    if (transform) {
        for (std::size_t j = 0; j < data.num_R; ++j) margins(BackgroundRow(data, j), margins_r.data() + j * outputs);
    }

    for (std::size_t i = 0; i < data.num_X; ++i) {
        double *out = phi + i * stride;
        const Row x = ForegroundRow(data, i);
        const double y = data.y ? data.y[i] : 0;
        if (transform) margins(x, margin_x.data());

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
                const double *margin_r = margins_r.data() + j * outputs;
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

// Benchmark builds may force one algorithm: 1 for traversal, 2 for patterns.
#ifndef SHAP_INTERVENTIONAL_FORCE
#define SHAP_INTERVENTIONAL_FORCE 0
#endif

// Whether the pattern algorithm should explain a tree with this many leaves. The
// traversal costs about N R t per tree for N explained and R background rows, while
// patterns cost about N a + R b, so patterns win when a / R + b / N < t. Measured on
// four datasets, depths 3-16, N 10-5,000 and R 5-1,000, both a and b grow with the
// square root of the leaf count. Fitted constants were stable under leaving out any
// one dataset, and stayed within 2.0-2.3x worst case over nearby values.
inline bool PreferPatterns(std::size_t leaves, const ExplanationDataset &data) {
    if (SHAP_INTERVENTIONAL_FORCE) return SHAP_INTERVENTIONAL_FORCE == 2;
    double n = data.num_X, r = data.num_R;
    return std::sqrt(static_cast<double>(leaves)) * (1.5 * n + 0.5 * r) <= n * r;
}

template <bool SingleOutput>
inline void ExplainPatterns(const TreeEnsemble &trees, const std::vector<TreeEnsemble> &views,
                            const ExplanationDataset &data, double *phi) {
    std::size_t depth = 0;
    for (const auto &view : views) depth = std::max(depth, Depth(view, 0));
    // Patterns have at most kMaxPatternBits players per leaf, however deep the tree.
    Coefficients coefficients(std::min<std::size_t>(kMaxPatternBits, data.M));
    Patterns<SingleOutput> patterns(coefficients, data, trees.num_outputs, depth);
    for (const auto &view : views) patterns.Run(view, phi);
}

// Matches dense_independent's output layout, transforms and reference averaging,
// computing every output in a single traversal per row/reference pair.
inline void Explain(const TreeEnsemble &trees, const ExplanationDataset &data, double *phi,
                    transform_f transform) {
    // Patterns sum each tree over references; the traversal then averages all trees.
    std::vector<TreeEnsemble> traversal, patterns;
    std::vector<unsigned> uses(data.M, 0);
    for (unsigned t = 0; t < trees.tree_limit; ++t) {
        TreeEnsemble view;
        trees.get_tree(view, t);
        bool use_patterns = !transform && PathFeatures(view, 0, uses) <= kMaxPatternBits &&
                            PreferPatterns(Leaves(view, 0), data);
        (use_patterns ? patterns : traversal).push_back(view);
    }
    if (trees.num_outputs == 1) {
        if (!patterns.empty()) ExplainPatterns<true>(trees, patterns, data, phi);
        ExplainRows<true>(trees, traversal, data, phi, transform);
    } else {
        if (!patterns.empty()) ExplainPatterns<false>(trees, patterns, data, phi);
        ExplainRows<false>(trees, traversal, data, phi, transform);
    }
}
}  // namespace shap_interventional

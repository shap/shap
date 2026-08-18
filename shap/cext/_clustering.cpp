#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cstdint>
#include <random>

namespace nb = nanobind;
using namespace nb::literals;

// Thread-local random engine for _pt_shuffle_rec (replaces np.random.randn)
static thread_local std::mt19937 rng{std::random_device{}()};

// --------------------------------------------------------------------------
// 1. _mask_delta_score  —  equivalent to: (m1 ^ m2).sum()
//
// BUG-FIX: The numba XOR (^) on int64 produces the full bitwise XOR value,
//          but then .sum() adds up those XOR values (not just 0/1 per element).
//          In the Python code, the masks only contain 0 or 1, so XOR is always
//          0 or 1 and the sum is a count.  We keep that semantic here.
// --------------------------------------------------------------------------
int64_t mask_delta_score(
    nb::ndarray<int64_t, nb::ndim<1>, nb::c_contig> m1,
    nb::ndarray<int64_t, nb::ndim<1>, nb::c_contig> m2)
{
    const int64_t* p1 = m1.data();
    const int64_t* p2 = m2.data();
    size_t len = m1.shape(0);
    int64_t score = 0;
    for (size_t i = 0; i < len; ++i) {
        score += (p1[i] ^ p2[i]);         // identical to numba: (m1^m2).sum()
    }
    return score;
}

// --------------------------------------------------------------------------
// 2. _reverse_window  —  in-place sub-array reversal
// --------------------------------------------------------------------------
void reverse_window(
    nb::ndarray<int64_t, nb::ndim<1>, nb::c_contig> order,
    int64_t start,
    int64_t length)
{
    int64_t* data = order.data();
    for (int64_t i = 0; i < length / 2; ++i) {
        int64_t tmp = data[start + i];
        data[start + i] = data[start + length - i - 1];
        data[start + length - i - 1] = tmp;
    }
}

// --------------------------------------------------------------------------
// Helper: compute _mask_delta_score for two rows inside a 2-D mask matrix.
//         Avoids creating temporary numpy slices.
// --------------------------------------------------------------------------
static int64_t mask_delta_score_2d(
    const int64_t* data,
    size_t cols,
    int64_t row1,
    int64_t row2)
{
    const int64_t* p1 = data + row1 * static_cast<int64_t>(cols);
    const int64_t* p2 = data + row2 * static_cast<int64_t>(cols);
    int64_t score = 0;
    for (size_t i = 0; i < cols; ++i) {
        score += (p1[i] ^ p2[i]);
    }
    return score;
}

// --------------------------------------------------------------------------
// 3. _reverse_window_score_gain
//
// Python original (lines 107-114 of _clustering.py):
//   forward  = delta(masks[order[start-1]], masks[order[start]])
//            + delta(masks[order[start+length-1]], masks[order[start+length]])
//   reverse  = delta(masks[order[start-1]], masks[order[start+length-1]])
//            + delta(masks[order[start]], masks[order[start+length]])
//   return forward - reverse
// --------------------------------------------------------------------------
int64_t reverse_window_score_gain(
    nb::ndarray<int64_t, nb::ndim<2>, nb::c_contig> masks,
    nb::ndarray<int64_t, nb::ndim<1>, nb::c_contig> order,
    int64_t start,
    int64_t length)
{
    const int64_t* m = masks.data();
    const int64_t* o = order.data();
    size_t cols = masks.shape(1);

    int64_t forward_score =
        mask_delta_score_2d(m, cols, o[start - 1], o[start]) +
        mask_delta_score_2d(m, cols, o[start + length - 1], o[start + length]);

    int64_t reverse_score =
        mask_delta_score_2d(m, cols, o[start - 1], o[start + length - 1]) +
        mask_delta_score_2d(m, cols, o[start], o[start + length]);

    return forward_score - reverse_score;
}

// --------------------------------------------------------------------------
// 4. delta_minimization_order
//
// Python original (lines 78-89):
//   order = np.arange(len(all_masks))
//   for _ in range(num_passes):
//       for length in range(2, max_swap_size):
//           for i in range(1, len(order) - length):
//               if _reverse_window_score_gain(...) > 0:
//                   _reverse_window(order, i, length)
//   return order
//
// BUG-FIX: We must cap max_swap_size to (num_masks - 1) so we never
//          read past the end of the order array.
// --------------------------------------------------------------------------
nb::ndarray<nb::numpy, int64_t, nb::ndim<1>> delta_minimization_order(
    nb::ndarray<int64_t, nb::ndim<2>, nb::c_contig> all_masks,
    int64_t max_swap_size,
    int64_t num_passes)
{
    int64_t num_masks = static_cast<int64_t>(all_masks.shape(0));

    // Allocate output order array  (owned by Python via capsule)
    int64_t* order_data = new int64_t[num_masks];
    for (int64_t i = 0; i < num_masks; ++i) {
        order_data[i] = i;
    }

    nb::capsule owner(order_data, [](void* p) noexcept {
        delete[] static_cast<int64_t*>(p);
    });

    size_t shape[1] = {static_cast<size_t>(num_masks)};
    auto order = nb::ndarray<nb::numpy, int64_t, nb::ndim<1>>(order_data, 1, shape, owner);

    // Cap max_swap_size to stay within bounds
    int64_t effective_max = std::min(max_swap_size, num_masks);

    const int64_t* m = all_masks.data();
    size_t cols = all_masks.shape(1);
    int64_t* o = order_data;  // direct pointer for inner loop

    for (int64_t p = 0; p < num_passes; ++p) {
        for (int64_t length = 2; length < effective_max; ++length) {
            for (int64_t i = 1; i < num_masks - length; ++i) {
                // Inline score gain calculation for speed
                int64_t fwd =
                    mask_delta_score_2d(m, cols, o[i - 1], o[i]) +
                    mask_delta_score_2d(m, cols, o[i + length - 1], o[i + length]);
                int64_t rev =
                    mask_delta_score_2d(m, cols, o[i - 1], o[i + length - 1]) +
                    mask_delta_score_2d(m, cols, o[i], o[i + length]);

                if (fwd > rev) {
                    // Inline reverse_window for speed
                    for (int64_t j = 0; j < length / 2; ++j) {
                        int64_t tmp = o[i + j];
                        o[i + j] = o[i + length - j - 1];
                        o[i + length - j - 1] = tmp;
                    }
                }
            }
        }
    }

    return order;
}

// --------------------------------------------------------------------------
// 5. _pt_shuffle_rec
//
// Python original (lines 50-74):
//   Uses np.random.randn() < 0 to decide left-right vs right-left traversal.
//
// BUG-FIX vs old C++:
//   - Old code used rand()/RAND_MAX Box-Muller which is non-thread-safe,
//     low quality, and M_PI is not defined on MSVC.
//   - Now uses std::normal_distribution + thread_local std::mt19937.
//   - Recursive function must accept raw pointers to avoid nanobind
//     re-wrapping overhead on every recursive call.
// --------------------------------------------------------------------------
static int pt_shuffle_rec_impl(
    int i,
    int64_t* indexes,
    const bool* index_mask,
    const double* pt_data,
    size_t pt_cols,
    int M,
    int pos)
{
    if (i < 0) {
        if (index_mask[i + M]) {
            indexes[pos] = i + M;
            return pos + 1;
        }
        return pos;
    }

    int left  = static_cast<int>(pt_data[i * pt_cols + 0]) - M;
    int right = static_cast<int>(pt_data[i * pt_cols + 1]) - M;

    std::normal_distribution<double> dist(0.0, 1.0);
    if (dist(rng) < 0.0) {
        pos = pt_shuffle_rec_impl(left,  indexes, index_mask, pt_data, pt_cols, M, pos);
        pos = pt_shuffle_rec_impl(right, indexes, index_mask, pt_data, pt_cols, M, pos);
    } else {
        pos = pt_shuffle_rec_impl(right, indexes, index_mask, pt_data, pt_cols, M, pos);
        pos = pt_shuffle_rec_impl(left,  indexes, index_mask, pt_data, pt_cols, M, pos);
    }
    return pos;
}

// Python-facing wrapper that unpacks nanobind arrays into raw pointers once
int pt_shuffle_rec(
    int i,
    nb::ndarray<int64_t, nb::ndim<1>, nb::c_contig> indexes,
    nb::ndarray<bool, nb::ndim<1>, nb::c_contig> index_mask,
    nb::ndarray<double, nb::ndim<2>, nb::c_contig> partition_tree,
    int M,
    int pos)
{
    return pt_shuffle_rec_impl(
        i,
        indexes.data(),
        index_mask.data(),
        partition_tree.data(),
        partition_tree.shape(1),
        M,
        pos
    );
}

// --------------------------------------------------------------------------
// Module definition
// --------------------------------------------------------------------------
NB_MODULE(_clustering_cpp, m) {
    m.doc() = "C++ implementation of SHAP clustering utilities (nanobind)";

    m.def("_mask_delta_score", &mask_delta_score,
          "m1"_a, "m2"_a,
          "Compute the XOR delta score between two binary mask arrays.");

    m.def("_reverse_window", &reverse_window,
          "order"_a, "start"_a, "length"_a,
          "Reverse a sub-window of an order array in-place.");

    m.def("_reverse_window_score_gain", &reverse_window_score_gain,
          "masks"_a, "order"_a, "start"_a, "length"_a,
          "Compute the score gain from reversing a window in the mask ordering.");

    m.def("delta_minimization_order", &delta_minimization_order,
          "all_masks"_a, "max_swap_size"_a = 100, "num_passes"_a = 2,
          "Find an ordering of masks that minimizes total delta between adjacent masks.");

    m.def("_pt_shuffle_rec", &pt_shuffle_rec,
          "i"_a, "indexes"_a, "index_mask"_a, "partition_tree"_a, "M"_a, "pos"_a,
          "Recursively shuffle indexes according to a partition tree.");
}

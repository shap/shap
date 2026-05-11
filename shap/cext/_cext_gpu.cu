#include <Python.h>

#include "gpu_treeshap.h"
#include "tree_shap.h"

const float inf = std::numeric_limits<tfloat>::infinity();

struct ShapSplitCondition {
  ShapSplitCondition() = default;
  ShapSplitCondition(tfloat feature_lower_bound, tfloat feature_upper_bound,
                     bool is_missing_branch)
      : feature_lower_bound(feature_lower_bound),
        feature_upper_bound(feature_upper_bound),
        is_missing_branch(is_missing_branch) {
    assert(feature_lower_bound <= feature_upper_bound);
  }

  /*! Feature values >= lower and < upper flow down this path. */
  tfloat feature_lower_bound;
  tfloat feature_upper_bound;
  /*! Do missing values flow down this path? */
  bool is_missing_branch;

  // Does this instance flow down this path?
  __host__ __device__ bool EvaluateSplit(float x) const {
    // is nan
    if (isnan(x)) {
      return is_missing_branch;
    }
    return x > feature_lower_bound && x <= feature_upper_bound;
  }

  // Combine two split conditions on the same feature
  __host__ __device__ void
  Merge(const ShapSplitCondition &other) {  // Combine duplicate features
    feature_lower_bound = max(feature_lower_bound, other.feature_lower_bound);
    feature_upper_bound = min(feature_upper_bound, other.feature_upper_bound);
    is_missing_branch = is_missing_branch && other.is_missing_branch;
  }
};


// Inspired by: https://en.cppreference.com/w/cpp/iterator/size
// Limited implementation of std::size fo arrays
template <class T, size_t N>
constexpr size_t array_size(const T (&array)[N]) noexcept
{
    return N;
}

void RecurseTree(
    unsigned pos, const TreeEnsemble &tree,
    std::vector<gpu_treeshap::PathElement<ShapSplitCondition>> *tmp_path,
    std::vector<gpu_treeshap::PathElement<ShapSplitCondition>> *paths,
    size_t *path_idx, int num_outputs) {
  if (tree.is_leaf(pos)) {
    for (auto j = 0ull; j < num_outputs; j++) {
      auto v = tree.values[pos * num_outputs + j];
      if (v == 0.0) {
        // The tree has no output for this class, don't bother adding the path
        continue;
      }
      // Go back over path, setting v, path_idx
      for (auto &e : *tmp_path) {
        e.v = v;
        e.group = j;
        e.path_idx = *path_idx;
      }

      paths->insert(paths->end(), tmp_path->begin(), tmp_path->end());
      // Increment path index
      (*path_idx)++;
    }
    return;
  }

  // Add left split to the path
  unsigned left_child = tree.children_left[pos];
  double left_zero_fraction =
      tree.node_sample_weights[left_child] / tree.node_sample_weights[pos];
  // Encode the range of feature values that flow down this path
  tmp_path->emplace_back(0, tree.features[pos], 0,
                         ShapSplitCondition{-inf, tree.thresholds[pos], false},
                         left_zero_fraction, 0.0f);

  RecurseTree(left_child, tree, tmp_path, paths, path_idx, num_outputs);

  // Add left split to the path
  tmp_path->back() = gpu_treeshap::PathElement<ShapSplitCondition>(
      0, tree.features[pos], 0,
      ShapSplitCondition{tree.thresholds[pos], inf, false},
      1.0 - left_zero_fraction, 0.0f);

  RecurseTree(tree.children_right[pos], tree, tmp_path, paths, path_idx,
              num_outputs);

  tmp_path->pop_back();
}

std::vector<gpu_treeshap::PathElement<ShapSplitCondition>>
ExtractPaths(const TreeEnsemble &trees) {
  std::vector<gpu_treeshap::PathElement<ShapSplitCondition>> paths;
  size_t path_idx = 0;
  for (auto i = 0; i < trees.tree_limit; i++) {
    TreeEnsemble tree;
    trees.get_tree(tree, i);
    std::vector<gpu_treeshap::PathElement<ShapSplitCondition>> tmp_path;
    tmp_path.reserve(tree.max_depth);
    tmp_path.emplace_back(0, -1, 0, ShapSplitCondition{-inf, inf, false}, 1.0,
                          0.0f);
    RecurseTree(0, tree, &tmp_path, &paths, &path_idx, tree.num_outputs);
  }
  return paths;
}

class DeviceExplanationDataset {
  thrust::device_vector<tfloat> data;
  thrust::device_vector<bool> missing;
  size_t num_features;
  size_t num_rows;

 public:
  DeviceExplanationDataset(const ExplanationDataset &host_data,
                           bool background_dataset = false) {
    num_features = host_data.M;
    if (background_dataset) {
      num_rows = host_data.num_R;
      data = thrust::device_vector<tfloat>(
          host_data.R, host_data.R + host_data.num_R * host_data.M);
      missing = thrust::device_vector<bool>(host_data.R_missing,
                                            host_data.R_missing +
                                                host_data.num_R * host_data.M);

    } else {
      num_rows = host_data.num_X;
      data = thrust::device_vector<tfloat>(
          host_data.X, host_data.X + host_data.num_X * host_data.M);
      missing = thrust::device_vector<bool>(host_data.X_missing,
                                            host_data.X_missing +
                                                host_data.num_X * host_data.M);
    }
  }

  class DenseDatasetWrapper {
    const tfloat *data;
    const bool *missing;
    int num_rows;
    int num_cols;

   public:
    DenseDatasetWrapper() = default;
    DenseDatasetWrapper(const tfloat *data, const bool *missing, int num_rows,
                        int num_cols)
        : data(data), missing(missing), num_rows(num_rows), num_cols(num_cols) {
    }
    __device__ tfloat GetElement(size_t row_idx, size_t col_idx) const {
      auto idx = row_idx * num_cols + col_idx;
      if (missing[idx]) {
        return std::numeric_limits<tfloat>::quiet_NaN();
      }
      return data[idx];
    }
    __host__ __device__ size_t NumRows() const { return num_rows; }
    __host__ __device__ size_t NumCols() const { return num_cols; }
  };

  DenseDatasetWrapper GetDeviceAccessor() {
    return DenseDatasetWrapper(data.data().get(), missing.data().get(),
                               num_rows, num_features);
  }
};

inline void dense_tree_path_dependent_gpu(
    const TreeEnsemble &trees, const ExplanationDataset &data,
    tfloat *out_contribs, tfloat transform(const tfloat, const tfloat)) {
  auto paths = ExtractPaths(trees);
  DeviceExplanationDataset device_data(data);
  DeviceExplanationDataset::DenseDatasetWrapper X =
      device_data.GetDeviceAccessor();

  thrust::device_vector<float> phis((X.NumCols() + 1) * X.NumRows() *
                                    trees.num_outputs);
  gpu_treeshap::GPUTreeShap(X, paths.begin(), paths.end(), trees.num_outputs,
                            phis.begin(), phis.end());
  // Add the base offset term to bias
  thrust::device_vector<double> base_offset(
      trees.base_offset, trees.base_offset + trees.num_outputs);
  auto counting = thrust::make_counting_iterator(size_t(0));
  auto d_phis = phis.data().get();
  auto d_base_offset = base_offset.data().get();
  size_t num_groups = trees.num_outputs;
  thrust::for_each(counting, counting + X.NumRows() * trees.num_outputs,
                   [=] __device__(size_t idx) {
                     size_t row_idx = idx / num_groups;
                     size_t group = idx % num_groups;
                     auto phi_idx = gpu_treeshap::IndexPhi(
                         row_idx, num_groups, group, X.NumCols(), X.NumCols());
                     d_phis[phi_idx] += d_base_offset[group];
                   });

  // Shap uses a slightly different layout for multiclass
  thrust::device_vector<float> transposed_phis(phis.size());
  auto d_transposed_phis = transposed_phis.data();
  thrust::for_each(
      counting, counting + phis.size(), [=] __device__(size_t idx) {
        size_t old_shape[] = {X.NumRows(), num_groups, (X.NumCols() + 1)};
        size_t old_idx[array_size(old_shape)];
        gpu_treeshap::FlatIdxToTensorIdx(idx, old_shape, old_idx);
        // Define new tensor format, switch num_groups axis to end
        size_t new_shape[] = {X.NumRows(), (X.NumCols() + 1), num_groups};
        size_t new_idx[] = {old_idx[0], old_idx[2], old_idx[1]};
        size_t transposed_idx =
            gpu_treeshap::TensorIdxToFlatIdx(new_shape, new_idx);
        d_transposed_phis[transposed_idx] = d_phis[idx];
      });
  thrust::copy(transposed_phis.begin(), transposed_phis.end(), out_contribs);
}

inline void
dense_tree_independent_gpu(const TreeEnsemble &trees,
                           const ExplanationDataset &data, tfloat *out_contribs,
                           tfloat transform(const tfloat, const tfloat)) {
  auto paths = ExtractPaths(trees);
  DeviceExplanationDataset device_data(data);
  DeviceExplanationDataset::DenseDatasetWrapper X =
      device_data.GetDeviceAccessor();

  size_t M = X.NumCols();
  size_t num_X = X.NumRows();
  size_t num_groups = trees.num_outputs;
  size_t phi_count = (M + 1) * num_X * num_groups;
  auto counting = thrust::make_counting_iterator(size_t(0));
  thrust::device_vector<float> phis(phi_count);

  if (transform != NULL) {
    // Exact per-reference transform rescaling, matching the CPU algorithm in
    // dense_independent() (tree_shap.h). We call the GPU library once per
    // reference sample to get per-reference raw SHAP values, then apply the
    // chain-rule rescaling individually before averaging.
    std::vector<tfloat> accum(phi_count, 0.0);
    std::vector<float> h_phis(phi_count);

    // Upload all reference data to GPU once
    thrust::device_vector<tfloat> all_r_data(
        data.R, data.R + data.num_R * data.M);
    thrust::device_vector<bool> all_r_missing(
        data.R_missing, data.R_missing + data.num_R * data.M);

    // Precompute margins for all X samples
    std::vector<std::vector<tfloat>> margins_x(num_groups);
    for (unsigned oind = 0; oind < num_groups; ++oind) {
      margins_x[oind].resize(num_X);
      for (unsigned i = 0; i < num_X; ++i) {
        const tfloat *x = data.X + i * data.M;
        const bool *x_missing = data.X_missing + i * data.M;
        margins_x[oind][i] = trees.base_offset[oind];
        for (unsigned k = 0; k < trees.tree_limit; ++k) {
          margins_x[oind][i] += tree_predict(k, trees, x, x_missing)[oind];
        }
      }
    }

    for (unsigned j = 0; j < data.num_R; ++j) {
      // Create a 1-row background view into the already-uploaded R data
      DeviceExplanationDataset::DenseDatasetWrapper R_j(
          all_r_data.data().get() + j * data.M,
          all_r_missing.data().get() + j * data.M, 1, data.M);

      // Compute raw SHAP values for all X against this single reference
      thrust::fill(phis.begin(), phis.end(), 0.0f);
      gpu_treeshap::GPUTreeShapInterventional(
          X, R_j, paths.begin(), paths.end(), trees.num_outputs, phis.begin(),
          phis.end());

      // Copy to host
      thrust::copy(phis.begin(), phis.end(), h_phis.begin());

      // Compute margin for this reference
      const tfloat *r = data.R + j * data.M;
      const bool *r_missing = data.R_missing + j * data.M;
      std::vector<tfloat> margin_r(num_groups);
      for (unsigned oind = 0; oind < num_groups; ++oind) {
        margin_r[oind] = trees.base_offset[oind];
        for (unsigned k = 0; k < trees.tree_limit; ++k) {
          margin_r[oind] += tree_predict(k, trees, r, r_missing)[oind];
        }
      }

      // Apply per-reference rescaling and accumulate
      for (unsigned i = 0; i < num_X; ++i) {
        const tfloat y_i = data.y == NULL ? 0 : data.y[i];

        for (unsigned oind = 0; oind < num_groups; ++oind) {
          // Chain-rule rescale factor for this (x_i, r_j) pair
          tfloat rescale;
          if (margins_x[oind][i] != margin_r[oind]) {
            rescale = (transform(margins_x[oind][i], y_i) -
                       transform(margin_r[oind], y_i)) /
                      (margins_x[oind][i] - margin_r[oind]);
          } else {
            rescale = 1.0;
          }

          // Accumulate rescaled feature contributions
          for (unsigned k = 0; k < M; ++k) {
            auto phi_idx =
                gpu_treeshap::IndexPhi(i, num_groups, oind, M, k);
            accum[phi_idx] += h_phis[phi_idx] * rescale;
          }

          // Accumulate transformed bias (y=0 for bias, matching CPU)
          auto bias_idx =
              gpu_treeshap::IndexPhi(i, num_groups, oind, M, M);
          accum[bias_idx] +=
              transform(trees.base_offset[oind] + h_phis[bias_idx], 0);
        }
      }
    }

    // Average over references
    for (auto &v : accum) v /= data.num_R;

    // Copy result back to device for transpose
    std::vector<float> accum_f(accum.begin(), accum.end());
    thrust::copy(accum_f.begin(), accum_f.end(), phis.begin());
  } else {
    // No transform: single call with full background dataset
    DeviceExplanationDataset background_device_data(data, true);
    DeviceExplanationDataset::DenseDatasetWrapper R =
        background_device_data.GetDeviceAccessor();
    gpu_treeshap::GPUTreeShapInterventional(X, R, paths.begin(), paths.end(),
                                            trees.num_outputs, phis.begin(),
                                            phis.end());

    // Add base offset on device
    auto d_phis = phis.data().get();
    thrust::device_vector<double> base_offset(
        trees.base_offset, trees.base_offset + num_groups);
    auto d_base_offset = base_offset.data().get();
    thrust::for_each(counting, counting + num_X * num_groups,
                     [=] __device__(size_t idx) {
                       size_t row_idx = idx / num_groups;
                       size_t group = idx % num_groups;
                       auto phi_idx = gpu_treeshap::IndexPhi(
                           row_idx, num_groups, group, M, M);
                       d_phis[phi_idx] += d_base_offset[group];
                     });
  }

  // Shap uses a slightly different layout for multiclass
  auto d_phis = phis.data().get();
  thrust::device_vector<float> transposed_phis(phis.size());
  auto d_transposed_phis = transposed_phis.data();
  thrust::for_each(
      counting, counting + phis.size(), [=] __device__(size_t idx) {
        size_t old_shape[] = {num_X, num_groups, (M + 1)};
        size_t old_idx[array_size(old_shape)];
        gpu_treeshap::FlatIdxToTensorIdx(idx, old_shape, old_idx);
        // Define new tensor format, switch num_groups axis to end
        size_t new_shape[] = {num_X, (M + 1), num_groups};
        size_t new_idx[] = {old_idx[0], old_idx[2], old_idx[1]};
        size_t transposed_idx =
            gpu_treeshap::TensorIdxToFlatIdx(new_shape, new_idx);
        d_transposed_phis[transposed_idx] = d_phis[idx];
      });
  thrust::copy(transposed_phis.begin(), transposed_phis.end(), out_contribs);
}

inline void dense_tree_path_dependent_interactions_gpu(
    const TreeEnsemble &trees, const ExplanationDataset &data,
    tfloat *out_contribs, tfloat transform(const tfloat, const tfloat)) {
  auto paths = ExtractPaths(trees);
  DeviceExplanationDataset device_data(data);
  DeviceExplanationDataset::DenseDatasetWrapper X =
      device_data.GetDeviceAccessor();

  thrust::device_vector<float> phis((X.NumCols() + 1) * (X.NumCols() + 1) *
                                    X.NumRows() * trees.num_outputs);
  gpu_treeshap::GPUTreeShapInteractions(X, paths.begin(), paths.end(),
                                        trees.num_outputs, phis.begin(),
                                        phis.end());
  // Add the base offset term to bias
  thrust::device_vector<double> base_offset(
      trees.base_offset, trees.base_offset + trees.num_outputs);
  auto counting = thrust::make_counting_iterator(size_t(0));
  auto d_phis = phis.data().get();
  auto d_base_offset = base_offset.data().get();
  size_t num_groups = trees.num_outputs;
  thrust::for_each(counting, counting + X.NumRows() * num_groups,
                   [=] __device__(size_t idx) {
                     size_t row_idx = idx / num_groups;
                     size_t group = idx % num_groups;
                     auto phi_idx = gpu_treeshap::IndexPhiInteractions(
                         row_idx, num_groups, group, X.NumCols(), X.NumCols(),
                         X.NumCols());
                     d_phis[phi_idx] += d_base_offset[group];
                   });
  // Shap uses a slightly different layout for multiclass
  thrust::device_vector<float> transposed_phis(phis.size());
  auto d_transposed_phis = transposed_phis.data();
  thrust::for_each(
      counting, counting + phis.size(), [=] __device__(size_t idx) {
        size_t old_shape[] = {X.NumRows(), num_groups, (X.NumCols() + 1),
                              (X.NumCols() + 1)};
        size_t old_idx[array_size(old_shape)];
        gpu_treeshap::FlatIdxToTensorIdx(idx, old_shape, old_idx);
        // Define new tensor format, switch num_groups axis to end
        size_t new_shape[] = {X.NumRows(), (X.NumCols() + 1), (X.NumCols() + 1),
                              num_groups};
        size_t new_idx[] = {old_idx[0], old_idx[2], old_idx[3], old_idx[1]};
        size_t transposed_idx =
            gpu_treeshap::TensorIdxToFlatIdx(new_shape, new_idx);
        d_transposed_phis[transposed_idx] = d_phis[idx];
      });
  thrust::copy(transposed_phis.begin(), transposed_phis.end(), out_contribs);
}

void dense_tree_shap_gpu(const TreeEnsemble &trees,
                         const ExplanationDataset &data, tfloat *out_contribs,
                         const int feature_dependence, unsigned model_transform,
                         bool interactions) {
  // Check for categorical features
  bool has_categorical = false;
  for (unsigned i = 0; i < trees.tree_limit * trees.max_nodes; i++) {
    if (trees.threshold_types[i] != 0) {
      has_categorical = true;
      break;
    }
  }
  if (has_categorical) {
    std::cerr << "Warning: Categorical features detected. GPU TreeSHAP currently "
                 "only supports numerical features. Results may be incorrect.\n";
  }

  // see what transform (if any) we have
  transform_f transform = get_transform(model_transform);

  // dispatch to the correct algorithm handler
  switch (feature_dependence) {
  case FEATURE_DEPENDENCE::independent:
    if (interactions) {
      std::cerr << "FEATURE_DEPENDENCE::independent with interactions not yet "
                   "supported\n";
    } else {
      dense_tree_independent_gpu(trees, data, out_contribs, transform);
    }
    return;

  case FEATURE_DEPENDENCE::tree_path_dependent:
    if (interactions) {
      dense_tree_path_dependent_interactions_gpu(trees, data, out_contribs,
                                                 transform);
    } else {
      dense_tree_path_dependent_gpu(trees, data, out_contribs, transform);
    }
    return;

  case FEATURE_DEPENDENCE::global_path_dependent:
    std::cerr << "FEATURE_DEPENDENCE::global_path_dependent not supported\n";
    return;
  default:
    std::cerr << "Unknown feature dependence option\n";
    return;
  }
}

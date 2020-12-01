#include <Python.h>

#include "gpu_treeshap.h"
#include "tree_shap.h"

const float inf = std::numeric_limits<float>::infinity();

void RecurseTree(unsigned pos, const TreeEnsemble &tree,
                 std::vector<gpu_treeshap::PathElement> *tmp_path,
                 std::vector<gpu_treeshap::PathElement> *paths,
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
  tmp_path->emplace_back(0, tree.features[pos], 0, -inf,
                         static_cast<float>(tree.thresholds[pos]), false,
                         left_zero_fraction, 0.0f);

  RecurseTree(left_child, tree, tmp_path, paths, path_idx, num_outputs);

  // Add left split to the path
  tmp_path->back() = gpu_treeshap::PathElement(
      0, tree.features[pos], 0, static_cast<float>(tree.thresholds[pos]), inf,
      false, 1.0 - left_zero_fraction, 0.0f);

  RecurseTree(tree.children_right[pos], tree, tmp_path, paths, path_idx,
              num_outputs);

  tmp_path->pop_back();
}

std::vector<gpu_treeshap::PathElement> ExtractPaths(const TreeEnsemble &trees) {
  std::vector<gpu_treeshap::PathElement> paths;
  size_t path_idx = 0;
  for (auto i = 0; i < trees.tree_limit; i++) {
    TreeEnsemble tree;
    trees.get_tree(tree, i);
    std::vector<gpu_treeshap::PathElement> tmp_path;
    tmp_path.reserve(tree.max_depth);
    tmp_path.emplace_back(0, -1, 0, -inf, inf, false, 1.0, 0.0f);
    RecurseTree(0, tree, &tmp_path, &paths, &path_idx, tree.num_outputs);
  }
  return paths;
}

class DeviceExplanationDataset {
  thrust::device_vector<float> data;
  thrust::device_vector<bool> missing;
  size_t num_features;
  size_t num_rows;

public:
  DeviceExplanationDataset(const ExplanationDataset &host_data,
                           bool background_dataset = false) {

    num_features = host_data.M;
    if (background_dataset) {
      num_rows = host_data.num_R;
      data = thrust::device_vector<float>(
          host_data.R, host_data.R + host_data.num_R * host_data.M);
      missing = thrust::device_vector<float>(host_data.R_missing,
                                             host_data.R_missing +
                                                 host_data.num_R * host_data.M);

    } else {
      num_rows = host_data.num_X;
      data = thrust::device_vector<float>(
          host_data.X, host_data.X + host_data.num_X * host_data.M);
      missing = thrust::device_vector<float>(host_data.X_missing,
                                             host_data.X_missing +
                                                 host_data.num_X * host_data.M);
    }
  }

  class DenseDatasetWrapper {
    const float *data;
    const bool *missing;
    int num_rows;
    int num_cols;

  public:
    DenseDatasetWrapper() = default;
    DenseDatasetWrapper(const float *data, const bool *missing, int num_rows,
                        int num_cols)
        : data(data), missing(missing), num_rows(num_rows), num_cols(num_cols) {
    }
    __device__ float GetElement(size_t row_idx, size_t col_idx) const {
      auto idx = row_idx * num_cols + col_idx;
      if (missing[idx]) {
        return std::numeric_limits<float>::quiet_NaN();
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

// Helper function that accepts an index into a flat contiguous array and the
// dimensions of a tensor and returns the indices with respect to the tensor
template <typename T, size_t N>
__device__ void flat_idx_to_tensor_idx(T flat_idx, const T (&shape)[N],
                                       T (&out_idx)[N]) {
  T current_size = shape[0];
  for (auto i = 1ull; i < N; i++) {
    current_size *= shape[i];
  }
  for (auto i = 0ull; i < N; i++) {
    current_size /= shape[i];
    out_idx[i] = flat_idx / current_size;
    flat_idx -= current_size * out_idx[i];
  }
}

// Given a shape and coordinates into a tensor, return the index into the
// backing storage one-dimensional array
template <typename T, size_t N>
__device__ T tensor_idx_to_flat_idx(const T (&shape)[N],
                                    const T (&tensor_idx)[N]) {
  T current_size = shape[0];
  for (auto i = 1ull; i < N; i++) {
    current_size *= shape[i];
  }
  T idx = 0;
  for (auto i = 0ull; i < N; i++) {
    current_size /= shape[i];
    idx += tensor_idx[i] * current_size;
  }
  return idx;
}

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
                            phis.data().get(), phis.size());
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
        size_t old_idx[std::size(old_shape)];
        flat_idx_to_tensor_idx(idx, old_shape, old_idx);
        // Define new tensor format, switch num_groups axis to end
        size_t new_shape[] = {X.NumRows(), (X.NumCols() + 1), num_groups};
        size_t new_idx[] = {old_idx[0], old_idx[2], old_idx[1]};
        size_t transposed_idx = tensor_idx_to_flat_idx(new_shape, new_idx);
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
  DeviceExplanationDataset background_device_data(data, true);
  DeviceExplanationDataset::DenseDatasetWrapper R =
      background_device_data.GetDeviceAccessor();

  thrust::device_vector<float> phis((X.NumCols() + 1) * X.NumRows() *
                                    trees.num_outputs);
  gpu_treeshap::GPUTreeShapInterventional(X, R, paths.begin(), paths.end(),
                                          trees.num_outputs, phis.data().get(),
                                          phis.size());
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
        size_t old_idx[std::size(old_shape)];
        flat_idx_to_tensor_idx(idx, old_shape, old_idx);
        // Define new tensor format, switch num_groups axis to end
        size_t new_shape[] = {X.NumRows(), (X.NumCols() + 1), num_groups};
        size_t new_idx[] = {old_idx[0], old_idx[2], old_idx[1]};
        size_t transposed_idx = tensor_idx_to_flat_idx(new_shape, new_idx);
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
                                        trees.num_outputs, phis.data().get(),
                                        phis.size());
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
        size_t old_idx[std::size(old_shape)];
        flat_idx_to_tensor_idx(idx, old_shape, old_idx);
        // Define new tensor format, switch num_groups axis to end
        size_t new_shape[] = {X.NumRows(), (X.NumCols() + 1), (X.NumCols() + 1),
                              num_groups};
        size_t new_idx[] = {old_idx[0], old_idx[2], old_idx[3], old_idx[1]};
        size_t transposed_idx = tensor_idx_to_flat_idx(new_shape, new_idx);
        d_transposed_phis[transposed_idx] = d_phis[idx];
      });
  thrust::copy(transposed_phis.begin(), transposed_phis.end(), out_contribs);
}

void dense_tree_shap_gpu(const TreeEnsemble &trees,
                         const ExplanationDataset &data, tfloat *out_contribs,
                         const int feature_dependence, unsigned model_transform,
                         bool interactions) {
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

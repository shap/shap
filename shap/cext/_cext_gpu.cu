#include <Python.h>

#include "gpu_treeshap.h"
#include "tree_shap.h"

const float inf = std::numeric_limits<float>::infinity();

void RecurseTree(unsigned pos, const TreeEnsemble &tree,
                 std::vector<gpu_treeshap::PathElement> *tmp_path,
                 std::vector<gpu_treeshap::PathElement> *paths,
                 size_t *path_idx) {
  if (tree.is_leaf(pos)) {
    auto v = tree.values[pos];
    // Go back over path, setting v, path_idx
    for (auto &e : *tmp_path) {
      e.v = v;
      e.path_idx = *path_idx;
    }

    paths->insert(paths->end(), tmp_path->begin(), tmp_path->end());
    // Increment path index
    (*path_idx)++;
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

  RecurseTree(left_child, tree, tmp_path, paths, path_idx);

  // Add left split to the path
  tmp_path->back() = gpu_treeshap::PathElement(
      0, tree.features[pos], 0, static_cast<float>(tree.thresholds[pos]), inf,
      false, 1.0 - left_zero_fraction, 0.0f);

  RecurseTree(tree.children_right[pos], tree, tmp_path, paths, path_idx);

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
    RecurseTree(0, tree, &tmp_path, &paths, &path_idx);
  }
  return paths;
}

class DeviceExplanationDataset {
  thrust::device_vector<float> data;
  thrust::device_vector<bool> missing;
  size_t num_features;
  size_t num_rows;

 public:
  DeviceExplanationDataset(const ExplanationDataset &host_data)
      : data(host_data.X, host_data.X + host_data.num_X * host_data.M),
        missing(host_data.X_missing,
                host_data.X_missing + host_data.num_X * host_data.M),
        num_features(host_data.M),
        num_rows(host_data.num_X) {}

  class DenseDatasetWrapper {
    const float *data;
    const bool *missing;
    int num_rows;
    int num_cols;

   public:
    DenseDatasetWrapper() = default;
    DenseDatasetWrapper(const float *data, const bool *missing, int num_rows,
                        int num_cols)
        : data(data),
          missing(missing),
          num_rows(num_rows),
          num_cols(num_cols) {}
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

inline void dense_tree_path_dependent_gpu(
    const TreeEnsemble &trees, const ExplanationDataset &data,
    tfloat *out_contribs, tfloat transform(const tfloat, const tfloat)) {
  auto paths = ExtractPaths(trees);
  DeviceExplanationDataset device_data(data);
  DeviceExplanationDataset::DenseDatasetWrapper X =
      device_data.GetDeviceAccessor();

  thrust::device_vector<float> phis((X.NumCols() + 1) * X.NumRows());
  gpu_treeshap::GPUTreeShap(X, paths.begin(), paths.end(), 1, phis.data().get(),
                            phis.size());
  thrust::copy(phis.begin(), phis.end(), out_contribs);
  //      for(auto e:paths){
  //      std::cout << e.path_idx<<"\n";
  //      std::cout << e.feature_idx<<"\n";
  //      std::cout << e.feature_lower_bound<<"\n";
  //      std::cout << e.feature_upper_bound<<"\n";
  //      std::cout << e.v<<"\n";
  //      std::cout << "\n";
  //      }
}

void dense_tree_shap_gpu(const TreeEnsemble &trees,
                         const ExplanationDataset &data, tfloat *out_contribs,
                         const int feature_dependence, unsigned model_transform,
                         bool interactions) {
  // see what transform (if any) we have
  transform_f transform = get_transform(model_transform);

  if (interactions) {
    std::cerr << "Interactions not yet supported\n";
    return;
  }

  // dispatch to the correct algorithm handler
  switch (feature_dependence) {
    case FEATURE_DEPENDENCE::independent:
      std::cerr << "FEATURE_DEPENDENCE::independent not yet supported\n";
      return;

    case FEATURE_DEPENDENCE::tree_path_dependent:
      if (interactions) {
        std::cerr << "Interactions not yet supported\n";
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

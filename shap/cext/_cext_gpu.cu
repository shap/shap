#include <Python.h>

#include "gpu_treeshap.h"
#include "tree_shap.h"


std::vector<gpu_treeshap::PathElement> ExtractPaths(const TreeEnsemble& trees) {
  std::vector<gpu_treeshap::PathElement> paths;
  size_t path_idx = 0;
  TreeEnsemble tree;
  trees.get_tree(tree, 0);
  return paths;
}

inline void dense_tree_path_dependent_gpu(const TreeEnsemble& trees, const ExplanationDataset &data,
                               tfloat *out_contribs, tfloat transform(const tfloat, const tfloat)) {
printf("Hello world\n");
auto paths=ExtractPaths(trees);
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
                  if (interactions){
                std::cerr << "Interactions not yet supported\n";
                  } else{
      dense_tree_path_dependent_gpu(trees, data, out_contribs,
                  transform);}
      return;

    case FEATURE_DEPENDENCE::global_path_dependent:
      std::cerr << "FEATURE_DEPENDENCE::global_path_dependent not supported\n";
      return;
    default:
      std::cerr << "Unknown feature dependence option\n";
      return;
  }
}

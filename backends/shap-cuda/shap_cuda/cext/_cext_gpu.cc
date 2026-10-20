#include "tree_shap_bindings.h"

void dense_tree_shap_gpu(const TreeEnsemble &trees, const ExplanationDataset &data,
                         tfloat *out_contribs, int feature_dependence,
                         unsigned model_transform, bool interactions);

NB_MODULE(_cext_gpu, m) {
    m.doc() = "Bindings for the GPU Tree SHAP implementation.";
    shap_bindings::bind_dense_tree_shap(m, &dense_tree_shap_gpu);
}

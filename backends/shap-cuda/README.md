# shap-cuda

CUDA-accelerated `GPUTreeExplainer` backend for [shap](https://github.com/shap/shap).

Installing this package registers `GPUTreeExplainer` with `shap` via the
`shap.tree_backends` entry-point group; use it through `shap.GPUTreeExplainer`,
not by importing `shap_cuda` directly.

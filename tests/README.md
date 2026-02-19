# SHAP Tests

## GPU TreeExplainer Tests

The GPU TreeExplainer tests require a CUDA-capable GPU and the `_cext_gpu` extension
built from source. Since GPU hardware is not available in our CI pipeline, these tests
can be run manually on Google Colab.

### Running on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shap/shap/blob/gpu-tree-tests-colab/tests/gpu_tree_tests.ipynb)

1. Click the badge above to open the notebook in Google Colab
2. Switch to a GPU runtime: **Runtime > Change runtime type > T4 GPU**
3. Run all cells

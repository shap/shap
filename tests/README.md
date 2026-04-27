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

---

### Running GPU tests locally

In addition to Google Colab, GPU TreeExplainer tests can also be run locally on a system with a CUDA-capable GPU.

#### Requirements
- CUDA toolkit installed
- Compatible GPU drivers
- SHAP built with GPU support (`_cext_gpu`)

#### Steps

1. Install dependencies:

```bash
pip install -r requirements.txt

2. Build SHAP with GPU support:

```bash
pip install .

3. Run GPU tests:

```bash
pytest tests --gpu

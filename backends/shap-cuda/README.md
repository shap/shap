# shap-cuda

CUDA-accelerated `GPUTreeExplainer` backend for [shap](https://github.com/shap/shap).

Installing this package registers `GPUTreeExplainer` with `shap` via the
`shap.tree_backends` entry-point group; use it through `shap.GPUTreeExplainer`,
not by importing `shap_cuda` directly.

## Build locally from source

Build both `shap` and `shap-cuda` from the same checkout: the backend uses SHAP
internals and checks that their release versions match. The commands below are
for Linux and run from the repository root, not this backend directory.

### Prerequisites

- Python 3.12 or newer; Python 3.13 is used in the examples below.
- A C++ compiler and Python development headers (on Ubuntu/Debian, install
  `build-essential` and the development package for your Python version).
- The NVIDIA CUDA toolkit, including `nvcc`. CI builds with CUDA 12.8; the
  backend currently targets GPU architectures `60`, `70`, `75`, and `80`, so
  use a toolkit that supports those targets.
- An NVIDIA GPU and a compatible NVIDIA driver to run explanations. Building
  the extension does not require a physical GPU.

Check that the toolkit and driver are available:

```bash
nvcc --version
nvidia-smi
```

If the toolkit is installed outside your `PATH`, point the build at it. Adjust
this example to your installation:

```bash
export PATH=/usr/local/cuda-12.8/bin:$PATH
export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
```

The build backend installs CMake (3.24 or newer) and its other Python build
dependencies in an isolated environment as needed. The separate CUDA backend
does not require the old `SHAP_ENABLE_CUDA` environment variable.

### Install with uv

```bash
uv venv --python 3.13
source .venv/bin/activate
uv sync --extra cuda
```

The repository's uv workspace resolves the `cuda` extra to the local
`backends/shap-cuda` package and builds both packages as editable installs.

### Install with pip

Alternatively, create and activate a virtual environment, then explicitly
install both local packages:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -v -e . -e ./backends/shap-cuda
```

Using `pip install -e '.[cuda]'` alone can resolve `shap-cuda` from a package
index instead of this checkout; pip does not use uv's workspace configuration.

### Verify and use the backend

Confirm that the compiled extension loads and that SHAP discovers the backend:

```bash
python -c "import shap; import shap_cuda._cext_gpu; print(shap.GPUTreeExplainer)"
```

Importing the extension directly here is a build check. For explanations, use
the public SHAP API:

```python
import numpy as np
import shap
from sklearn.tree import DecisionTreeRegressor

X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
model = DecisionTreeRegressor(max_depth=2).fit(X, X[:, 0] + X[:, 1])
explainer = shap.GPUTreeExplainer(model)
values = explainer.shap_values(X)
print(values)
```

This example also checks that GPU execution works. Importing the package alone
does not prove that the CUDA extension was built. In verbose build output,
look for `CUDA toolkit found, building the _cext_gpu extension.` If the
extension is missing or the build cannot find `nvcc`, check `CUDACXX` and the
toolkit installation, then rebuild:

```bash
# uv
uv sync --extra cuda --reinstall-package shap-cuda

# Or pip; --no-deps keeps the existing local SHAP installation.
python -m pip install -v --no-cache-dir --force-reinstall --no-deps -e ./backends/shap-cuda
```

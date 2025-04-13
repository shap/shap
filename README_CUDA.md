# SHAP CUDA Support

This document explains how to build SHAP with CUDA support using the meson-python build system.

## Prerequisites

To build SHAP with CUDA support, you need:

1. CUDA Toolkit (recommended: CUDA 11.0 or newer)
2. Python 3.9 or newer
3. A compatible C++ compiler
4. Python development packages

## Installation

### Automatic Installation (Recommended)

The easiest way to install SHAP with CUDA support is to use the provided installation script:

```bash
python install.py
```

This script will:
1. Check if CUDA is available on your system
2. Try to build SHAP with CUDA support
3. If CUDA build fails, fall back to CPU-only build

### Manual Installation

You can also control the build manually:

#### With CUDA support (default):

```bash
pip install -e .
```

#### Without CUDA support:

```bash
pip install -e . --config-settings=options=-Dwith_cuda=false
```

#### Without C/C++ extensions:

```bash
pip install -e . --config-settings=options="-Dwith_cuda=false -Dwith_binary=false"
```

## Building Wheels

To build wheels for distribution, use the provided wheel building script:

```bash
python build_wheels.py
```

This script will:
1. Build a CPU-only wheel first (always works)
2. Check if CUDA is available, and if so, build a CUDA-enabled wheel
3. Place both wheels in the `dist/` directory

For official distribution, you'll need to:
- Build CPU wheels on all platforms
- Build CUDA wheels on platforms with CUDA available
- Upload the wheels to PyPI

## Installing from Wheels

When installing from wheels, the CUDA support is determined at build time:

```bash
# Install from a local wheel (CPU version)
pip install dist/shap-x.y.z-py3-none-any.whl

# Install from PyPI
pip install shap
```

## Verifying CUDA Support

To check if SHAP was built with CUDA support:

```python
import shap
print("CUDA available:", hasattr(shap, "_cext_gpu"))
```

## Troubleshooting

### Common Issues:

1. **NVCC not found**:
   Make sure CUDA is in your PATH or set CUDA_PATH environment variable.

2. **Compilation errors**:
   Ensure your C++ compiler is compatible with your CUDA version.

3. **Missing dependencies**:
   Install required packages with `pip install numpy cython`.

4. **Build fails with CUDA enabled**:
   Try building with --config-settings=options=-Dwith_cuda=false

### Debugging Build Issues:

For more verbose output during build:

```bash
MESONPY_VERBOSE=1 pip install -e .
```

## CUDA Version Compatibility

SHAP has been tested with:
- CUDA 10.0 and newer
- Compute capabilities 6.0 through 8.0

If you need support for older GPUs, modify the meson.build file to include appropriate architecture flags.
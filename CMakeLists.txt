# Followed https://nanobind.readthedocs.io/en/latest/building.html
cmake_minimum_required(VERSION 3.15...3.27)
project(shap_extensions LANGUAGES CXX)

if (NOT SKBUILD)
  message(FATAL_ERROR "\
  This CMakeLists.txt is meant to be used with scikit-build.
  Please use 'python -m pip install .', not 'cmake'.")
endif()

# Find the Python interpreter and development components.
if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.12
  COMPONENTS Interpreter ${DEV_MODULE} REQUIRED
  OPTIONAL_COMPONENTS Development.SABIModule)

# Perform an optimized release build unless otherwise specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# ==============================================================================
# Build the nanobind extension module (cutils)
# ==============================================================================

# Detect the installed nanobind package and import it into CMake
find_package(nanobind CONFIG REQUIRED)

# Create the cutils module using nanobind
nanobind_add_module(
  _cutils

  # This extension is free-threaded
  FREE_THREADED

  # Target the stable ABI of Python 3.12+, reducing the number of binary wheels
  STABLE_ABI

  # Build libnanobind as a static library and link it into the module
  NB_STATIC

  # Sources for the cutils module
  shap/cutils/cutils.cpp
)

nanobind_add_stub(
  _cutils_stub
  MODULE _cutils
  OUTPUT _cutils.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:_cutils>
  MARKER_FILE py.typed
  DEPENDS _cutils
)

# Keep generated typing artifacts visible to language servers during local/editable
# development. VS Code/Pylance resolves this workspace package from source.
add_custom_command(
  TARGET _cutils_stub POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/_cutils.pyi
    ${CMAKE_CURRENT_SOURCE_DIR}/shap/_cutils.pyi
)

# Install the module to shap
install(TARGETS _cutils LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})

# ==============================================================================
# Build the Tree Logic extension module (cext)
# ==============================================================================

# Check if the SABI version is being requested
if(NOT "${SKBUILD_SABI_VERSION}" STREQUAL "")
  set(USE_SABI USE_SABI ${SKBUILD_SABI_VERSION})
endif()

python_add_library(
  _cext MODULE

  # Target the stable ABI of Python 3.12+, reducing the number of binary wheels
  ${USE_SABI} WITH_SOABI

  # Sources for the cext module
  shap/cext/_cext.cc
)

# Get the include directory for NumPy and add it to the include path
execute_process(
    COMMAND "${PYTHON_EXECUTABLE}"
    -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

target_include_directories(_cext PUBLIC ${NUMPY_INCLUDE_DIR})

# Install directive for scikit-build-core
install(TARGETS _cext LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})

# ==============================================================================
# Optional: Build the GPU Tree SHAP extension module (cext_gpu)
# Enabled via: SHAP_ENABLE_CUDA=1 pip install .
# ==============================================================================

if(DEFINED ENV{SHAP_ENABLE_CUDA})
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  python_add_library(
    _cext_gpu MODULE

    ${USE_SABI} WITH_SOABI

    # The .cu file is compiled by nvcc, the .cc file by the host compiler
    shap/cext/_cext_gpu.cu
    shap/cext/_cext_gpu.cc
  )

  target_include_directories(_cext_gpu PUBLIC ${NUMPY_INCLUDE_DIR})
  target_link_libraries(_cext_gpu PRIVATE CUDA::cudart)

  set_target_properties(_cext_gpu PROPERTIES
    CUDA_ARCHITECTURES "60;70;75;80"
    CUDA_STANDARD 14
    CUDA_EXTENSIONS ON
  )

  # nvcc flags matching the previous setup.py build
  target_compile_options(_cext_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      --expt-extended-lambda
      --expt-relaxed-constexpr
    >
  )

  install(TARGETS _cext_gpu LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})
endif()

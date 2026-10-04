cmake_minimum_required(VERSION 3.24...3.27)
project(shap_cuda_extensions LANGUAGES CXX)

if (NOT SKBUILD)
  message(FATAL_ERROR "\
  This CMakeLists.txt is meant to be used with scikit-build.
  Please use 'python -m pip install .', not 'cmake'.")
endif()

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.12
  COMPONENTS Interpreter ${DEV_MODULE} REQUIRED
  OPTIONAL_COMPONENTS Development.SABIModule)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Optional: CUDA may not be present (e.g. a docs build). In that case skip
# _cext_gpu and install the pure-Python package only - GPUTreeExplainer's own
# try/except around importing it degrades to a clear runtime error instead of
# making the whole install fail.
enable_language(CUDA OPTIONAL)

if(CMAKE_CUDA_COMPILER)
  find_package(CUDAToolkit)
endif()

if(CMAKE_CUDA_COMPILER AND CUDAToolkit_FOUND)
  message(STATUS "CUDA toolkit found, building the _cext_gpu extension.")

  if(NOT "${SKBUILD_SABI_VERSION}" STREQUAL "")
    set(USE_SABI USE_SABI ${SKBUILD_SABI_VERSION})
  endif()

  python_add_library(
    _cext_gpu MODULE

    ${USE_SABI} WITH_SOABI

    # The .cu file is compiled by nvcc, the .cc file by the host compiler
    shap_cuda/cext/_cext_gpu.cu
    shap_cuda/cext/_cext_gpu.cc
  )

  execute_process(
      COMMAND "${PYTHON_EXECUTABLE}"
      -c "import numpy; print(numpy.get_include())"
      OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  target_include_directories(_cext_gpu PUBLIC ${NUMPY_INCLUDE_DIR})
  target_link_libraries(_cext_gpu PRIVATE CUDA::cudart)

  set_target_properties(_cext_gpu PROPERTIES
    CUDA_ARCHITECTURES "60;70;75;80"
    CUDA_STANDARD 14
    CUDA_EXTENSIONS ON
  )

  target_compile_options(_cext_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      --expt-extended-lambda
      --expt-relaxed-constexpr
    >
  )

  install(TARGETS _cext_gpu LIBRARY DESTINATION shap_cuda)
else()
  message(STATUS "CUDA toolkit not found, skipping the _cext_gpu extension. "
                  "shap-cuda will install as Python-only; GPUTreeExplainer will "
                  "raise a clear error when actually used.")
endif()

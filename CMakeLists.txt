# Followed https://nanobind.readthedocs.io/en/latest/building.html
cmake_minimum_required(VERSION 3.15...3.27)
project(cutils LANGUAGES CXX)

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
  cutils

  # This extension is free-threaded
  FREE_THREADED

  # Target the stable ABI of Python 3.12+, reducing the number of binary wheels
  STABLE_ABI

  # Build libnanobind as a static library and link it into the module
  NB_STATIC

  # Sources for the cutils module
  shap/cutils/cutils.cpp
)

# Install the module to shap
install(TARGETS cutils LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})

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

# Followed https://nanobind.readthedocs.io/en/latest/building.html
cmake_minimum_required(VERSION 3.15...3.27)
project(shap_extensions LANGUAGES CXX)

if(EMSCRIPTEN)
  # nanobind uses stderr without including its declaration in this build.
  add_compile_options(-include cstdio)
endif()

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

# Stub generation imports the extension and cannot run on the cross-build host.
if(NOT EMSCRIPTEN)
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
endif()

install(TARGETS _cutils LIBRARY DESTINATION shap)

# ==============================================================================
# Build the Tree Logic extension module (cext)
# ==============================================================================

nanobind_add_module(
  _cext
  STABLE_ABI
  NB_STATIC
  shap/cext/_cext.cc
)

install(TARGETS _cext LIBRARY DESTINATION shap)

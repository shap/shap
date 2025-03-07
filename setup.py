import os
import platform
import subprocess
import sys
import sysconfig

import numpy as np
from packaging.version import Version, parse
from setuptools import Extension, setup

_BUILD_ATTEMPTS = 0

# This is copied from @robbuckley's fix for Panda's
# For mac, ensure extensions are built for macos 10.9 when compiling on a
# 10.9 system or above, overriding distuitls behavior which is to target
# the version that python was built for. This may be overridden by setting
# MACOSX_DEPLOYMENT_TARGET before calling setup.pcuda-comp-generalizey
if sys.platform == "darwin":
    if "MACOSX_DEPLOYMENT_TARGET" not in os.environ:
        current_system: Version = parse(platform.mac_ver()[0])
        python_target: Version = parse(sysconfig.get_config_var("MACOSX_DEPLOYMENT_TARGET"))
        if python_target < Version("10.9") and current_system >= Version("10.9"):
            os.environ["MACOSX_DEPLOYMENT_TARGET"] = "10.9"


def find_in_path(name, path):
    """Find a file in a search path and return its full path."""
    # adapted from:
    # http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = os.path.join(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def get_cuda_path():
    """Return a tuple with (base_cuda_directory, full_path_to_nvcc_compiler)."""
    # Inspired by https://github.com/benfred/implicit/blob/master/cuda_setup.py
    nvcc_bin = "nvcc.exe" if sys.platform == "win32" else "nvcc"

    if "CUDAHOME" in os.environ:
        cuda_home = os.environ["CUDAHOME"]
    elif "CUDA_PATH" in os.environ:
        cuda_home = os.environ["CUDA_PATH"]
    else:
        # otherwise, search the PATH for NVCC
        found_nvcc = find_in_path(nvcc_bin, os.environ["PATH"])
        if found_nvcc is None:
            print(
                "The nvcc binary could not be located in your $PATH. Either "
                "add it to your path, or set $CUDAHOME to enable CUDA.",
            )
            return None
        cuda_home = os.path.dirname(os.path.dirname(found_nvcc))
    if not os.path.exists(os.path.join(cuda_home, "include")):
        print("Failed to find cuda include directory, using /usr/local/cuda")
        cuda_home = "/usr/local/cuda"

    nvcc = os.path.join(cuda_home, "bin", nvcc_bin)
    if not os.path.exists(nvcc):
        print(f"Failed to find nvcc compiler in {nvcc}, trying /usr/local/cuda")
        cuda_home = "/usr/local/cuda"
        nvcc = os.path.join(cuda_home, "bin", nvcc_bin)

    return cuda_home, nvcc


def compile_cuda_module(host_args):
    libname = "_cext_gpu.lib" if sys.platform == "win32" else "lib_cext_gpu.a"
    lib_out = "build/" + libname
    if not os.path.exists("build/"):
        os.makedirs("build/")

    _, nvcc = get_cuda_path()

    print("NVCC ==> ", nvcc)
    arch_flags = (
        "-gencode=arch=compute_60,code=sm_60 "
        "-gencode=arch=compute_70,code=sm_70 "
        "-gencode=arch=compute_75,code=sm_75 "
        "-gencode=arch=compute_75,code=compute_75 "
        "-gencode=arch=compute_80,code=sm_80"
    )
    nvcc_command = (
        f"-allow-unsupported-compiler shap/cext/_cext_gpu.cu -lib -o {lib_out} "
        f"-Xcompiler {','.join(host_args)} "
        f"--include-path {sysconfig.get_path('include')} "
        "--std c++14 "
        "--expt-extended-lambda "
        f"--expt-relaxed-constexpr {arch_flags}"
    )
    print("Compiling cuda extension, calling nvcc with arguments:")
    print([nvcc] + nvcc_command.split(" "))
    subprocess.run([nvcc] + nvcc_command.split(" "), check=True)
    return "build", "_cext_gpu"


def run_setup(*, with_binary, with_cuda):
    ext_modules = []
    if with_binary:
        compile_args = []
        if sys.platform == "zos":
            compile_args.append("-qlonglong")
        if sys.platform == "win32":
            compile_args.append("/MD")

        ext_modules.append(
            Extension(
                "shap._cext",
                sources=["shap/cext/_cext.cc"],
                include_dirs=[np.get_include()],
                extra_compile_args=compile_args,
            )
        )
    if with_cuda:
        try:
            cuda_home, _ = get_cuda_path()
            if sys.platform == "win32":
                cudart_path = cuda_home + "/lib/x64"
            else:
                cudart_path = cuda_home + "/lib64"
                compile_args.append("-fPIC")

            lib_dir, lib = compile_cuda_module(compile_args)

            ext_modules.append(
                Extension(
                    "shap._cext_gpu",
                    sources=["shap/cext/_cext_gpu.cc"],
                    extra_compile_args=compile_args,
                    include_dirs=[np.get_include()],
                    library_dirs=[lib_dir, cudart_path],
                    libraries=[lib, "cudart"],
                    depends=["shap/cext/_cext_gpu.cu", "shap/cext/gpu_treeshap.h", "setup.py"],
                )
            )
        except Exception as e:
            raise Exception("Error building cuda module: " + repr(e)) from e

    ext_modules.append(
        Extension("_kernel_lib", sources=["shap/explainers/_kernel_lib.pyx"], include_dirs=[np.get_include()])
    )

    setup(ext_modules=ext_modules)


def try_run_setup(*, with_binary, with_cuda):
    """Fails gracefully when various install steps don't work."""
    global _BUILD_ATTEMPTS
    _BUILD_ATTEMPTS += 1

    try:
        print(f"Attempting to build SHAP: {with_binary=}, {with_cuda=} (Attempt {_BUILD_ATTEMPTS})")
        run_setup(with_binary=with_binary, with_cuda=with_cuda)
    except Exception as e:
        print("Exception occurred during setup,", str(e))

        if with_cuda:
            with_cuda = False
            print("WARNING: Could not compile cuda extensions.")
            print("Retrying SHAP build without cuda extension...")
        elif with_binary:
            with_binary = False
            print("WARNING: The C extension could not be compiled, sklearn tree models not supported.")
            print("Retrying SHAP build without binary extension...")
        else:
            print("ERROR: Failed to build!")
            raise

        try_run_setup(with_binary=with_binary, with_cuda=with_cuda)


# we seem to need this import guard for appveyor
if __name__ == "__main__":
    try_run_setup(with_binary=True, with_cuda=True)

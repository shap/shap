from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
import os
import re
import codecs
import platform
from distutils.sysconfig import get_config_var, get_python_inc
from distutils.version import LooseVersion
import sys
import subprocess

# to publish use:
# > python setup.py sdist bdist_wheel upload
# which depends on ~/.pypirc


# This is copied from @robbuckley's fix for Panda's
# For mac, ensure extensions are built for macos 10.9 when compiling on a
# 10.9 system or above, overriding distuitls behavior which is to target
# the version that python was built for. This may be overridden by setting
# MACOSX_DEPLOYMENT_TARGET before calling setup.pcuda-comp-generalizey
if sys.platform == 'darwin':
    if 'MACOSX_DEPLOYMENT_TARGET' not in os.environ:
        current_system = LooseVersion(platform.mac_ver()[0])
        python_target = LooseVersion(get_config_var('MACOSX_DEPLOYMENT_TARGET'))
        if python_target < '10.9' and current_system >= '10.9':
            os.environ['MACOSX_DEPLOYMENT_TARGET'] = '10.9'

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


# Extend the default build_ext class to bootstrap numpy installation
# that are needed to build C extensions.
# see https://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        if isinstance(__builtins__, dict):
            __builtins__["__NUMPY_SETUP__"] = False
        else:
            setattr(__builtins__, "__NUMPY_SETUP__", False)
        import numpy
        print("numpy.get_include()", numpy.get_include())
        self.include_dirs.append(numpy.get_include())


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
                "The nvcc binary could not be located in your $PATH. Either " +
                " add it to your path, or set $CUDAHOME to enable CUDA"
            )
            return None
        cuda_home = os.path.dirname(os.path.dirname(found_nvcc))
    if not os.path.exists(os.path.join(cuda_home, "include")):
        print("Failed to find cuda include directory, using /usr/local/cuda")
        cuda_home = "/usr/local/cuda"

    nvcc = os.path.join(cuda_home, "bin", nvcc_bin)
    if not os.path.exists(nvcc):
        print("Failed to find nvcc compiler in %s, trying /usr/local/cuda" % nvcc)
        cuda_home = "/usr/local/cuda"
        nvcc = os.path.join(cuda_home, "bin", nvcc_bin)

    return (cuda_home, nvcc)


def compile_cuda_module(host_args):
    libname = '_cext_gpu.lib' if sys.platform == 'win32' else 'lib_cext_gpu.a'
    lib_out = 'build/' + libname
    if not os.path.exists('build/'):
        os.makedirs('build/')

    cuda_home, nvcc = get_cuda_path()

    print("NVCC ==> ", nvcc)
    arch_flags = "-arch=sm_60 " + \
                 "-gencode=arch=compute_70,code=sm_70 " + \
                 "-gencode=arch=compute_75,code=sm_75 " + \
                 "-gencode=arch=compute_75,code=compute_75"
    nvcc_command = "shap/cext/_cext_gpu.cu -lib -o {} -Xcompiler {} -I{} " \
                   "--std c++14 " \
                   "--expt-extended-lambda " \
                   "--expt-relaxed-constexpr {}".format(
                       lib_out,
                       ','.join(host_args),
                       get_python_inc(), arch_flags)
    print("Compiling cuda extension, calling nvcc with arguments:")
    print([nvcc] + nvcc_command.split(' '))
    subprocess.run([nvcc] + nvcc_command.split(' '), check=True)
    return 'build', '_cext_gpu'


def run_setup(with_binary, test_xgboost, test_lightgbm, test_catboost, test_spark, test_pyod,
              with_cuda, test_transformers, test_pytorch, test_sentencepiece, test_opencv):
    ext_modules = []
    if with_binary:
        compile_args = []
        if sys.platform == 'zos':
            compile_args.append('-qlonglong')
        if sys.platform == 'win32':
            compile_args.append('/MD')

        ext_modules.append(
            Extension('shap._cext', sources=['shap/cext/_cext.cc'],
                      extra_compile_args=compile_args))
    if with_cuda:
        try:
            cuda_home, nvcc = get_cuda_path()
            if sys.platform == 'win32':
                cudart_path = cuda_home + '/lib/x64'
            else:
                cudart_path = cuda_home + '/lib64'
                compile_args.append('-fPIC')

            lib_dir, lib = compile_cuda_module(compile_args)

            ext_modules.append(
                Extension('shap._cext_gpu', sources=['shap/cext/_cext_gpu.cc'],
                          extra_compile_args=compile_args,
                          library_dirs=[lib_dir, cudart_path],
                          libraries=[lib, 'cudart'],
                          depends=['shap/cext/_cext_gpu.cu', 'shap/cext/gpu_treeshap.h','setup.py'])
            )
        except Exception as e:
            raise Exception("Error building cuda module: " + repr(e))

    tests_require = ['pytest', 'pytest-mpl', 'pytest-cov']
    if test_xgboost:
        tests_require += ['xgboost']
    if test_lightgbm:
        tests_require += ['lightgbm']
    if test_catboost:
        tests_require += ['catboost']
    if test_spark:
        tests_require += ['pyspark']
    if test_pyod:
        tests_require += ['pyod']
    if test_transformers:
        tests_require += ['transformers']
    if test_pytorch:
        tests_require += ['torch']
    if test_sentencepiece:
        tests_require += ['sentencepiece']
    if test_opencv:
        tests_require += ['opencv-python']

    extras_require = {
        'plots': [
            'matplotlib',
            'ipython'
        ],
        'others': [
            'lime',
        ],
        'docs': [
            'matplotlib',
            'ipython',
            'numpydoc',
            'sphinx_rtd_theme',
            'sphinx',
            'nbsphinx',
        ]
    }
    extras_require['test'] = tests_require
    extras_require['all'] = list(set(i for val in extras_require.values() for i in val))

    setup(
        name='shap',
        version=find_version("shap", "__init__.py"),
        description='A unified approach to explain the output of any machine learning model.',
        long_description="SHAP (SHapley Additive exPlanations) is a unified approach to explain "
                         "the output of " + \
                         "any machine learning model. SHAP connects game theory with local "
                         "explanations, uniting " + \
                         "several previous methods and representing the only possible consistent "
                         "and locally accurate " + \
                         "additive feature attribution method based on expectations.",
        long_description_content_type="text/markdown",
        url='http://github.com/slundberg/shap',
        author='Scott Lundberg',
        author_email='slund1@cs.washington.edu',
        license='MIT',
        packages=[
            'shap', 'shap.explainers', 'shap.explainers.other', 'shap.explainers._deep',
            'shap.plots', 'shap.plots.colors', 'shap.benchmark', 'shap.maskers', 'shap.utils',
            'shap.actions', 'shap.models'
        ],
        package_data={'shap': ['plots/resources/*', 'cext/tree_shap.h']},
        cmdclass={'build_ext': build_ext},
        setup_requires=['oldest-supported-numpy'],
        install_requires=['numpy', 'scipy', 'scikit-learn', 'pandas', 'tqdm>4.25.0',
                          'slicer==0.0.7', 'numba', 'cloudpickle'],
        extras_require=extras_require,
        ext_modules=ext_modules,
        classifiers=[
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
        ],
        zip_safe=False
        # python_requires='>3.0' we will add this at some point
    )


def try_run_setup(**kwargs):
    """ Fails gracefully when various install steps don't work.
    """

    try:
        run_setup(**kwargs)
    except Exception as e:
        print(str(e))
        if "xgboost" in str(e).lower():
            kwargs["test_xgboost"] = False
            print("Couldn't install XGBoost for testing!")
            try_run_setup(**kwargs)
        elif "lightgbm" in str(e).lower():
            kwargs["test_lightgbm"] = False
            print("Couldn't install LightGBM for testing!")
            try_run_setup(**kwargs)
        elif "catboost" in str(e).lower():
            kwargs["test_catboost"] = False
            print("Couldn't install CatBoost for testing!")
            try_run_setup(**kwargs)
        elif "cuda" in str(e).lower():
            kwargs["with_cuda"] = False
            print("WARNING: Could not compile cuda extensions")
            try_run_setup(**kwargs)
        elif kwargs["with_binary"]:
            kwargs["with_binary"] = False
            print("WARNING: The C extension could not be compiled, sklearn tree models not supported.")
            try_run_setup(**kwargs)
        elif "pyod" in str(e).lower():
            kwargs["test_pyod"] = False
            print("Couldn't install PyOD for testing!")
            try_run_setup(**kwargs)
        elif "transformers" in str(e).lower():
            kwargs["test_transformers"] = False
            print("Couldn't install Transformers for testing!")
            try_run_setup(**kwargs)
        elif "torch" in str(e).lower():
            kwargs["test_pytorch"] = False
            print("Couldn't install PyTorch for testing!")
            try_run_setup(**kwargs)
        elif "sentencepiece" in str(e).lower():
            kwargs["test_sentencepiece"] = False
            print("Couldn't install sentencepiece for testing!")
            try_run_setup(**kwargs)
        else:
            print("ERROR: Failed to build!")


# we seem to need this import guard for appveyor
if __name__ == "__main__":
    try_run_setup(
        with_binary=True, test_xgboost=True, test_lightgbm=True, test_catboost=True,
        test_spark=True, test_pyod=True, with_cuda=True, test_transformers=True, test_pytorch=True,
        test_sentencepiece=True, test_opencv=True
    )

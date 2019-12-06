from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
import os
import re
import codecs
import platform
from distutils.sysconfig import get_config_var
from distutils.version import LooseVersion
import sys

# to publish use:
# > python setup.py sdist bdist_wheel upload
# which depends on ~/.pypirc


# This is copied from @robbuckley's fix for Panda's
# For mac, ensure extensions are built for macos 10.9 when compiling on a
# 10.9 system or above, overriding distuitls behavior which is to target
# the version that python was built for. This may be overridden by setting
# MACOSX_DEPLOYMENT_TARGET before calling setup.py
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


def run_setup(with_binary=True, test_xgboost=True, test_lightgbm=True, test_catboost=True, test_spark=True):
    ext_modules = []
    if with_binary:
        ext_modules.append(
            Extension('shap._cext', sources=['shap/_cext.cc'])
        )

    tests_require = ['nose']
    if test_xgboost:
        tests_require += ['xgboost']
    if test_lightgbm:
        tests_require += ['lightgbm']
    if test_catboost:
        tests_require += ['catboost']
    if test_spark:
        tests_require += ['pyspark']

    extras_require = {
        'plots': [
            'matplotlib',
            'ipython'
        ],
        'others': [
            'lime',
        ],
    }
    extras_require['all'] = list(set(i for val in extras_require.values() for i in val))

    setup(
        name='shap',
        version=find_version("shap", "__init__.py"),
        description='A unified approach to explain the output of any machine learning model.',
        long_description="SHAP (SHapley Additive exPlanations) is a unified approach to explain the output of " + \
                         "any machine learning model. SHAP connects game theory with local explanations, uniting " + \
                         "several previous methods and representing the only possible consistent and locally accurate " + \
                         "additive feature attribution method based on expectations.",
        long_description_content_type="text/markdown",
        url='http://github.com/slundberg/shap',
        author='Scott Lundberg',
        author_email='slund1@cs.washington.edu',
        license='MIT',
        packages=[
            'shap', 'shap.explainers', 'shap.explainers.other', 'shap.explainers.deep',
            'shap.plots', 'shap.benchmark'
        ],
        package_data={'shap': ['plots/resources/*', 'tree_shap.h']},
        cmdclass={'build_ext': build_ext},
        setup_requires=['numpy'],
        install_requires=['numpy', 'scipy', 'scikit-learn', 'pandas', 'tqdm>4.25.0'],
        extras_require=extras_require,
        test_suite='nose.collector',
        tests_require=tests_require,
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
        elif kwargs["with_binary"]:
            kwargs["with_binary"] = False
            print("WARNING: The C extension could not be compiled, sklearn tree models not supported.")
            try_run_setup(**kwargs)
        else:
            print("ERROR: Failed to build!")

# we seem to need this import guard for appveyor
if __name__ == "__main__":
    try_run_setup(with_binary=True, test_xgboost=True, test_lightgbm=True, test_spark=True)

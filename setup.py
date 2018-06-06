from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext

# to publish use:
# > python setup.py sdist upload
# which depends on ~/.pypirc

# Extend the default build_ext class to bootstrap numpy installation
# that are needed to build C extensions.
# see https://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        print("numpy.get_include()", numpy.get_include())
        self.include_dirs.append(numpy.get_include())


def run_setup(with_binary):
    ext_modules = []
    if with_binary:
        ext_modules.append(
            Extension('shap._cext', sources=['shap/_cext.cc'])
        )

    setup(
        name='shap',
        version='0.17.2',
        description='A unified approach to explain the output of any machine learning model.',
        url='http://github.com/slundberg/shap',
        author='Scott Lundberg',
        author_email='slund1@cs.washington.edu',
        license='MIT',
        packages=['shap', 'shap.explainers'],
        cmdclass={'build_ext': build_ext},
        setup_requires=['numpy'],
        install_requires=['numpy', 'scipy', 'iml>=0.6.0', 'scikit-learn', 'matplotlib', 'pandas', 'tqdm'],
        test_suite='nose.collector',
        tests_require=['nose', 'xgboost'], # , 'lightgbm'
        ext_modules = ext_modules,
        zip_safe=False
    )

try:
    run_setup(True)
except Exception as e:
    print(e)
    print("WARNING: The C extension could not be compiled, sklearn tree models not supported.")
    run_setup(False)

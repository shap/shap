from setuptools import setup, Extension
import numpy

# to publish use:
# > python setup.py sdist upload
# which depends on ~/.pypirc

def run_setup(with_binary):
    ext_modules = []
    if with_binary:
        ext_modules.append(
            Extension('shap._cext', sources=['shap/_cext.cc'], include_dirs=[numpy.get_include()])
        )

    setup(
        name='shap',
        version='0.16.1',
        description='A unified approach to explain the output of any machine learning model.',
        url='http://github.com/slundberg/shap',
        author='Scott Lundberg',
        author_email='slund1@cs.washington.edu',
        license='MIT',
        packages=['shap', 'shap.explainers'],
        install_requires=['numpy', 'scipy', 'iml>=0.6.0', 'scikit-learn', 'matplotlib', 'pandas', 'tqdm'],
        test_suite='nose.collector',
        tests_require=['nose', 'xgboost'],
        ext_modules = ext_modules,
        zip_safe=False
    )

try:
    run_setup(True)
except:
    print("WARNING: The C extension could not be compiled, sklearn tree models not supported.")
    run_setup(False)

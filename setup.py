from setuptools import setup, Extension
import numpy

# to publish use:
# > python setup.py sdist upload
# which depends on ~/.pypirc

setup(
    name='shap',
    version='0.13.2',
    description='A unified approach to explain the output of any machine learning model.',
    url='http://github.com/slundberg/shap',
    author='Scott Lundberg',
    author_email='slund1@cs.washington.edu',
    license='MIT',
    packages=['shap', 'shap.explainers'],
    install_requires=['numpy', 'scipy', 'iml>=0.5.1', 'scikit-learn', 'matplotlib', 'pandas', 'tqdm'],
    test_suite='nose.collector',
    tests_require=['nose', 'xgboost'],
    ext_modules = [Extension('shap._cext', sources=['shap/_cext.cc'], include_dirs=[numpy.get_include()])],
    zip_safe=False
)

#! /bin/sh

# depends on ~/.pypirc
python setup.py sdist upload #-r https://pypi.python.org/pypi

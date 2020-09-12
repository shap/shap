.. SHAP documentation master file, created by
   sphinx-quickstart on Tue May 22 10:44:55 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the SHAP Documentation
---------------------------------

.. image:: artwork/shap_header.png
   :width: 600px
   :align: center

**SHAP (SHapley Additive exPlanations)** is a game theoretic approach to explain the output of
any machine learning model. It connects optimal credit allocation with local explanations
using the classic Shapley values from game theory and their related extensions (see 
`papers <https://github.com/slundberg/shap#citations>`_ for details and citations.

Install
=======

Shap can be installed from either `PyPI <https://pypi.org/project/shap>`_::

   pip install shap

or `conda-forge <https://anaconda.org/conda-forge/shap>`_::

   conda install -c conda-forge shap


Contents
========

.. toctree::
   :maxdepth: 2

   API Reference <api>
   Examples <examples>

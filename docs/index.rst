.. SHAP documentation master file, created by
   sphinx-quickstart on Tue May 22 10:44:55 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SHAP Community Fork
-------------------

This repository is a fork of Scott Lundberg's popular `shap <https://github.com/slundberg/shap>`_ library. Unfortunately, the original SHAP repository is not currently maintained. This fork attempts to fix SHAP's current issues and merge old PRs.

What has changed on this fork?
==============================

This fork primarily adds bug fixes and deprecation updates, to ensure that SHAP works with the latest versions of other libaries. 

Contributing
============

**New contributors are very welcome** so please feel free to get involved, for example by submitting PRs or opening issues!

We are eager to build a broad pool of maintainers, to avoid having a single person responsible for the entire repository. This repo adopts a *liberal contribution governance model*, where project decisions are based on a consensus seeking process. For more information, see `here <https://medium.com/the-node-js-collection/healthy-open-source-967fa8be7951>`_.

Installation
============

This fork is not yet available on PyPI or conda. Our goal is to merge the changes from this fork back into `slundberg/shap <https://github.com/slundberg/shap>`_. If you would like to use this fork, the currently supported installation method is:

.. code-block::

   pip install git+https://github.com/dsgibbons/shap.git

If we are unable to merge our changes back into `slundberg/shap <https://github.com/slundberg/shap>`_, we will create our own release on PyPI.

.. warning::
   From this point onward, the documentation is mostly copied from `slundberg/shap <https://github.com/slundberg/shap>`_. Assume that any links or installation instructions refer to the original project and not this fork.

Welcome to the SHAP documentation
---------------------------------

.. image:: artwork/shap_header.png
   :width: 600px
   :align: center

**SHAP (SHapley Additive exPlanations)** is a game theoretic approach to explain the output of
any machine learning model. It connects optimal credit allocation with local explanations
using the classic Shapley values from game theory and their related extensions (see 
`papers <https://github.com/slundberg/shap#citations>`_ for details and citations).

Install
=======

SHAP can be installed from either `PyPI <https://pypi.org/project/shap>`_ or 
`conda-forge <https://anaconda.org/conda-forge/shap>`_::

   pip install shap
   or
   conda install -c conda-forge shap


Contents
========

.. toctree::
   :maxdepth: 2

   Topical overviews <overviews>
   Tabular examples <tabular_examples>
   Text examples <text_examples>
   Image examples <image_examples>
   Genomic examples <genomic_examples>
   Benchmarks <benchmarks>
   API reference <api>
   API examples <api_examples>

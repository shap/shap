.. SHAP documentation master file, created by
   sphinx-quickstart on Tue May 22 10:44:55 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: artwork/shap_header.png
   :width: 600px
   :align: center


**SHAP (SHapley Additive exPlanations)** is a game theoretic approach to explain the output of
any machine learning model. It connects optimal credit allocation with local explanations
using the classic Shapley values from game theory and their related extensions (see 
[papers](https://github.com/slundberg/shap#citations) for details and citations.

.. toctree::
   :maxdepth: 2

Explainers
=====================

.. autoclass:: shap.TreeExplainer
   :members:

.. autoclass:: shap.GradientExplainer
   :members:

.. autoclass:: shap.DeepExplainer
   :members:

.. autoclass:: shap.KernelExplainer
   :members:

.. autoclass:: shap.SamplingExplainer
   :members:

.. autoclass:: shap.PartitionExplainer
   :members:
   

Plots
=====================

.. autofunction:: shap.summary_plot
.. autofunction:: shap.dependence_plot
.. autofunction:: shap.waterfall_plot
.. autofunction:: shap.force_plot
.. autofunction:: shap.image_plot
.. autofunction:: shap.decision_plot

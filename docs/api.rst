.. currentmodule:: shap

API Reference
=============

Core Explainers
---------------
.. autosummary::
    :toctree: generated/

    shap.Explainer
    shap.TreeExplainer
    shap.GradientExplainer
    shap.DeepExplainer
    shap.KernelExplainer
    shap.SamplingExplainer
    shap.PartitionExplainer
    shap.LinearExplainer
    shap.PermutationExplainer
    shap.AdditiveExplainer

Other Explainers
----------------
.. autosummary::
    :toctree: generated/

    shap.explainers.other.Coefficent
    shap.explainers.other.Random
    shap.explainers.other.LimeTabular
    shap.explainers.other.Maple
    shap.explainers.other.TreeMaple
    shap.explainers.other.TreeGain

.. _plot_api:

Plots
-----
For usage examples, see :ref:`Plotting Examples <plots_examples>`

.. autosummary::
    :toctree: generated/

    summary_plot
    decision_plot
    multioutput_decision_plot
    dependence_plot
    force_plot
    image_plot
    monitoring_plot
    embedding_plot
    partial_dependence_plot
    bar_plot
    waterfall_plot
    group_difference_plot
    text_plot

Datasets
--------
.. automodule:: shap.datasets
    :members:

.. currentmodule:: shap

API Reference
=============
This page contains the API reference for public objects and functions in SHAP.
There are also :ref:`example notebooks <api_examples>` available that demonstrate how
to use the API of each object/function.


.. _explanation_api:

Explanation
-----------
.. autosummary::
    :toctree: generated/

    shap.Explanation


.. _explainers_api:

explainers
----------
.. autosummary::
    :toctree: generated/

    shap.Explainer
    shap.explainers.Tree
    shap.explainers.GPUTree
    shap.explainers.Linear
    shap.explainers.Permutation
    shap.explainers.Partition
    shap.explainers.Sampling
    shap.explainers.Additive
    shap.explainers.other.Coefficent
    shap.explainers.other.Random
    shap.explainers.other.LimeTabular
    shap.explainers.other.Maple
    shap.explainers.other.TreeMaple
    shap.explainers.other.TreeGain


.. _plots_api:

plots
-----
.. autosummary::
    :toctree: generated/

    shap.plots.bar
    shap.plots.waterfall
    shap.plots.scatter
    shap.plots.heatmap
    shap.plots.force
    shap.plots.text
    shap.plots.image
    shap.plots.partial_dependence


.. _maskers_api:

maskers
-------
.. autosummary::
    :toctree: generated/

    shap.maskers.Masker
    shap.maskers.Independent
    shap.maskers.Partition
    shap.maskers.Text
    shap.maskers.Image


.. _models_api:

models
------
.. autosummary::
    :toctree: generated/

    shap.models.Model
    shap.models.TeacherForcingLogits


.. _utils_api:

utils
-----
.. autosummary::
    :toctree: generated/

    shap.utils.hclust
    shap.utils.sample
    shap.utils.shapley_coefficients
    shap.utils.MaskedModel


.. _datasets_api:

datasets
--------
.. autosummary::
    :toctree: generated/

    shap.datasets.adult
    shap.datasets.boston
    shap.datasets.communitiesandcrime
    shap.datasets.corrgroups60
    shap.datasets.diabetes
    shap.datasets.imagenet50
    shap.datasets.imdb
    shap.datasets.independentlinear60
    shap.datasets.iris
    shap.datasets.nhanesi

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
    shap.Cohorts


.. _explainers_api:

explainers
----------
.. autosummary::
    :toctree: generated/

    shap.Explainer
    shap.TreeExplainer
    shap.GPUTreeExplainer
    shap.LinearExplainer
    shap.PermutationExplainer
    shap.PartitionExplainer
    shap.SamplingExplainer
    shap.AdditiveExplainer
    shap.DeepExplainer
    shap.KernelExplainer
    shap.GradientExplainer
    shap.ExactExplainer
    shap.explainers.other.Coefficient
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
    shap.plots.decision
    shap.plots.embedding
    shap.plots.initjs
    shap.plots.group_difference
    shap.plots.image_to_text
    shap.plots.monitoring
    shap.plots.beeswarm
    shap.plots.violin


.. _maskers_api:

maskers
-------
.. autosummary::
    :toctree: generated/

    shap.maskers.Masker
    shap.maskers.Independent
    shap.maskers.Partition
    shap.maskers.Impute
    shap.maskers.Fixed
    shap.maskers.Composite
    shap.maskers.FixedComposite
    shap.maskers.OutputComposite
    shap.maskers.Text
    shap.maskers.Image


.. _models_api:

models
------
.. autosummary::
    :toctree: generated/

    shap.models.Model
    shap.models.TeacherForcing
    shap.models.TextGeneration
    shap.models.TopKLM
    shap.models.TransformersPipeline


.. _utils_api:

utils
-----
.. autosummary::
    :toctree: generated/

    shap.utils.hclust
    shap.utils.hclust_ordering
    shap.utils.partition_tree
    shap.utils.partition_tree_shuffle
    shap.utils.delta_minimization_order
    shap.utils.approximate_interactions
    shap.utils.potential_interactions
    shap.utils.sample
    shap.utils.shapley_coefficients
    shap.utils.convert_name
    shap.utils.OpChain
    shap.utils.show_progress
    shap.utils.MaskedModel
    shap.utils.make_masks


.. _datasets_api:

datasets
--------
.. autosummary::
    :toctree: generated/

    shap.datasets.a1a
    shap.datasets.adult
    shap.datasets.california
    shap.datasets.communitiesandcrime
    shap.datasets.corrgroups60
    shap.datasets.diabetes
    shap.datasets.imagenet50
    shap.datasets.imdb
    shap.datasets.independentlinear60
    shap.datasets.iris
    shap.datasets.linnerud
    shap.datasets.nhanesi
    shap.datasets.rank

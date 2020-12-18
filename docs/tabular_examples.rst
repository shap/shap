Tabular examples
----------------

These examples explain machine learning models applied to tabular data. They are all generated from Jupyter
notebooks `available on GitHub <https://github.com/slundberg/shap/tree/master/notebooks/tabular_examples>`_.


Tree-based models
==============
Examples demonstrating how to explain tree-based machine learning models.

.. toctree::
    :glob:
    :maxdepth: 1

    example_notebooks/tabular_examples/tree_based_models/*


Linear models
================
Examples demonstrating how to explain linear machine learning models.

.. toctree::
    :glob:
    :maxdepth: 1

    example_notebooks/tabular_examples/linear_models/*


Neural networks
================
Examples demonstrating how to explain machine learning models based on neural networks.

.. toctree::
    :glob:
    :maxdepth: 1

    example_notebooks/tabular_examples/neural_network_model/*


Model agnostic
================
Examples demonstrating how to explain arbitrary machine learning pipelines.

.. toctree::
    :glob:
    :maxdepth: 1

    example_notebooks/tabular_examples/model_agnostic/*


.. Partition explainer
.. ================
.. Examples using :class:`shap.explainers.Partition` to produce explanations in a model agnostic manner based on a hierarchical grouping of the features.

.. .. toctree::
..     :glob:
..     :maxdepth: 1

..     example_notebooks/tabular_examples/partition/*


.. Kernel explainer
.. ================
.. Examples using :class:`shap.explainers.Kernel` to produce explanations in a model agnostic manner.

.. .. toctree::
..     :glob:
..     :maxdepth: 1

..     example_notebooks/tabular_examples/kernel/*


.. Deep explainer
.. ================
.. Examples using :class:`shap.explainers.Deep` to produce approximate explanations of PyTorch or TensorFlow models.

.. .. toctree::
..     :glob:
..     :maxdepth: 1

..     example_notebooks/tabular_examples/deep/*


.. Gradient explainer
.. ================
.. Examples using :class:`shap.explainers.Gradient` to produce approximate explanations of PyTorch or TensorFlow models.

.. .. toctree::
..     :glob:
..     :maxdepth: 1

..     example_notebooks/tabular_examples/gradient/*
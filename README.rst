.. -*- mode: rst -*-

.. image:: https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/shap_header.png
  :target: https://github.com/slundberg/shap/
  :width: 800px

|Travis|_ |Binder|_ |Docs|_

.. |Travis| image:: https://travis-ci.org/slundberg/shap
.. _Travis: https://travis-ci.org/slundberg/shap.svg?branch=master

.. |Binder| image:: https://mybinder.org/badge_logo.svg
.. _Binder: https://mybinder.org/v2/gh/slundberg/shap/master

.. |Docs| image:: https://readthedocs.org/projects/shap/badge/?version=latest
.. _Docs: https://shap.readthedocs.io/en/latest/?badge=latest

**SHAP (SHapley Additive exPlanations)** is a game theoretic approach to explain the output of any machine learning model.
It connects optimal credit allocation with local explanations using the classic Shapley values from game theory and their
related extensions (see :ref:`papers <Citations>` for details and citations).

************
Installation
************

Shap can be installed from either `PyPI <https://pypi.org/project/shap>`_::

    pip install shap

or `conda-forge <https://anaconda.org/conda-forge/shap>`_::

    conda install -c conda-forge shap

****************
Sample notebooks
****************

The notebooks below demonstrate different use cases for SHAP. Look inside the
notebooks directory of the repository if you want to try playing with the original
notebooks yourself.

*************
TreeExplainer
*************

An implementation of Tree SHAP, a fast and exact algorithm to compute SHAP values for
trees and ensembles of trees.

- `NHANES survival model with XGBoost and SHAP interaction values <https://slundberg.github.io/shap/notebooks/NHANES%20I%20Survival%20Model.html>`_ 
  - Using mortality data from 20 years of followup this notebook demonstrates how to use
  XGBoost and `shap` to uncover complex risk factor relationships.

- `Census income classification with LightGBM <https://slundberg.github.io/shap/notebooks/tree_explainer/Census%20income%20classification%20with%20LightGBM.html>`_
  - Using the standard adult census income dataset, this notebook trains a gradient boosting
  tree model with LightGBM and then explains predictions using `shap`.

- `League of Legends Win Prediction with XGBoost <https://slundberg.github.io/shap/notebooks/League%20of%20Legends%20Win%20Prediction%20with%20XGBoost.html>`_
  - Using a Kaggle dataset of 180,000 ranked matches from League of Legends we train and explain
  a gradient boosting tree model with XGBoost to predict if a player will win their match.

*************
DeepExplainer
*************

An implementation of Deep SHAP, a faster (but only approximate) algorithm to compute SHAP values
for deep learning models that is based on connections between SHAP and the DeepLIFT algorithm.

- `MNIST Digit classification with Keras <https://slundberg.github.io/shap/notebooks/deep_explainer/Front%20Page%20DeepExplainer%20MNIST%20Example.html>`_
  - Using the MNIST handwriting recognition dataset, this notebook trains a neural network with
  Keras and then explains predictions using `shap`.

- `Keras LSTM for IMDB Sentiment Classification <https://slundberg.github.io/shap/notebooks/deep_explainer/Keras%20LSTM%20for%20IMDB%20Sentiment%20Classification.html>`_
  - This notebook trains an LSTM with Keras on the IMDB text sentiment analysis dataset and
  then explains predictions using `shap`. 

*****************
GradientExplainer
*****************

An implementation of expected gradients to approximate SHAP values for deep learning models.
It is based on connections between SHAP and the Integrated Gradients algorithm.
GradientExplainer is slower than DeepExplainer and makes different approximation
assumptions.

- `Explain an Intermediate Layer of VGG16 on ImageNet <https://slundberg.github.io/shap/notebooks/gradient_explainer/Explain%20an%20Intermediate%20Layer%20of%20VGG16%20on%20ImageNet.html>`_
  - This notebook demonstrates how to explain the output of a pre-trained
  VGG16 ImageNet model using an internal convolutional layer.

***************
LinearExplainer
***************

For a linear model with independent features we can analytically compute the exact SHAP
values. We can also account for feature correlation if we are willing to estimate the
feature covaraince matrix. LinearExplainer supports both of these options.

- `Sentiment Analysis with Logistic Regression <https://slundberg.github.io/shap/notebooks/linear_explainer/Sentiment%20Analysis%20with%20Logistic%20Regression.html>`_
  - This notebook demonstrates how to explain a linear logistic regression sentiment
  analysis model.

***************
KernelExplainer
***************

An implementation of Kernel SHAP, a model agnostic method to estimate SHAP values for any
model. Because it makes not assumptions about the model type, KernelExplainer is slower
than the other model type specific algorithms.

- `Census income classification with scikit-learn <https://slundberg.github.io/shap/notebooks/Census%20income%20classification%20with%20scikit-learn.html>`_
  - Using the standard adult census income dataset, this notebook trains a k-nearest
  neighbors classifier using scikit-learn and then explains predictions using `shap`.

- `ImageNet VGG16 Model with Keras <https://slundberg.github.io/shap/notebooks/ImageNet%20VGG16%20Model%20with%20Keras.html>`_
  - Explain the classic VGG16 convolutional nerual network's predictions for an image.
  This works by applying the model agnostic Kernel SHAP method to a super-pixel segmented
  image.

- `Iris classification <https://slundberg.github.io/shap/notebooks/Iris%20classification%20with%20scikit-learn.html>`_
  - A basic demonstration using the popular iris species dataset. It explains predictions
  from six different models in scikit-learn using `shap`.

***********************
Documentation notebooks
***********************

These notebooks comprehensively demonstrate how to use specific functions and objects. 

- `shap.decision_plot and shap.multioutput_decision_plot <https://slundberg.github.io/shap/notebooks/plots/decision_plot.html>`_

- `shap.dependence_plot <https://slundberg.github.io/shap/notebooks/plots/dependence_plot.html>`_

***********************
Methods Unified by SHAP
***********************

1. *LIME:* Ribeiro, Marco Tulio, Sameer Singh, and Carlos Guestrin. "Why should i trust you?: Explaining the predictions of any classifier." Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 2016.

2. *Shapley sampling values:* Strumbelj, Erik, and Igor Kononenko. "Explaining prediction models and individual predictions with feature contributions." Knowledge and information systems 41.3 (2014): 647-665.

3. *DeepLIFT:* Shrikumar, Avanti, Peyton Greenside, and Anshul Kundaje. "Learning important features through propagating activation differences." arXiv preprint arXiv:1704.02685 (2017).

4. *QII:* Datta, Anupam, Shayak Sen, and Yair Zick. "Algorithmic transparency via quantitative input influence: Theory and experiments with learning systems." Security and Privacy (SP), 2016 IEEE Symposium on. IEEE, 2016.

5. *Layer-wise relevance propagation:* Bach, Sebastian, et al. "On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation." PloS one 10.7 (2015): e0130140.

6. *Shapley regression values:* Lipovetsky, Stan, and Michael Conklin. "Analysis of regression in game theory approach." Applied Stochastic Models in Business and Industry 17.4 (2001): 319-330.

7. *Tree interpreter:* Saabas, Ando. Interpreting random forests. http://blog.datadive.net/interpreting-random-forests/

.. _Citations:

*********
Citations
*********

The algorithms and visualizations used in this package came primarily out of research in
`Su-In Lee's lab <https://suinlee.cs.washington.edu>`_ at the University of Washington, and
Microsoft Research. If you use SHAP in your research we would appreciate a citation to the
appropriate paper(s):

- For general use of SHAP you can read/cite our `NeurIPS paper <http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions>`_. 
- For TreeExplainer you can read/cite our `Nature Machine Intelligence paper <https://www.nature.com/articles/s42256-019-0138-9>`_.
- For `force_plot` visualizations and medical applications you can read/cite our `Nature Biomedical Engineering paper <https://www.nature.com/articles/s41551-018-0304-0>`_.

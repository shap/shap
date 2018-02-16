

<p align="center">
  <img src="https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/shap_diagram.png" width="400" />
</p>

---
<a href="https://travis-ci.org/slundberg/shap"><img src="https://travis-ci.org/slundberg/shap.svg?branch=master"></a>

**SHAP (SHapley Additive exPlanations)** is a unified approach to explain the output of any machine learning model. SHAP connects game theory with local explanations, uniting several previous methods [1-7] and representing the only possible consistent and locally accurate additive feature attribution method based on expectations (see the [SHAP NIPS paper](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions) for details).



## Install

```
pip install shap
```

## XGBoost (LightGBM) example

While SHAP values can explain the output of any machine learning model, we have developed a high-speed exact algorithm for ensemble tree methods ([Tree SHAP arXiv paper](https://arxiv.org/abs/1802.03888)). This has been integrated directly into XGBoost and LightGBM (*make sure you have the latest checkout of master*), and you can use the `shap` package for visualization in a Jupyter notebook:

```python
import xgboost
import shap

# load JS visualization code to notebook
shap.initjs() 

# train XGBoost model
X,y = shap.datasets.boston()
bst = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)

# explain the model's predictions using SHAP values (use pred_contrib in LightGBM)
shap_values = bst.predict(xgboost.DMatrix(X), pred_contribs=True)

# visualize the first prediction's explaination
shap.force_plot(shap_values[0,:], X.iloc[0,:])
```

<p align="center">
  <img width="811" src="https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/boston_instance.png" />
</p>

The above explanation shows features each contributing to push the model output from the base value (the average model output over the training dataset we passed) to the model output. Features pushing the prediction higher are shown in red, those pushing the prediction lower are in blue.

If we take many explanations such as the one shown above, rotate them 90 degrees, and then stack them horizontally, we can see explanations for an entire dataset (in the notebook this plot is interactive):

```python
# visualize the training set predictions
shap.force_plot(shap_values, X)
```

<p align="center">
  <img width="811" src="https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/boston_dataset.png" />
</p>

To understand how a single feature effects the output of the model we can plot the SHAP value of that feature vs. the value of the feature for all the examples in a dataset. Since SHAP values represent a feature's responsibility for a change in the model output, the plot below represents the change in predicted house price as RM (the average number of rooms per house in an area) changes. Vertical dispersion at a single value of RM represents interaction effects with other features. To help reveal these interactions `dependence_plot` automatically selects another feature for coloring. In this case coloring by RAD (index of accessibility to radial highways) highlights that RM has less impact on home price for areas close to radial highways.

```python
# create a SHAP dependence plot to show the effect of a single feature across the whole dataset
shap.dependence_plot("RM", shap_values, X)
```

<p align="center">
  <img width="544" src="https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/boston_dependence_plot.png" />
</p>


To get an overview of which features are most important for a model we can plot the SHAP values of every feature for every sample. The plot below sorts features by the sum of SHAP value magnitudes over all samples, and uses SHAP values to show the distribution of the impacts each feature has on the model output. The color represents the feature value (red high, blue low). This reveals for example that a high LSTAT (% lower status of the population) lowers the predicted home price.

```python
# summarize the effects of all the features
shap.summary_plot(shap_values, X)
```

<p align="center">
  <img width="483" src="https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/boston_summary_plot.png" />
</p>


## Model agnostic SVM example

Kernel SHAP uses a specially-weighted local linear regression to estimate SHAP values for any model. Below is a simple example for explaining a multi-class SVM on the classic iris dataset.

```python
import sklearn
import shap
from sklearn.model_selection import train_test_split

# print the JS visualization code to the notebook
shap.initjs()

# train a SVM classifier
X_train,X_test,Y_train,Y_test = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)
svm = sklearn.svm.SVC(kernel='rbf', probability=True)
svm.fit(X_train, Y_train)

# use Kernel SHAP to explain test set predictions
explainer = shap.KernelExplainer(svm.predict_proba, X_train, link="logit")
shap_values = explainer.shap_values(X_test, nsamples=100)

# plot the SHAP values for the Setosa output of the first instance
shap.force_plot(shap_values[0][0,:], X_test.iloc[0,:], link="logit")
```
<p align="center">
  <img width="810" src="https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/iris_instance.png" />
</p>

The above explanation shows four features each contributing to push the model output from the base value (the average model output over the training dataset we passed) towards zero. If there were any features pushing the class label higher they would be shown in red.

If we take many explanations such as the one shown above, rotate them 90 degrees, and then stack them horizontally, we can see explanations for an entire dataset. This is exactly what we do below for all the examples in the iris test set:

```python
# plot the SHAP values for the Setosa output of all instances
shap.force_plot(shap_values[0], X_test, link="logit")
```
<p align="center">
  <img width="813" src="https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/iris_dataset.png" />
</p>

## Model agnostic VGG16 example

The VGG16 notebook below illustrates how to apply Kernel SHAP to image classification, which can help identify what parts of an image caused a prediction.

<p align="center">
  <img width="583" src="https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/strawberry_example.png" />
</p>

## SHAP Interaction Values

SHAP interaction values are a generalization of SHAP values to higher order interactions. Fast exact computation of pairwise interactions are implemented in the latest version of XGBoost with the `pred_interactions` flag. With this flag XGBoost returns a matrix for every prediction, where the main effects are on the diagonal and the interaction effects are off-diagonal. These values often reveal interesting hidden relationships, such as how the increased risk of death peaks for men at age 60 (see the NHANES notebook for details):

<p align="center">
  <img width="483" src="https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/nhanes_age_sex_interaction.png" />
</p>

## Sample notebooks

The notebooks below demonstrate different use cases for SHAP. Look inside the notebooks directory of the repository if you want to try playing with the original notebooks yourself. Note that the LightGBM and XGBoost examples use the fast and exact Tree SHAP algorithm, the others use the model agnostic Kernel SHAP algorithm.

- [**NHANES survival model with XGBoost and SHAP interaction values**](https://slundberg.github.io/shap/notebooks/NHANES+I+Survival+Model.html) - Using mortality data from 20 years of followup this notebook demonstrates how to use XGBoost and `shap` to uncover complex risk factor relationships.

- [**Census income classification with LightGBM**](https://slundberg.github.io/shap/notebooks/Census+income+classification+with+LightGBM.html) - Using the standard adult census income dataset, this notebook trains a gradient boosting tree model with LightGBM and then explains predictions using `shap`.

- [**Census income classification with Keras**](https://slundberg.github.io/shap/notebooks/Census+income+classification+with+Keras.html) - Using the standard adult census income dataset, this notebook trains a neural network with Keras and then explains predictions using `shap`.

- [**Census income classification with scikit-learn**](https://slundberg.github.io/shap/notebooks/Census+income+classification+with+scikit-learn.html) - Using the standard adult census income dataset, this notebook trains a k-nearest neighbors classifier using scikit-learn and then explains predictions using `shap`.

- [**League of Legends Win Prediction with XGBoost**](https://slundberg.github.io/shap/notebooks/League+of+Legends+Win+Prediction+with+XGBoost.html) - Using a Kaggle dataset of 180,000 ranked matches from League of Legends we train and explain a gradient boosting tree model with XGBoost to predict if a player will win their match.

- [**ImageNet VGG16 Model with Keras**](https://slundberg.github.io/shap/notebooks/ImageNet+VGG16+Model+with+Keras.html) - Explain the classic VGG16 convolutional nerual network's predictions for an image. This works by applying the model agnostic Kernel SHAP method to a super-pixel segmented image.

- [**Iris classification**](https://slundberg.github.io/shap/notebooks/Iris+classification+with+scikit-learn.html) - A basic demonstration using the popular iris species dataset. It explains predictions from six different models in scikit-learn using `shap`.

### Methods Unified by SHAP

1. *LIME:* Ribeiro, Marco Tulio, Sameer Singh, and Carlos Guestrin. "Why should i trust you?: Explaining the predictions of any classifier." Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 2016.

2. *Shapley sampling values:* Štrumbelj, Erik, and Igor Kononenko. "Explaining prediction models and individual predictions with feature contributions." Knowledge and information systems 41.3 (2014): 647-665.

3. *DeepLIFT:* Shrikumar, Avanti, Peyton Greenside, and Anshul Kundaje. "Learning important features through propagating activation differences." arXiv preprint arXiv:1704.02685 (2017).

4. *QII:* Datta, Anupam, Shayak Sen, and Yair Zick. "Algorithmic transparency via quantitative input influence: Theory and experiments with learning systems." Security and Privacy (SP), 2016 IEEE Symposium on. IEEE, 2016.

5. *Layer-wise relevance propagation:* Bach, Sebastian, et al. "On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation." PloS one 10.7 (2015): e0130140.

6. *Shapley regression values:* Lipovetsky, Stan, and Michael Conklin. "Analysis of regression in game theory approach." Applied Stochastic Models in Business and Industry 17.4 (2001): 319-330.

7. *Tree interpreter:* Saabas, Ando. Interpreting random forests. http://blog.datadive.net/interpreting-random-forests/

<img height="1" width="1" style="display:none" src="https://www.facebook.com/tr?id=189147091855991&ev=PageView&noscript=1" />


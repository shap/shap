<p align="center">
  <img src="https://slundberg.github.io/shap/artwork/logo.png" />
</p>

---

**Shap** explains the output of any machine learning model using expectations and Shapley values. Under certain assumptions it can be shown to be the optimal linear explanation of a model prediction (see our current [short paper](https://arxiv.org/abs/1611.07478) for details).

## Install

```
pip install shap
```

## Example (run in a Jupyter notebook)

```python
from shap import ShapExplainer, DenseData, visualize, initjs
from sklearn import datasets,neighbors
from numpy import random, arange

# print the JS visualization code to the notebook
initjs()

# train a k-nearest neighbors classifier on a random subset 
iris = datasets.load_iris()
random.seed(2)
inds = arange(len(iris.target))
random.shuffle(inds)
knn = neighbors.KNeighborsClassifier()
knn.fit(iris.data, iris.target == 0)

# use Shap to explain a single prediction
X = DenseData(iris.feature_names, iris.data[inds[:100],:]) # name the features
explainer = ShapExplainer(knn.predict, X, nsamples=100)
visualize(explainer.explain(iris.data[inds[102:103],:]))
```
<p align="center">
  <img src="https://slundberg.github.io/shap/artwork/simple_iris_explanation.png" />
</p>

The above explanation shows three features each contributing to push the model output from the base value (the average model output over the training dataset we passed) to zero. If there were any features pushing the class label higher they would be shown in red.

If we take many explanations such as the one shown above, rotate them 90 degrees, and then stack them horizontally, we can see explanations for an entire dataset. This is exactly we do below for all the examples in the iris test set:

```python
# use Shap to explain all test set predictions
visualize([explainer.explain(iris.data[inds[i:i+1],:]) for i in range(100,len(iris.target))])
```
<p align="center">
  <img src="https://slundberg.github.io/shap/artwork/simple_iris_dataset.png" />
</p>

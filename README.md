# Shap

Shap explains the output of any machine learning model using expectations and Shapley values. It is easy to install and use, and under mild assumptions can be shown to be the optimal additive explanation of a model using the model inputs.

## Install

```
pip install shap
```

## Example

```python
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.neighbors
import shap
from shap import ShapExplainer, visualize

shap.initjs()
```

```python
iris = sklearn.datasets.load_iris()
iris_data = shap.DenseData(iris.feature_names, iris.data)
knn = sklearn.neighbors.KNeighborsClassifier()
knn.fit(iris.data, iris.target == 0)

explaination = ShapExplainer(knn.predict, iris_data, nsamples=100).explain(iris.data[60:61,:])
visualize(explaination)
```

# Shap

Shap explains the output of any machine learning model using expectations and Shapley values. It is easy to install and use, and under mild assumptions can be shown to be the optimal additive explanation of a model using the model inputs.

## Install

```
pip install shap
```

## Usage

```python
from shap import ShapExplainer, visualize

explanation = ShapExplainer(model_function, reference_dataset).explain(x)
visualize(explaination) # works in a Jupyter notebook
```

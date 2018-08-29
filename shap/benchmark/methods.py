from .. import LinearExplainer
from .. import KernelExplainer
from .. import SamplingExplainer
from .. import TreeExplainer
from .. import kmeans
from ..explainers import other
import numpy as np


def linear_shap_corr(model, data):
    """ Linear SHAP (corr 1000)
    """
    return LinearExplainer(model, data, nsamples=1000).shap_values

def linear_shap_ind(model, data):
    """ Linear SHAP (ind)
    """
    return LinearExplainer(model, data, feature_dependence="interventional").shap_values

def coef(model, data):
    """ Coefficents
    """
    return other.CoefficentExplainer(model).attributions

def random(model, data):
    """ Random
    """
    return other.RandomExplainer().attributions

def kernel_shap_1000_meanref(model, data):
    """ Kernel SHAP 1000 mean ref.
    """
    return lambda X: KernelExplainer(model.predict, kmeans(data, 1)).shap_values(X, nsamples=1000, l1_reg=0)

def sampling_shap_1000(model, data):
    """ Sampling SHAP 1000
    """
    return lambda X: SamplingExplainer(model.predict, data).shap_values(X, nsamples=1000)

def tree_shap(model, data):
    """ Tree SHAP
    """
    return TreeExplainer(model).shap_values

def mean_abs_tree_shap(model, data):
    """ mean(|Tree SHAP|)
    """
    return lambda X: np.tile(np.abs(TreeExplainer(model).shap_values(X)).mean(0), (X.shape[0], 1))

def saabas(model, data):
    """ Saabas
    """
    return lambda X: TreeExplainer(model).shap_values(X, approximate=True)

def tree_gain(model, data):
    """ Gain/Gini Importance
    """
    return other.TreeGainExplainer(model).attributions

def lime_tabular_regression_1000(model, data):
    """ LIME Tabular 1000
    """
    return lambda X: other.LimeTabularExplainer(model.predict, data, mode="regression").attributions(X, nsamples=1000)

# functions = {
#     "linear_shap_corr": lambda model, X: LinearExplainer(model, X, nsamples=1000).shap_values,
#     "linear_shap_ind": lambda model, X: LinearExplainer(model, X, feature_dependence="interventional").shap_values,
#     "coef": lambda model, X: other.CoefficentExplainer(model).attributions,
#     "random": lambda model, X: other.RandomExplainer().attributions,
#     "kernel_shap_1000_meanref": lambda model, Xt: lambda X: KernelExplainer(model.predict, kmeans(Xt, 1)).shap_values(X, nsamples=1000, l1_reg=0), # pylint: disable=E0602
#     "kernel_shap_100_meanref": lambda model, Xt: lambda X: KernelExplainer(model.predict, kmeans(Xt, 1)).shap_values(X, nsamples=100, l1_reg=0), # pylint: disable=E0602
#     "sampling_shap_10000": lambda model, Xt: lambda X: SamplingExplainer(model.predict, Xt).shap_values(X, nsamples=10000), # pylint: disable=E0602
#     "sampling_shap_1000": lambda model, Xt: lambda X: SamplingExplainer(model.predict, Xt).shap_values(X, nsamples=1000), # pylint: disable=E0602
#     "sampling_shap_100": lambda model, Xt: lambda X: SamplingExplainer(model.predict, Xt).shap_values(X, nsamples=100), # pylint: disable=E0602
#     "tree_shap": lambda model, Xt: TreeExplainer(model).shap_values,
#     "mean_abs_tree_shap": lambda model, Xt: lambda X: np.tile(np.abs(TreeExplainer(model).shap_values(X)).mean(0), (X.shape[0], 1)), # pylint: disable=E0602
#     "saabas": lambda model, Xt: lambda X: TreeExplainer(model).shap_values(X, approximate=True), # pylint: disable=E0602
#     "tree_gain": lambda model, X: other.TreeGainExplainer(model).attributions,
#     "lime_tabular_regression_1000": lambda model, Xt: lambda X: other.LimeTabularExplainer(model.predict, Xt, mode="regression").attributions(X, nsamples=1000), # pylint: disable=E0602
# }

linear_regress = [
    # NEED LIME
    "linear_shap_corr",
    "linear_shap_ind",
    "coef",
    "random",
    "kernel_shap_1000_meanref",
    #"kernel_shap_100_meanref",
    #"sampling_shap_10000",
    "sampling_shap_1000",
    "lime_tabular_regression_1000"
    #"sampling_shap_100"
]

tree_regress = [
    # NEED tree_shap_ind
    # NEED split_count?
    "tree_shap",
    "saabas",
    "random",
    "tree_gain",
    "kernel_shap_1000_meanref",
    "mean_abs_tree_shap",
    #"kernel_shap_100_meanref",
    #"sampling_shap_10000",
    "sampling_shap_1000",
    "lime_tabular_regression_1000"
    #"sampling_shap_100"
]

deep_regress = [
    # NEED deepexplainer
    # NEED gradientexplainer
    "random",
    "kernel_shap_1000_meanref",
    "sampling_shap_1000",
    #"lime_tabular_regression_1000"
]

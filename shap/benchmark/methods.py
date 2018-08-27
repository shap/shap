from .. import LinearExplainer
from .. import KernelExplainer
from .. import SamplingExplainer
from .. import TreeExplainer
from ..explainers import other
import numpy as np

method_dict = {
    "linear_shap_corr": lambda model, X: LinearExplainer(model, X, nsamples=1000).shap_values,
    "linear_shap_ind": lambda model, X: LinearExplainer(model, X, feature_dependence="interventional").shap_values,
    "coef": lambda model, X: other.CoefficentExplainer(model).attributions,
    "random": lambda model, X: other.RandomExplainer().attributions,
    "kernel_shap_1000_meanref": lambda model, Xt: lambda X: KernelExplainer(model.predict, Xt.mean(0)).shap_values(X, nsamples=1000, l1_reg=0), # pylint: disable=E0602
    "kernel_shap_100_meanref": lambda model, Xt: lambda X: KernelExplainer(model.predict, Xt.mean(0)).shap_values(X, nsamples=100, l1_reg=0), # pylint: disable=E0602
    "sampling_shap_10000": lambda model, Xt: lambda X: SamplingExplainer(model.predict, Xt).shap_values(X, nsamples=10000), # pylint: disable=E0602
    "sampling_shap_1000": lambda model, Xt: lambda X: SamplingExplainer(model.predict, Xt).shap_values(X, nsamples=1000), # pylint: disable=E0602
    "sampling_shap_100": lambda model, Xt: lambda X: SamplingExplainer(model.predict, Xt).shap_values(X, nsamples=100), # pylint: disable=E0602
    "tree_shap": lambda model, Xt: TreeExplainer(model).shap_values,
    "mean_abs_tree_shap": lambda model, Xt: lambda X: np.tile(np.abs(TreeExplainer(model).shap_values(X)).mean(0), (X.shape[0], 1)), # pylint: disable=E0602
    "saabas": lambda model, Xt: lambda X: TreeExplainer(model).shap_values(X, approximate=True), # pylint: disable=E0602
    "tree_gini": lambda model, X: other.TreeGiniExplainer(model).attributions
}

linear_regress = [[m, method_dict[m]] for m in [
    "linear_shap_corr",
    "linear_shap_ind",
    "coef",
    "random",
    "kernel_shap_1000_meanref",
    #"kernel_shap_100_meanref",
    #"sampling_shap_10000",
    "sampling_shap_1000",
    #"sampling_shap_100"
]]

tree_regress = [[m, method_dict[m]] for m in [
    # NEED tree_shap_ind
    # NEED split_count?
    "tree_shap",
    "saabas",
    "random",
    "tree_gini",
    "kernel_shap_1000_meanref",
    "mean_abs_tree_shap",
    #"kernel_shap_100_meanref",
    #"sampling_shap_10000",
    "sampling_shap_1000",
    #"sampling_shap_100"
]]

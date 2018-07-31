from .. import LinearExplainer
from .. import KernelExplainer
from .. import SamplingExplainer
from .. import TreeExplainer
from ..explainers import other

method_dict = {
    "Linear SHAP (corr)": lambda model, X: LinearExplainer(model, X, nsamples=1000).shap_values,
    "Linear SHAP (ind)": lambda model, X: LinearExplainer(model, X, feature_dependence="interventional").shap_values,
    "Coef": lambda model, X: other.CoefficentExplainer(model).attributions,
    "Random": lambda model, X: other.RandomExplainer().attributions,
    "Kernel SHAP 1000 mean ref.": lambda model, Xt: lambda X: KernelExplainer(model.predict, Xt.mean(0)).shap_values(X, nsamples=1000, l1_reg=0),
    "Kernel SHAP 100 mean ref.": lambda model, Xt: lambda X: KernelExplainer(model.predict, Xt.mean(0)).shap_values(X, nsamples=100, l1_reg=0),
    "Sampling SHAP 10000": lambda model, Xt: lambda X: SamplingExplainer(model.predict, Xt).shap_values(X, nsamples=10000),
    "Sampling SHAP 1000": lambda model, Xt: lambda X: SamplingExplainer(model.predict, Xt).shap_values(X, nsamples=1000),
    "Sampling SHAP 100": lambda model, Xt: lambda X: SamplingExplainer(model.predict, Xt).shap_values(X, nsamples=100),
    "Tree SHAP": lambda model, Xt: TreeExplainer(model).shap_values,
    "Saabas": lambda model, Xt: lambda X: TreeExplainer(model).shap_values(X, approximate=True)
}

linear = [[m, method_dict[m]] for m in [
    "Linear SHAP (corr)",
    "Linear SHAP (ind)",
    "Coef",
    "Random",
    ##"Kernel SHAP 1000 mean ref.",
    #"Kernel SHAP 100 mean ref.",
    #"Sampling SHAP 10000",
    ##"Sampling SHAP 1000",
    #"Sampling SHAP 100"
]]

tree = [[m, method_dict[m]] for m in [
    "Tree SHAP",
    "Saabas",
    "Random"
    ##"Kernel SHAP 1000 mean ref.",
    #"Kernel SHAP 100 mean ref.",
    #"Sampling SHAP 10000",
    ##"Sampling SHAP 1000",
    #"Sampling SHAP 100"
]]

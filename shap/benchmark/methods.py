from .. import LinearExplainer
from .. import KernelExplainer
from .. import SamplingExplainer
from .. import TreeExplainer
from .. import DeepExplainer
from .. import GradientExplainer
from .. import kmeans
from ..explainers import other
from .models import KerasWrap
import numpy as np
import sklearn

def linear_shap_corr(model, data):
    """ Linear SHAP (corr 1000)
    """
    return LinearExplainer(model, data, feature_dependence="correlation", nsamples=1000).shap_values

def linear_shap_ind(model, data):
    """ Linear SHAP (ind)
    """
    return LinearExplainer(model, data, feature_dependence="independent").shap_values

def coef(model, data):
    """ Coefficents
    """
    return other.CoefficentExplainer(model).attributions

def random(model, data):
    """ Random
    color = #777777
    linestyle = solid
    """
    return other.RandomExplainer().attributions

def kernel_shap_1000_meanref(model, data):
    """ Kernel SHAP 1000 mean ref.
    color = red_blue_circle(0.5)
    linestyle = solid
    """
    return lambda X: KernelExplainer(model.predict, kmeans(data, 1)).shap_values(X, nsamples=1000, l1_reg=0)

def sampling_shap_1000(model, data):
    """ Sampling SHAP 1000
    color = red_blue_circle(0.5)
    linestyle = dashed
    """
    return lambda X: SamplingExplainer(model.predict, data).shap_values(X, nsamples=1000)

def tree_shap_tree_path_dependent(model, data):
    """ Tree SHAP (path dependent)
    color = red_blue_circle(0)
    linestyle = solid
    """
    return TreeExplainer(model, feature_dependence="tree_path_dependent").shap_values

def tree_shap_independent_1000(model, data):
    """ Tree SHAP (independent)
    color = red_blue_circle(0)
    linestyle = dashed
    """
    data1000 = sklearn.utils.resample(data, replace=False, n_samples=min(1000, data.shape[0]), random_state=0)
    return TreeExplainer(model, data1000, feature_dependence="independent").shap_values

def mean_abs_tree_shap(model, data):
    """ mean(|Tree SHAP|)
    color = red_blue_circle(0.25)
    linestyle = solid
    """
    def f(X):
        v = TreeExplainer(model).shap_values(X)
        if isinstance(v, list):
            return [np.tile(np.abs(sv).mean(0), (X.shape[0], 1)) for sv in v]
        else:
            return np.tile(np.abs(v).mean(0), (X.shape[0], 1))
    return f

def saabas(model, data):
    """ Saabas
    color = red_blue_circle(0)
    linestyle = dotted
    """
    return lambda X: TreeExplainer(model).shap_values(X, approximate=True)

def tree_gain(model, data):
    """ Gain/Gini Importance
    color = red_blue_circle(0.25)
    linestyle = dotted
    """
    return other.TreeGainExplainer(model).attributions

def lime_tabular_regression_1000(model, data):
    """ LIME Tabular 1000
    """
    return lambda X: other.LimeTabularExplainer(model.predict, data, mode="regression").attributions(X, nsamples=1000)

def deep_shap(model, data):
    """ Deep SHAP (DeepLIFT)
    """
    if isinstance(model, KerasWrap):
        model = model.model
    explainer = DeepExplainer(model, kmeans(data, 1).data)
    def f(X):
        phi = explainer.shap_values(X)
        if type(phi) is list and len(phi) == 1:
            return phi[0]
        else:
            return phi
    
    return f

def expected_gradients(model, data):
    """ Expected Gradients
    """
    if isinstance(model, KerasWrap):
        model = model.model
    explainer = GradientExplainer(model, data)
    def f(X):
        phi = explainer.shap_values(X)
        if type(phi) is list and len(phi) == 1:
            return phi[0]
        else:
            return phi
    
    return f

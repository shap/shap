# flake8: noqa

import warnings
import sys

__version__ = '0.39.0'

# check python version
if (sys.version_info < (3, 0)):
    warnings.warn("As of version 0.29.0 shap only supports Python 3 (not 2)!")

from ._explanation import Explanation, Cohorts

# explainers
from .explainers._explainer import Explainer
from .explainers._kernel import Kernel as KernelExplainer
from .explainers._sampling import Sampling as SamplingExplainer
from .explainers._tree import Tree as TreeExplainer
from .explainers._gpu_tree import GPUTree as GPUTreeExplainer
from .explainers._deep import Deep as DeepExplainer
from .explainers._gradient import Gradient as GradientExplainer
from .explainers._linear import Linear as LinearExplainer
from .explainers._partition import Partition as PartitionExplainer
from .explainers._permutation import Permutation as PermutationExplainer
from .explainers._additive import Additive as AdditiveExplainer
from .explainers import other

# plotting (only loaded if matplotlib is present)
def unsupported(*args, **kwargs):
    warnings.warn("matplotlib is not installed so plotting is not available! Run `pip install matplotlib` to fix this.")

try:
    import matplotlib
    have_matplotlib = True
except ImportError:
    have_matplotlib = False
if have_matplotlib:
    from .plots._beeswarm import summary_legacy as summary_plot
    from .plots._decision import decision as decision_plot, multioutput_decision as multioutput_decision_plot
    from .plots._scatter import dependence_legacy as dependence_plot
    from .plots._force import force as force_plot, initjs, save_html, getjs
    from .plots._image import image as image_plot
    from .plots._monitoring import monitoring as monitoring_plot
    from .plots._embedding import embedding as embedding_plot
    from .plots._partial_dependence import partial_dependence as partial_dependence_plot
    from .plots._bar import bar_legacy as bar_plot
    from .plots._waterfall import waterfall as waterfall_plot
    from .plots._group_difference import group_difference as group_difference_plot
    from .plots._text import text as text_plot
else:
    summary_plot = unsupported
    decision_plot = unsupported
    multioutput_decision_plot = unsupported
    dependence_plot = unsupported
    force_plot = unsupported
    initjs = unsupported
    save_html = unsupported
    image_plot = unsupported
    monitoring_plot = unsupported
    embedding_plot = unsupported
    partial_dependence_plot = unsupported
    bar_plot = unsupported
    waterfall_plot = unsupported
    text_plot = unsupported


# other stuff :)
from . import datasets
from . import utils
from . import links
from . import exceptions

from .actions._optimizer import ActionOptimizer

#from . import benchmark

from .utils._legacy import kmeans
from .utils import sample, approximate_interactions

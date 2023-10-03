
__version__ = "0.43.0"

from ._explanation import Cohorts, Explanation

# explainers
from .explainers import other
from .explainers._additive import AdditiveExplainer
from .explainers._deep import DeepExplainer
from .explainers._exact import ExactExplainer
from .explainers._explainer import Explainer
from .explainers._gpu_tree import GPUTreeExplainer
from .explainers._gradient import GradientExplainer
from .explainers._kernel import KernelExplainer
from .explainers._linear import LinearExplainer
from .explainers._partition import PartitionExplainer
from .explainers._permutation import PermutationExplainer
from .explainers._sampling import SamplingExplainer
from .explainers._tree import TreeExplainer

_no_matplotlib_warning = "matplotlib is not installed so plotting is not available! Run `pip install matplotlib` " \
                         "to fix this."


# plotting (only loaded if matplotlib is present)
def unsupported(*args, **kwargs):
    raise ImportError(_no_matplotlib_warning)


class UnsupportedModule:
    def __getattribute__(self, item):
        raise ImportError(_no_matplotlib_warning)


try:
    import matplotlib  # noqa: F401
    have_matplotlib = True
except ImportError:
    have_matplotlib = False
if have_matplotlib:
    from . import plots
    from .plots._bar import bar_legacy as bar_plot
    from .plots._beeswarm import summary_legacy as summary_plot
    from .plots._decision import decision as decision_plot
    from .plots._decision import multioutput_decision as multioutput_decision_plot
    from .plots._embedding import embedding as embedding_plot
    from .plots._force import force as force_plot
    from .plots._force import getjs, initjs, save_html
    from .plots._group_difference import group_difference as group_difference_plot
    from .plots._heatmap import heatmap as heatmap_plot
    from .plots._image import image as image_plot
    from .plots._monitoring import monitoring as monitoring_plot
    from .plots._partial_dependence import partial_dependence as partial_dependence_plot
    from .plots._scatter import dependence_legacy as dependence_plot
    from .plots._text import text as text_plot
    from .plots._violin import violin as violin_plot
    from .plots._waterfall import waterfall as waterfall_plot
else:
    bar_plot = unsupported
    summary_plot = unsupported
    decision_plot = unsupported
    multioutput_decision_plot = unsupported
    embedding_plot = unsupported
    force_plot = unsupported
    getjs = unsupported
    initjs = unsupported
    save_html = unsupported
    group_difference_plot = unsupported
    heatmap_plot = unsupported
    image_plot = unsupported
    monitoring_plot = unsupported
    partial_dependence_plot = unsupported
    dependence_plot = unsupported
    text_plot = unsupported
    violin_plot = unsupported
    waterfall_plot = unsupported
    # If matplotlib is available, then the plots submodule will be directly available.
    # If not, we need to define something that will issue a meaningful warning message
    # (rather than ModuleNotFound).
    plots = UnsupportedModule()


# other stuff :)
from . import datasets, links, utils
from .actions._optimizer import ActionOptimizer
from .utils import approximate_interactions, sample

#from . import benchmark
from .utils._legacy import kmeans

# Use __all__ to let type checkers know what is part of the public API.
__all__ = [
    "Cohorts",
    "Explanation",

    # Explainers
    "other",
    "AdditiveExplainer",
    "DeepExplainer",
    "ExactExplainer",
    "Explainer",
    "GPUTreeExplainer",
    "GradientExplainer",
    "KernelExplainer",
    "LinearExplainer",
    "PartitionExplainer",
    "PermutationExplainer",
    "SamplingExplainer",
    "TreeExplainer",

    # Plots
    "plots",
    "bar_plot",
    "summary_plot",
    "decision_plot",
    "multioutput_decision_plot",
    "embedding_plot",
    "force_plot",
    "getjs",
    "initjs",
    "save_html",
    "group_difference_plot",
    "heatmap_plot",
    "image_plot",
    "monitoring_plot",
    "partial_dependence_plot",
    "dependence_plot",
    "text_plot",
    "violin_plot",
    "waterfall_plot",

    # Other stuff
    "datasets",
    "links",
    "utils",
    "ActionOptimizer",
    "approximate_interactions",
    "sample",
    "kmeans",
]

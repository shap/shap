# flake8: noqa

import warnings

__version__ = '0.35.0'

# explainers
from .explainers.kernel import KernelExplainer, kmeans
from .explainers.sampling import SamplingExplainer
from .explainers.tree import TreeExplainer, Tree
from .explainers.deep import DeepExplainer
from .explainers.gradient import GradientExplainer
from .explainers.linear import LinearExplainer
from .explainers.partition import PartitionExplainer, TokenMasker, ImageMasker
from .explainers.bruteforce import BruteForceExplainer
from .explainers.permutation import PermutationExplainer
from .explainers.additive import AdditiveExplainer
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
    from .plots.summary import summary_plot
    from .plots.decision import decision_plot, multioutput_decision_plot
    from .plots.dependence import dependence_plot
    from .plots.force import force_plot, initjs, save_html
    from .plots.image import image_plot
    from .plots.monitoring import monitoring_plot
    from .plots.embedding import embedding_plot
    from .plots.partial_dependence import partial_dependence_plot
    from .plots.bar import bar_plot
    from .plots.waterfall import waterfall_plot
    from .plots.text import text_plot
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
#from . import benchmark
from .common import approximate_interactions, hclust_ordering, sample, partition_tree

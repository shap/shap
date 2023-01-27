__version__ = "0.0.1"

# from .explainers._explainer import Explainer
from .explainers._kernel import KernelExplainer
from .plots._beeswarm import summary_legacy as summary_plot
from .plots._decision import decision as decision_plot
from .plots._decision import multioutput_decision as multioutput_decision_plot
from .plots._force import force as force_plot
from .plots._force import initjs, save_html
from .plots._monitoring import monitoring as monitoring_plot
from .plots._partial_dependence import partial_dependence as partial_dependence_plot
from .plots._scatter import dependence_legacy as dependence_plot

__all__ = [
    "KernelExplainer",
    "summary_plot",
    "decision_plot",
    "multioutput_decision_plot",
    "force_plot",
    "initjs",
    "save_html",
    "monitoring_plot",
    "partial_dependence_plot",
    "dependence_plot",
]

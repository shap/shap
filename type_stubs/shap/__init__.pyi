# Type stubs for shap
from typing import Any

# other modules
from . import datasets as datasets
from . import links as links
from . import utils as utils
from ._explanation import Cohorts as Cohorts
from ._explanation import Explanation as Explanation
from .actions._optimizer import ActionOptimizer as ActionOptimizer
from .explainers import (
    AdditiveExplainer as AdditiveExplainer,
)
from .explainers import (
    CoalitionExplainer as CoalitionExplainer,
)
from .explainers import (
    DeepExplainer as DeepExplainer,
)
from .explainers import (
    ExactExplainer as ExactExplainer,
)
from .explainers import (
    Explainer as Explainer,
)
from .explainers import (
    GPUTreeExplainer as GPUTreeExplainer,
)
from .explainers import (
    GradientExplainer as GradientExplainer,
)
from .explainers import (
    KernelExplainer as KernelExplainer,
)
from .explainers import (
    LinearExplainer as LinearExplainer,
)
from .explainers import (
    PartitionExplainer as PartitionExplainer,
)
from .explainers import (
    PermutationExplainer as PermutationExplainer,
)
from .explainers import (
    SamplingExplainer as SamplingExplainer,
)
from .explainers import (
    TreeExplainer as TreeExplainer,
)
from .utils import approximate_interactions as approximate_interactions
from .utils import sample as sample
from .utils._legacy import kmeans as kmeans

__version__: str

# plotting functions (conditional on matplotlib)
def bar_plot(*args: Any, **kwargs: Any) -> Any: ...
def summary_plot(*args: Any, **kwargs: Any) -> Any: ...
def decision_plot(*args: Any, **kwargs: Any) -> Any: ...
def multioutput_decision_plot(*args: Any, **kwargs: Any) -> Any: ...
def embedding_plot(*args: Any, **kwargs: Any) -> Any: ...
def force_plot(*args: Any, **kwargs: Any) -> Any: ...
def getjs(*args: Any, **kwargs: Any) -> Any: ...
def initjs(*args: Any, **kwargs: Any) -> Any: ...
def save_html(*args: Any, **kwargs: Any) -> Any: ...
def group_difference_plot(*args: Any, **kwargs: Any) -> Any: ...
def heatmap_plot(*args: Any, **kwargs: Any) -> Any: ...
def image_plot(*args: Any, **kwargs: Any) -> Any: ...
def monitoring_plot(*args: Any, **kwargs: Any) -> Any: ...
def partial_dependence_plot(*args: Any, **kwargs: Any) -> Any: ...
def dependence_plot(*args: Any, **kwargs: Any) -> Any: ...
def text_plot(*args: Any, **kwargs: Any) -> Any: ...
def violin_plot(*args: Any, **kwargs: Any) -> Any: ...
def waterfall_plot(*args: Any, **kwargs: Any) -> Any: ...

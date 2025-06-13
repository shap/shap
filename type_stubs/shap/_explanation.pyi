# Type stubs for shap._explanation
# Generated for SHAP library to provide code completion in VS Code

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

import pandas as pd
from numpy.typing import NDArray

from .utils._general import OpChain

Self = TypeVar("Self", bound="Explanation")

@dataclass
class OpHistoryItem:
    """An operation that has been applied to an Explanation object."""

    name: str
    prev_shape: tuple[int, ...]
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    collapsed_instances: bool = False

class MetaExplanation(type):
    """This metaclass exposes the Explanation object's class methods for creating template op chains."""

    def __getitem__(cls, item) -> OpChain: ...
    @property
    def abs(cls) -> OpChain:
        """Element-wise absolute value op."""
        ...

    @property
    def identity(cls) -> OpChain:
        """A no-op."""
        ...

    @property
    def argsort(cls) -> OpChain:
        """Numpy style argsort."""
        ...

    @property
    def flip(cls) -> OpChain:
        """Numpy style flip."""
        ...

    @property
    def sum(cls) -> OpChain:
        """Numpy style sum."""
        ...

    @property
    def max(cls) -> OpChain:
        """Numpy style max."""
        ...

    @property
    def min(cls) -> OpChain:
        """Numpy style min."""
        ...

    @property
    def mean(cls) -> OpChain:
        """Numpy style mean."""
        ...

    @property
    def sample(cls) -> OpChain:
        """Numpy style sample."""
        ...

    @property
    def hclust(cls) -> OpChain:
        """Hierarchical clustering op."""
        ...

class Explanation(metaclass=MetaExplanation):
    """A sliceable set of parallel arrays representing a SHAP explanation.

    Notes
    -----
    The *instance* methods such as `.max()` return new Explanation objects with the
    operation applied.

    The *class* methods such as `Explanation.max` return OpChain objects that represent
    a set of dot chained operations without actually running them.
    """

    op_history: list[OpHistoryItem]
    compute_time: float | None
    output_dims: tuple[int, ...]
    _s: Any  # Slicer object

    def __init__(
        self,
        values: NDArray[Any] | list[NDArray[Any]] | Any,
        base_values: NDArray[Any] | list[NDArray[Any]] | float | int | None = None,
        data: NDArray[Any] | list[NDArray[Any]] | pd.DataFrame | None = None,
        display_data: NDArray[Any] | list[NDArray[Any]] | pd.DataFrame | None = None,
        instance_names: list[str] | NDArray[Any] | None = None,
        feature_names: list[str] | NDArray[Any] | Any | None = None,
        output_names: list[str] | NDArray[Any] | str | None = None,
        output_indexes: Any | None = None,
        lower_bounds: NDArray[Any] | list[NDArray[Any]] | None = None,
        upper_bounds: NDArray[Any] | list[NDArray[Any]] | None = None,
        error_std: NDArray[Any] | list[NDArray[Any]] | None = None,
        main_effects: NDArray[Any] | list[NDArray[Any]] | None = None,
        hierarchical_values: NDArray[Any] | list[NDArray[Any]] | None = None,
        clustering: NDArray[Any] | None = None,
        compute_time: float | None = None,
    ) -> None:
        """Initialize a SHAP Explanation object.

        Parameters
        ----------
        values : array-like
            The SHAP values to explain.
        base_values : array-like, optional
            The base values (expected values) for the explanation.
        data : array-like, optional
            The input data that was explained.
        display_data : array-like, optional
            Alternative data to display instead of raw data.
        instance_names : list, optional
            Names for each instance/sample.
        feature_names : list, optional
            Names for each feature.
        output_names : list, optional
            Names for each output.
        output_indexes : optional
            Indexes for outputs.
        lower_bounds : array-like, optional
            Lower bounds for confidence intervals.
        upper_bounds : array-like, optional
            Upper bounds for confidence intervals.
        error_std : array-like, optional
            Standard error values.
        main_effects : array-like, optional
            Main effect values.
        hierarchical_values : array-like, optional
            Hierarchical SHAP values.
        clustering : array-like, optional
            Clustering information.
        compute_time : float, optional
            Time taken to compute the explanation.
        """
        ...

    # =================== Properties ===================

    @property
    def values(self) -> NDArray[Any]:
        """Pass-through from the underlying slicer object."""
        ...

    @values.setter
    def values(self, new_values: NDArray[Any]) -> None: ...
    @property
    def base_values(self) -> NDArray[Any] | float | int | None:
        """Pass-through from the underlying slicer object."""
        ...

    @base_values.setter
    def base_values(self, new_base_values: NDArray[Any] | float | int | None) -> None: ...
    @property
    def data(self) -> NDArray[Any] | None:
        """Pass-through from the underlying slicer object."""
        ...

    @data.setter
    def data(self, new_data: NDArray[Any] | None) -> None: ...
    @property
    def display_data(self) -> NDArray[Any] | None:
        """Pass-through from the underlying slicer object."""
        ...

    @display_data.setter
    def display_data(self, new_display_data: NDArray[Any] | pd.DataFrame | None) -> None: ...
    @property
    def instance_names(self) -> list[str] | NDArray[Any] | None:
        """Pass-through from the underlying slicer object."""
        ...

    @property
    def output_names(self) -> list[str] | NDArray[Any] | str | None:
        """Pass-through from the underlying slicer object."""
        ...

    @output_names.setter
    def output_names(self, new_output_names: list[str] | NDArray[Any] | str | None) -> None: ...
    @property
    def output_indexes(self) -> Any:
        """Pass-through from the underlying slicer object."""
        ...

    @property
    def feature_names(self) -> list[str] | NDArray[Any] | None:
        """Pass-through from the underlying slicer object."""
        ...

    @feature_names.setter
    def feature_names(self, new_feature_names: list[str] | NDArray[Any] | None) -> None: ...
    @property
    def lower_bounds(self) -> NDArray[Any] | None:
        """Pass-through from the underlying slicer object."""
        ...

    @property
    def upper_bounds(self) -> NDArray[Any] | None:
        """Pass-through from the underlying slicer object."""
        ...

    @property
    def error_std(self) -> NDArray[Any] | None:
        """Pass-through from the underlying slicer object."""
        ...

    @property
    def main_effects(self) -> NDArray[Any] | None:
        """Pass-through from the underlying slicer object."""
        ...

    @main_effects.setter
    def main_effects(self, new_main_effects: NDArray[Any] | None) -> None: ...
    @property
    def hierarchical_values(self) -> NDArray[Any] | None:
        """Pass-through from the underlying slicer object."""
        ...

    @hierarchical_values.setter
    def hierarchical_values(self, new_hierarchical_values: NDArray[Any] | None) -> None: ...
    @property
    def clustering(self) -> NDArray[Any] | None:
        """Pass-through from the underlying slicer object."""
        ...

    @clustering.setter
    def clustering(self, new_clustering: NDArray[Any] | None) -> None: ...
    @property
    def shape(self) -> tuple[int, ...]:
        """Compute the shape over potentially complex data nesting."""
        ...

    # =================== Operations ===================

    def __repr__(self) -> str:
        """Display some basic printable info, but not everything."""
        ...

    def __getitem__(self, item: Any) -> Explanation:
        """This adds support for OpChain indexing."""
        ...

    def __len__(self) -> int: ...
    def __copy__(self) -> Explanation: ...
    def __add__(self, other: Explanation | NDArray[Any] | float | int) -> Explanation: ...
    def __radd__(self, other: Explanation | NDArray[Any] | float | int) -> Explanation: ...
    def __sub__(self, other: Explanation | NDArray[Any] | float | int) -> Explanation: ...
    def __rsub__(self, other: Explanation | NDArray[Any] | float | int) -> Explanation: ...
    def __mul__(self, other: Explanation | NDArray[Any] | float | int) -> Explanation: ...
    def __rmul__(self, other: Explanation | NDArray[Any] | float | int) -> Explanation: ...
    def __truediv__(self, other: Explanation | NDArray[Any] | float | int) -> Explanation: ...

    # =================== Numpy-style methods ===================

    @property
    def abs(self) -> Explanation:
        """Element-wise absolute value."""
        ...

    @property
    def identity(self) -> Explanation:
        """A no-op that returns self."""
        ...

    @property
    def argsort(self) -> Explanation:
        """Numpy-style argsort."""
        ...

    @property
    def flip(self) -> Explanation:
        """Numpy-style flip."""
        ...

    def mean(self, axis: int) -> Explanation:
        """Numpy-style mean function."""
        ...

    def max(self, axis: int) -> Explanation:
        """Numpy-style max function."""
        ...

    def min(self, axis: int) -> Explanation:
        """Numpy-style min function."""
        ...

    def sum(self, axis: int | None = None, grouping: dict[str, str] | None = None) -> Explanation:
        """Numpy-style sum function.

        Parameters
        ----------
        axis : int, optional
            Axis along which to sum.
        grouping : dict, optional
            Feature grouping dictionary for grouping features before summing.
        """
        ...

    def percentile(self, q: float | int, axis: int | None = None) -> Explanation:
        """Compute percentiles along an axis.

        Parameters
        ----------
        q : float
            Percentile to compute (0-100).
        axis : int, optional
            Axis along which to compute percentiles.
        """
        ...

    def sample(self, max_samples: int, replace: bool = False, random_state: int = 0) -> Explanation:
        """Randomly samples the instances (rows) of the Explanation object.

        Parameters
        ----------
        max_samples : int
            The number of rows to sample. Note that if ``replace=False``, then
            fewer than max_samples will be drawn if ``len(explanation) < max_samples``.

        replace : bool
            Sample with or without replacement.

        random_state : int
            Random seed to use for sampling, defaults to 0.
        """
        ...

    def hclust(self, metric: str = "sqeuclidean", axis: int = 0) -> NDArray[Any]:
        """Computes an optimal leaf ordering sort order using hclustering.

        Parameters
        ----------
        metric : str
            A metric supported by scipy clustering. Defaults to "sqeuclidean".

        axis : int
            The axis to cluster along.
        """
        ...

    # =================== Utilities ===================

    def hstack(self, other: Explanation) -> Explanation:
        """Stack two explanations column-wise.

        Parameters
        ----------
        other : shap.Explanation
            The other Explanation object to stack with.

        Returns
        -------
        exp : shap.Explanation
            A new Explanation object representing the stacked explanations.
        """
        ...

    def cohorts(self, cohorts: int | list[int] | tuple[int] | NDArray[Any]) -> Cohorts:
        """Split this explanation into several cohorts.

        Parameters
        ----------
        cohorts : int or array
            If this is an integer then we auto build that many cohorts using a decision tree. If this is
            an array then we treat that as an array of cohort names/ids for each instance.

        Returns
        -------
        Cohorts object
        """
        ...

    def _apply_binary_operator(
        self, other: Explanation | NDArray[Any] | float | int, binary_op: Callable[[Any, Any], Any], op_name: str
    ) -> Explanation: ...
    def _numpy_func(self, fname: str, **kwargs: Any) -> Explanation: ...
    def _flatten_feature_names(self) -> dict[Any, Any]: ...
    def _use_data_as_feature_names(self) -> dict[Any, Any]: ...

class Cohorts:
    """A collection of :class:`.Explanation` objects, typically each explaining a cluster of similar samples.

    Examples
    --------
    A ``Cohorts`` object can be initialized in a variety of ways.

    By explicitly specifying the cohorts:

    >>> exp = Explanation(
    ...     values=np.random.uniform(low=-1, high=1, size=(500, 5)),
    ...     data=np.random.normal(loc=1, scale=3, size=(500, 5)),
    ...     feature_names=list("abcde"),
    ... )
    >>> cohorts = Cohorts(
    ...     col_a_neg=exp[exp[:, "a"].data < 0],
    ...     col_a_pos=exp[exp[:, "a"].data >= 0],
    ... )
    >>> cohorts
    <shap._explanation.Cohorts object with 2 cohorts of sizes: [(198, 5), (302, 5)]>

    Or using the :meth:`.Explanation.cohorts` method:

    >>> cohorts2 = exp.cohorts(3)
    >>> cohorts2
    <shap._explanation.Cohorts object with 3 cohorts of sizes: [(182, 5), (12, 5), (306, 5)]>

    Most of the :class:`.Explanation` interface is also exposed in ``Cohorts``. For example, to retrieve the
    SHAP values corresponding to column 'a' across all cohorts, you can use:

    >>> cohorts[..., 'a'].values
    <shap._explanation.Cohorts object with 2 cohorts of sizes: [(198,), (302,)]>

    To actually retrieve the values of a particular :class:`.Explanation`, you'll need to access it via the
    :meth:`.Cohorts.cohorts` property:

    >>> cohorts.cohorts["col_a_neg"][..., 'a'].values
    array([...])  # truncated
    """

    cohorts: dict[str, Explanation]
    _callables: dict[str, Callable]

    def __init__(self, **kwargs: Explanation) -> None:
        """Initialize cohorts with named Explanation objects."""
        ...

    @property
    def cohorts(self) -> dict[str, Explanation]:
        """Internal collection of cohorts, stored as a dictionary."""
        ...

    @cohorts.setter
    def cohorts(self, cval: dict[str, Explanation]) -> None: ...
    def __getitem__(self, item: Any) -> Cohorts: ...
    def __getattr__(self, name: str) -> Cohorts: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Cohorts:
        """Call the bound methods on the Explanation objects retrieved during attribute access."""
        ...

    def __repr__(self) -> str: ...

# =================== Helper functions ===================

def group_features(shap_values: Explanation, feature_map: dict[str, str]) -> Explanation:
    """Group features in a SHAP explanation according to a feature mapping.

    Parameters
    ----------
    shap_values : Explanation
        The SHAP explanation to group features for.
    feature_map : dict
        Mapping from original feature names to grouped feature names.

    Returns
    -------
    Explanation
        New explanation with grouped features.
    """
    ...

def compute_output_dims(values: Any, base_values: Any, data: Any, output_names: Any) -> tuple[int, ...]:
    """Uses the passed data to infer which dimensions correspond to the model's output."""
    ...

def is_1d(val: Any) -> bool:
    """Check if a value is 1-dimensional."""
    ...

def _compute_shape(x: Any) -> tuple[int | None, ...]:
    """Computes the shape of a generic object ``x``."""
    ...

def _auto_cohorts(shap_values: Explanation, max_cohorts: int) -> Cohorts:
    """This uses a DecisionTreeRegressor to build a group of cohorts with similar SHAP values."""
    ...

def list_wrap(x: Any) -> Any:
    """A helper to patch things since slicer doesn't handle arrays of arrays (it does handle lists of arrays)"""
    ...

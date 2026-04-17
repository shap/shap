"""Lower level layer for slicer.
Mom's spaghetti.
"""
# TODO: Consider boolean array indexing.

from typing import Any, AnyStr, Union, List, Tuple
from abc import abstractmethod
import numbers
from typing import Sequence


class AtomicSlicer:
    """Wrapping object that will unify slicing across data structures.

    What we support:
        Basic indexing (return references):
        - (start:stop:step) slicing
        - support ellipses
        Advanced indexing (return references):
        - integer array indexing

    Numpy Reference:
        Basic indexing (return views):
        - (start:stop:step) slicing
        - support ellipses and newaxis (alias for None)
        Advanced indexing (return copy):
        - integer array indexing, i.e. X[[1,2], [3,4]]
        - boolean array indexing
        - mixed array indexing (has integer array, ellipses, newaxis in same slice)
    """

    def __init__(self, o: Any, max_dim: Union[None, int, str] = "auto"):
        """Provides a consistent slicing API to the object provided.

        Args:
            o: Object to enable consistent slicing.
                Currently supports numpy dense arrays, recursive lists ending with list or numpy.
            max_dim: Max number of dimensions the wrapped object has.
                If set to "auto", max dimensions will be inferred. This comes at compute cost.
        """
        self.o = o
        self.max_dim: Union[None, int, str] = max_dim
        if self.max_dim == "auto":
            self.max_dim = UnifiedDataHandler.max_dim(o)

    def __repr__(self) -> str:
        """Override default repr for human readability.

        Returns:
            String to display.
        """
        return f"{self.__class__.__name__}({self.o.__repr__()})"

    def __getitem__(self, item: Any) -> Any:
        """Consistent slicing into wrapped object.

        Args:
            item: Slicing key of type integer or slice.

        Returns:
            Sliced object.

        Raises:
            ValueError: If slicing is not compatible with wrapped object.
        """
        assert isinstance(self.max_dim, int)

        # Turn item into tuple if not already.
        index_tup = unify_slice(item, self.max_dim)

        # Slice according to object type.
        return UnifiedDataHandler.slice(self.o, index_tup, self.max_dim)


def unify_slice(item: Any, max_dim: int, alias_lookup=None) -> Tuple:
    """Resolves aliases and ellipses in a slice item.

    Args:
        item: Slicing key that is passed to __getitem__.
        max_dim: Max dimension of object to be sliced.
        alias_lookup: AliasLookup structure.

    Returns:
        A tuple representation of the item.
    """
    item = _normalize_slice_key(item)
    index_tup = _normalize_subkey_types(item)
    index_tup = _handle_newaxis_ellipses(index_tup, max_dim)
    if alias_lookup:
        index_tup = _handle_aliases(index_tup, alias_lookup)
    return index_tup


def _normalize_subkey_types(index_tup: Tuple) -> Tuple:
    """Casts subkeys into basic types such as int.

    Args:
        key: Slicing key that is passed within __getitem__.

    Returns:
        Tuple with subkeys casted to basic types.
    """
    new_index_tup = []  # Gets casted to tuple at the end

    np_int_types = {
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    }
    for subkey in index_tup:
        if _safe_isinstance(subkey, "numpy", np_int_types):
            new_subkey = int(subkey)
        elif _safe_isinstance(subkey, "numpy", "ndarray"):
            if len(subkey.shape) == 1:
                new_subkey = subkey.tolist()
            else:
                raise ValueError(f"Cannot use array of shape {subkey.shape} as subkey.")
        else:
            new_subkey = subkey

        new_index_tup.append(new_subkey)
    return tuple(new_index_tup)


def _normalize_slice_key(key: Any) -> Tuple:
    """Normalizes slice key into always being a top-level tuple.

    Args:
        key: Slicing key that is passed within __getitem__.

    Returns:
        Expanded slice as a tuple.
    """
    if not isinstance(key, tuple):
        return (key,)
    else:
        return key


def _handle_newaxis_ellipses(index_tup: Tuple, max_dim: int) -> Tuple:
    """Expands newaxis and ellipses within a slice for simplification.
    This code is mostly adapted from: https://github.com/clbarnes/h5py_like/blob/master/h5py_like/shape_utils.py#L111

    Args:
        index_tup: Slicing key as a tuple.
        max_dim: Maximum number of dimensions in the respective sliceable object.

    Returns:
        Expanded slice as a tuple.
    """
    non_indexes = (None, Ellipsis)
    concrete_indices = sum(idx not in non_indexes for idx in index_tup)
    index_list: List[Any] = []
    # newaxis_at = []
    has_ellipsis = False
    int_count = 0
    for item in index_tup:
        if isinstance(item, numbers.Number):
            int_count += 1

        # NOTE: If we need locations of new axis, re-enable this.
        if item is None:  # pragma: no cover
            pass
            # newaxis_at.append(len(index_list) + len(newaxis_at) - int_count)
        elif item == Ellipsis:
            if has_ellipsis:  # pragma: no cover
                raise IndexError("an index can only have a single ellipsis ('...')")
            has_ellipsis = True
            initial_len = len(index_list)
            while len(index_list) + (concrete_indices - initial_len) < max_dim:
                index_list.append(slice(None))
        else:
            index_list.append(item)

    if len(index_list) > max_dim:  # pragma: no cover
        raise IndexError("too many indices for array")
    while len(index_list) < max_dim:
        index_list.append(slice(None))

    # return index_list, newaxis_at
    return tuple(index_list)


def _handle_aliases(index_tup: Tuple, alias_lookup) -> Tuple:
    new_index_tup = []

    def resolve(item, dim):
        if isinstance(item, slice):
            return item
        # Replace element if in alias lookup, otherwise use original.
        item = alias_lookup.get(dim, item, item)
        return item

    # Go through each element within the index and resolve if needed.
    for dim, item in enumerate(index_tup):
        if isinstance(item, list):
            new_item = []
            for sub_item in item:
                new_item.append(resolve(sub_item, dim))
        else:
            new_item = resolve(item, dim)
        new_index_tup.append(new_item)

    return tuple(new_index_tup)


class Tracked(AtomicSlicer):
    """Tracked defines an object that slicer wraps."""

    def __init__(self, o: Any, dim: Union[int, List, tuple, None, str] = "auto"):
        """Defines an object that will be wrapped by slicer.

        Args:
            o: Object that will be tracked for slicer.
            dim: Target dimension(s) slicer will index on for this object.
        """
        super().__init__(o)

        # Protected attribute that can be overriden.
        self._name: Union[str, None] = None

        # Place dim into coordinate form.
        if dim == "auto":
            assert isinstance(self.max_dim, int)
            self.dim = list(range(self.max_dim))
        elif dim is None:
            self.dim = []
        elif isinstance(dim, int):
            self.dim = [dim]
        elif isinstance(dim, list):
            self.dim = dim
        elif isinstance(dim, tuple):
            self.dim = list(dim)
        else:  # pragma: no cover
            raise ValueError(f"Cannot handle dim of type: {type(dim)}")


class Obj(Tracked):
    """An object that slicer wraps."""

    def __init__(self, o, dim="auto"):
        super().__init__(o, dim)


class Alias(Tracked):
    """Defines a tracked object as well as additional __getitem__ keys."""

    def __init__(self, o, dim):
        if not (isinstance(dim, int) or (isinstance(dim, (list, tuple)) and len(dim) <= 1)):  # pragma: no cover
            raise ValueError("Aliases must track a single dimension")
        super().__init__(o, dim)


class AliasLookup:
    def __init__(self, aliases):
        self._lookup = {}

        # Populate lookup and merge indexes.
        for _, alias in aliases.items():
            self.update(alias)

    def update(self, alias):
        if alias.dim is None or len(alias.dim) == 0:
            return

        dim = alias.dim[0]
        if dim not in self._lookup:
            self._lookup[dim] = {}

        dim_lookup = self._lookup[dim]
        # NOTE: Alias must be backed by either a list or dictionary.
        itr = enumerate(alias.o) if isinstance(alias.o, list) else alias.o.items()
        for i, x in itr:
            if x not in dim_lookup:
                dim_lookup[x] = set()
            dim_lookup[x].add(i)

    def delete(self, alias):
        """Delete an alias that exists from lookup"""
        dim = alias.dim[0]
        dim_lookup = self._lookup[dim]
        # NOTE: Alias must be backed by either a list or dictionary.
        itr = enumerate(alias.o) if isinstance(alias.o, list) else alias.o.items()
        for i, x in itr:
            del dim_lookup[x]

    def get(self, dim, target, default=None):
        if dim not in self._lookup:
            return default

        indexes = self._lookup[dim].get(target, None)
        if indexes is None:
            return default

        if len(indexes) == 1:
            return next(iter(indexes))
        else:
            return list(indexes)


def resolve_dim(slicer_index: Tuple, slicer_dim: List) -> List:
    """Extracts new dim after applying slicing index and maps it back to the original index list."""

    new_slicer_dim = []
    reduced_mask = []

    for _, curr_idx in enumerate(slicer_index):
        if isinstance(curr_idx, (tuple, list, slice)):
            reduced_mask.append(0)
        else:
            reduced_mask.append(1)

    for curr_dim in slicer_dim:
        if reduced_mask[curr_dim] == 0:
            new_slicer_dim.append(curr_dim - sum(reduced_mask[:curr_dim]))

    return new_slicer_dim


def reduced_o(tracked: Sequence[Tracked]) -> Union[List, Any]:
    os = [t.o for t in tracked]
    os = os[0] if len(os) == 1 else os
    return os


class BaseHandler:
    @classmethod
    @abstractmethod
    def head_slice(cls, o, index_tup, max_dim):
        raise NotImplementedError()  # pragma: no cover

    @classmethod
    @abstractmethod
    def tail_slice(cls, o, tail_index, max_dim, flatten=True):
        raise NotImplementedError()  # pragma: no cover

    @classmethod
    @abstractmethod
    def max_dim(cls, o):
        raise NotImplementedError()  # pragma: no cover

    @classmethod
    def default_alias(cls, o):
        return []


class SeriesHandler(BaseHandler):
    @classmethod
    def head_slice(cls, o, index_tup, max_dim):
        head_index = index_tup[0]
        is_element = True if isinstance(head_index, int) else False
        sliced_o = o.iloc[head_index]

        return is_element, sliced_o, 1

    @classmethod
    def tail_slice(cls, o, tail_index, max_dim, flatten=True):
        # NOTE: Series only has one dimension,
        #       call slicer again to end the recursion.
        return AtomicSlicer(o, max_dim=max_dim)[tail_index]

    @classmethod
    def max_dim(cls, o):
        return len(o.shape)

    @classmethod
    def default_alias(cls, o):
        index_alias = Alias(o.index.to_list(), 0)
        index_alias._name = "index"
        return [index_alias]


class DataFrameHandler(BaseHandler):
    @classmethod
    def head_slice(cls, o, index_tup, max_dim):
        # NOTE: At head slice, we know there are two fixed dimensions.
        cut_index = index_tup
        is_element = True if isinstance(cut_index[-1], int) else False
        sliced_o = o.iloc[cut_index]

        return is_element, sliced_o, 2

    @classmethod
    def tail_slice(cls, o, tail_index, max_dim, flatten=True):
        # NOTE: Dataframe has fixed dimensions,
        #       call slicer again to end the recursion.
        return AtomicSlicer(o, max_dim=max_dim)[tail_index]

    @classmethod
    def max_dim(cls, o):
        return len(o.shape)

    @classmethod
    def default_alias(cls, o):
        index_alias = Alias(o.index.to_list(), 0)
        index_alias._name = "index"
        column_alias = Alias(o.columns.to_list(), 1)
        column_alias._name = "columns"
        return [index_alias, column_alias]


class ArrayHandler(BaseHandler):
    @classmethod
    def head_slice(cls, o, index_tup, max_dim):
        # Check if head is string
        head_index, tail_index = index_tup[0], index_tup[1:]
        cut = 1

        for sub_index in tail_index:
            if isinstance(sub_index, str) or cut == len(o.shape):
                break
            cut += 1

        # Process native array dimensions
        cut_index = index_tup[:cut]
        is_element = any([True if isinstance(x, int) else False for x in cut_index])
        sliced_o = o[cut_index]

        return is_element, sliced_o, cut

    @classmethod
    def tail_slice(cls, o, tail_index, max_dim, flatten=True):
        if flatten:
            # NOTE: If we're dealing with a scipy matrix,
            #       we have to manually flatten it ourselves
            #       to keep consistent to the rest of slicer's API.
            if _safe_isinstance(o, "scipy.sparse.csc", "csc_matrix"):
                return AtomicSlicer(o.toarray().flatten(), max_dim=max_dim)[tail_index]
            elif _safe_isinstance(o, "scipy.sparse.csr", "csr_matrix"):
                return AtomicSlicer(o.toarray().flatten(), max_dim=max_dim)[tail_index]
            elif _safe_isinstance(o, "scipy.sparse.dok", "dok_matrix"):
                return AtomicSlicer(o.toarray().flatten(), max_dim=max_dim)[tail_index]
            elif _safe_isinstance(o, "scipy.sparse.lil", "lil_matrix"):
                return AtomicSlicer(o.toarray().flatten(), max_dim=max_dim)[tail_index]
            else:
                return AtomicSlicer(o, max_dim=max_dim)[tail_index]
        else:
            inner = [AtomicSlicer(e, max_dim=max_dim)[tail_index] for e in o]
            if _safe_isinstance(o, "numpy", "ndarray"):
                import numpy

                if len(inner) > 0 and hasattr(inner[0], "__len__"):
                    ragged = not all(len(x) == len(inner[0]) for x in inner)
                else:
                    ragged = False
                if ragged:
                    return numpy.array(inner, dtype=object)
                else:
                    return numpy.array(inner)
            elif _safe_isinstance(o, "torch", "Tensor"):
                import torch

                if len(inner) > 0 and isinstance(inner[0], torch.Tensor):
                    return torch.stack(inner)
                else:
                    return torch.tensor(inner)
            elif _safe_isinstance(o, "scipy.sparse.csc", "csc_matrix"):
                from scipy.sparse import vstack

                out = vstack(inner, format="csc")
                return out
            elif _safe_isinstance(o, "scipy.sparse.csr", "csr_matrix"):
                from scipy.sparse import vstack

                out = vstack(inner, format="csr")
                return out
            elif _safe_isinstance(o, "scipy.sparse.dok", "dok_matrix"):
                from scipy.sparse import vstack

                out = vstack(inner, format="dok")
                return out
            elif _safe_isinstance(o, "scipy.sparse.lil", "lil_matrix"):
                from scipy.sparse import vstack

                out = vstack(inner, format="lil")
                return out
            else:
                raise ValueError(f"Cannot handle type {type(o)}.")  # pragma: no cover

    @classmethod
    def max_dim(cls, o):
        if _safe_isinstance(o, "numpy", "ndarray") and o.dtype == "object":
            return max([UnifiedDataHandler.max_dim(x) for x in o], default=-1) + 1
        else:
            return len(o.shape)


class DictHandler(BaseHandler):
    @classmethod
    def head_slice(cls, o, index_tup, max_dim):
        head_index = index_tup[0]
        if isinstance(head_index, (tuple, list)) and len(index_tup) == 0:
            return False, o, 1

        if isinstance(head_index, (list, tuple)):
            return (
                False,
                {sub_index: AtomicSlicer(o, max_dim=max_dim)[sub_index] for sub_index in head_index},
                1,
            )
        elif isinstance(head_index, slice):
            if head_index == slice(None, None, None):
                return False, o, 1
            return False, o[head_index], 1
        else:
            return True, o[head_index], 1

    @classmethod
    def tail_slice(cls, o, tail_index, max_dim, flatten=True):
        if flatten:
            return AtomicSlicer(o, max_dim=max_dim)[tail_index]
        else:
            return {k: AtomicSlicer(e, max_dim=max_dim)[tail_index] for k, e in o.items()}

    @classmethod
    def max_dim(cls, o):
        return max([UnifiedDataHandler.max_dim(x) for x in o.values()], default=-1) + 1


class ListTupleHandler(BaseHandler):
    @classmethod
    def head_slice(cls, o, index_tup, max_dim):
        head_index = index_tup[0]
        if isinstance(head_index, (tuple, list)) and len(index_tup) == 0:
            return False, o, 1

        if isinstance(head_index, (list, tuple)):
            if len(head_index) == 0:
                return False, o, 1
            else:
                list_results: List[Any] = [AtomicSlicer(o, max_dim=max_dim)[sub_index] for sub_index in head_index]
                final_results = tuple(list_results) if isinstance(o, tuple) else list_results
                return False, final_results, 1
        elif isinstance(head_index, slice):
            return False, o[head_index], 1
        elif isinstance(head_index, int):
            return True, o[head_index], 1
        else:  # pragma: no cover
            raise ValueError(f"Invalid key {head_index} for {o}")

    @classmethod
    def tail_slice(cls, o, tail_index, max_dim, flatten=True):
        if flatten:
            return AtomicSlicer(o, max_dim=max_dim)[tail_index]
        else:
            list_results = [AtomicSlicer(e, max_dim=max_dim)[tail_index] for e in o]
            return tuple(list_results) if isinstance(o, tuple) else list_results

    @classmethod
    def max_dim(cls, o):
        return max([UnifiedDataHandler.max_dim(x) for x in o], default=-1) + 1


class UnifiedDataHandler:
    """Registry that maps types to their unified slice calls."""

    """ Class attribute that maps type to their unified slice calls."""
    type_map = {
        ("builtins", "list"): ListTupleHandler,
        ("builtins", "tuple"): ListTupleHandler,
        ("builtins", "dict"): DictHandler,
        ("torch", "Tensor"): ArrayHandler,
        ("numpy", "ndarray"): ArrayHandler,
        ("scipy.sparse.csc", "csc_matrix"): ArrayHandler,
        ("scipy.sparse.csr", "csr_matrix"): ArrayHandler,
        ("scipy.sparse.dok", "dok_matrix"): ArrayHandler,
        ("scipy.sparse.lil", "lil_matrix"): ArrayHandler,
        ("pandas.core.frame", "DataFrame"): DataFrameHandler,
        ("pandas.core.series", "Series"): SeriesHandler,
    }

    @classmethod
    def slice(cls, o, index_tup, max_dim):
        import numpy as np

        # NOTE: Unified handles base cases such as empty tuples, which
        #       specialized handlers do not.
        if isinstance(index_tup, (tuple, list)) and len(index_tup) == 0:
            return o

        # Slice as delegated by data handler.
        o_type = _type_name(o)

        if o_type[0].startswith("scipy.sparse"):
            return o[index_tup[:2]]

        if not isinstance(o, dict):
            ndim = getattr(o, "ndim", None)
            if ndim is None and isinstance(o, (list, tuple)):
                ndim = np.ndim(o)

            if ndim is not None and isinstance(index_tup, tuple) and len(index_tup) > ndim:
                index_tup = index_tup[:ndim]

        head_slice = cls.type_map[o_type].head_slice
        tail_slice = cls.type_map[o_type].tail_slice

        is_element, sliced_o, cut = head_slice(o, index_tup, max_dim)
        out = tail_slice(sliced_o, index_tup[cut:], max_dim - cut, is_element)
        return out

    @classmethod
    def max_dim(cls, o):
        o_type = _type_name(o)
        if o_type not in cls.type_map:
            return 0
        return cls.type_map[o_type].max_dim(o)

    @classmethod
    def default_alias(cls, o):
        o_type = _type_name(o)
        if o_type not in cls.type_map:
            return {}
        return cls.type_map[o_type].default_alias(o)


def _type_name(o: object) -> Tuple[str, str]:
    return _handle_module_aliases(o.__class__.__module__), o.__class__.__name__


def _safe_isinstance(o: object, module_name: str, type_name: Union[str, set, tuple]) -> bool:
    o_module, o_type = _type_name(o)
    if isinstance(type_name, str):
        return o_module == module_name and o_type == type_name
    else:
        return o_module == module_name and o_type in type_name


def _handle_module_aliases(module_name):
    # scipy modules such as "scipy.sparse.csc" were renamed to "scipy.sparse._csc" in v1.8
    # Standardise by removing underscores for compatibility with either name
    # Else just pass module name unchanged
    module_map = {
        "scipy.sparse._csc": "scipy.sparse.csc",
        "scipy.sparse._csr": "scipy.sparse.csr",
        "scipy.sparse._dok": "scipy.sparse.dok",
        "scipy.sparse._lil": "scipy.sparse.lil",
    }
    return module_map.get(module_name, module_name)


class Slicer:
    """Provides unified slicing to tensor-like objects."""

    _max_dim: int
    _anon: list[Any]
    _objects: dict[str, Any]
    _aliases: dict[str, Any]
    _alias_lookup: Union["AliasLookup", None]

    def __init__(self, *args, **kwargs):
        """Wraps objects in args and provides unified numpy-like slicing.

        Currently supports (with arbitrary nesting):

        - lists and tuples
        - dictionaries
        - numpy arrays
        - pandas dataframes and series
        - pytorch tensors

        Args:
            *args: Unnamed tensor-like objects.
            **kwargs: Named tensor-like objects.

        Examples:

            Basic anonymous slicing:

            >>> from slicer import Slicer as S
            >>> li = [[1, 2, 3], [4, 5, 6]]
            >>> S(li)[:, 0:2].o
            [[1, 2], [4, 5]]
            >>> di = {'x': [1, 2, 3], 'y': [4, 5, 6]}
            >>> S(di)[:, 0:2].o
            {'x': [1, 2], 'y': [4, 5]}

            Basic named slicing:

            >>> import pandas as pd
            >>> import numpy as np
            >>> df = pd.DataFrame({'A': [1, 3], 'B': [2, 4]})
            >>> ar = np.array([[5, 6], [7, 8]])
            >>> sliced = S(first=df, second=ar)[0, :]
            >>> sliced.first
            A    1
            B    2
            Name: 0, dtype: int64
            >>> sliced.second
            array([5, 6])

        """
        self.__class__._init_slicer(self, *args, **kwargs)

    @classmethod
    def from_slicer(cls, *args, **kwargs):
        """Alternative to SUPER SLICE
        Args:
            *args:
            **kwargs:

        Returns:

        """
        slicer_instance = cls.__new__(cls)
        cls._init_slicer(slicer_instance, *args, **kwargs)
        return slicer_instance

    @classmethod
    def _init_slicer(cls, slicer_instance, *args, **kwargs):
        # NOTE: Protected attributes.
        slicer_instance._max_dim = 0

        # NOTE: Private attributes.
        slicer_instance._anon = []
        slicer_instance._objects = {}
        slicer_instance._aliases = {}
        slicer_instance._alias_lookup = None

        # Go through unnamed objects / aliases
        slicer_instance.__setattr__("o", args)

        # Go through named objects / aliases
        for key, value in kwargs.items():
            slicer_instance.__setattr__(key, value)

        # Generate default aliases only if one object and no aliases exist
        objects_len = len(slicer_instance._objects)
        anon_len = len(slicer_instance._anon)
        aliases_len = len(slicer_instance._aliases)
        if ((objects_len == 1) ^ (anon_len == 1)) and aliases_len == 0:
            obj = None
            for _, t in slicer_instance._iter_tracked():
                obj = t
            if obj is not None:
                generated_aliases = UnifiedDataHandler.default_alias(obj.o)
                for generated_alias in generated_aliases:
                    slicer_instance.__setattr__(generated_alias._name, generated_alias)

    def __getitem__(self, item):
        index_tup = unify_slice(item, self._max_dim, self._alias_lookup)
        new_args = []
        new_kwargs = {}
        for name, tracked in self._iter_tracked(include_aliases=True):
            if len(tracked.dim) == 0:  # No slice on empty dim
                new_tracked = tracked
            else:
                index_slicer = AtomicSlicer(index_tup, max_dim=1)
                slicer_index = index_slicer[tracked.dim]
                sliced_o = tracked[slicer_index]
                sliced_dim = resolve_dim(index_tup, tracked.dim)

                new_tracked = tracked.__class__(sliced_o, sliced_dim)
                new_tracked._name = tracked._name

            if name == "o":
                new_args.append(new_tracked)
            else:
                new_kwargs[name] = new_tracked

        return self.__class__.from_slicer(*new_args, **new_kwargs)

    def __getattr__(self, item):
        """Override default getattr to return tracked attribute.

        Args:
            item: Name of tracked attribute.
        Returns:
            Corresponding object.
        """
        if item.startswith("_"):
            return super(Slicer, self).__getattribute__(item)

        if item == "o":
            return reduced_o(self._anon)
        else:
            tracked = self._objects.get(item, None)
            if tracked is None:
                tracked = self._aliases.get(item, None)

            if tracked is None:
                underlying_obj = reduced_o(self._anon)
                if hasattr(underlying_obj, item):
                    return getattr(underlying_obj, item)
                raise AttributeError(f"Attribute '{item}' does not exist.")

            return tracked.o

    def __setattr__(self, key, value):
        """Override default setattr to sync tracking of slicer.

        Args:
            key: Name of tracked attribute.
            value: Either an Obj, Alias or Python Object.
        """
        if key.startswith("_"):
            return super(Slicer, self).__setattr__(key, value)

        # Grab previous objects if they exist:
        old_obj = self._objects.get(key, None)
        old_alias = self._aliases.get(key, None)

        # For existing attributes, honor Alias status and dimension unless specified otherwise
        if getattr(self, key, None) is not None and key != "o":
            if not isinstance(value, Tracked):
                if old_obj:
                    value = Obj(value, dim=old_obj.dim)
                elif old_alias:
                    value = Alias(value, dim=old_alias.dim)

        if isinstance(value, Alias):
            value._name = key
            self._aliases[key] = value

            if old_obj:  # If object previously existed as an object, clean up all references.
                del self._objects[key]

            # Build lookup (for perf)
            if self._alias_lookup is None:
                self._alias_lookup = AliasLookup(self._aliases)
            else:
                if old_alias:
                    self._alias_lookup.delete(old_alias)
                self._alias_lookup.update(value)
        else:
            if key == "o":
                tracked = [Obj(x) if not isinstance(x, Obj) else x for x in value]
                self._anon = tracked
                for t in tracked:
                    self._update_max_dim(t)

                os = reduced_o(self._anon)
                super(Slicer, self).__setattr__(key, os)
            else:
                if old_alias:  # If object previously existed as an alias, clean up all references.
                    if self._alias_lookup is not None:
                        self._alias_lookup.delete(old_alias)
                    del self._aliases[key]

                value = Obj(value) if not isinstance(value, Obj) else value
                value._name = key
                self._objects[key] = value
                self._update_max_dim(value)
                super(Slicer, self).__setattr__(key, value.o)

    def __delattr__(self, item):
        """Override default delattr to remove tracked attribute.

        Args:
            item: Name of tracked attribute to delete.
        """
        if item.startswith("_"):
            return super(Slicer, self).__delattr__(item)

        # Sync private attributes that help track
        self._objects.pop(item, None)
        self._aliases.pop(item, None)
        if item == "o":
            self._anon.clear()

        # Recompute max_dim
        self._recompute_max_dim()

        # Recompute alias lookup
        # NOTE: This doesn't use diff-style deletes, but we don't care (not a perf target).
        self._alias_lookup = AliasLookup(self._aliases)

        # TODO: Mutate and check interactively what it does
        super(Slicer, self).__delattr__(item)

    def __repr__(self):
        """Override default repr for human readability.

        Returns:
            String to display.
        """
        orig = self.__dict__
        di = {}
        for key, value in orig.items():
            if not key.startswith("_"):
                di[key] = value
        return f"{self.__class__.__name__}({str(di)})"

    def _update_max_dim(self, tracked):
        self._max_dim = max(self._max_dim, max(tracked.dim, default=-1) + 1)

    def _iter_tracked(self, include_aliases=False):
        for tracked in self._anon:
            yield "o", tracked
        for name, tracked in self._objects.items():
            yield name, tracked
        if include_aliases:
            for name, tracked in self._aliases.items():
                yield name, tracked

    def _recompute_max_dim(self):
        self._max_dim = max([max(o.dim, default=-1) + 1 for _, o in self._iter_tracked()], default=0)

"""
Vendored from: https://github.com/non-maintained-slicer-repo/slicer
This is a unified, single-file version of the slicer library.
"""

import numbers
from abc import abstractmethod
from typing import Any


class AtomicSlicer:
    def __init__(self, o: Any, max_dim: None | int | str = "auto"):
        self.o = o
        self.max_dim: None | int | str = max_dim
        if self.max_dim == "auto":
            self.max_dim = UnifiedDataHandler.max_dim(o)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.o.__repr__()})"

    def __getitem__(self, item: Any) -> Any:
        index_tup = unify_slice(item, self.max_dim)
        return UnifiedDataHandler.slice(self.o, index_tup, self.max_dim)


def unify_slice(item: Any, max_dim: int, alias_lookup=None) -> tuple:
    item = _normalize_slice_key(item)
    index_tup = _normalize_subkey_types(item)
    index_tup = _handle_newaxis_ellipses(index_tup, max_dim)
    if alias_lookup:
        index_tup = _handle_aliases(index_tup, alias_lookup)
    return index_tup


def _normalize_subkey_types(index_tup: tuple) -> tuple:
    new_index_tup = []
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


def _normalize_slice_key(key: Any) -> tuple:
    if not isinstance(key, tuple):
        return (key,)
    else:
        return key


def _handle_newaxis_ellipses(index_tup: tuple, max_dim: int) -> tuple:
    non_indexes = (None, Ellipsis)
    concrete_indices = sum(idx not in non_indexes for idx in index_tup)
    index_list: list[Any] = []
    has_ellipsis = False
    int_count = 0
    for item in index_tup:
        if isinstance(item, numbers.Number):
            int_count += 1
        if item is None:
            pass
        elif item == Ellipsis:
            if has_ellipsis:
                raise IndexError("an index can only have a single ellipsis ('...')")
            has_ellipsis = True
            initial_len = len(index_list)
            while len(index_list) + (concrete_indices - initial_len) < max_dim:
                index_list.append(slice(None))
        else:
            index_list.append(item)

    if len(index_list) > max_dim:
        raise IndexError("too many indices for array")
    while len(index_list) < max_dim:
        index_list.append(slice(None))

    return tuple(index_list)


def _handle_aliases(index_tup: tuple, alias_lookup) -> tuple:
    new_index_tup = []

    def resolve(item, dim):
        if isinstance(item, slice):
            return item
        item = alias_lookup.get(dim, item, item)
        return item

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
    def __init__(self, o: Any, dim: int | list | tuple | None | str = "auto"):
        super().__init__(o)
        self._name: str | None = None
        if dim == "auto":
            self.dim = list(range(self.max_dim))
        elif dim is None:
            self.dim = []
        elif isinstance(dim, int):
            self.dim = [dim]
        elif isinstance(dim, list):
            self.dim = dim
        elif isinstance(dim, tuple):
            self.dim = list(dim)
        else:
            raise ValueError(f"Cannot handle dim of type: {type(dim)}")


class Obj(Tracked):
    def __init__(self, o, dim="auto"):
        super().__init__(o, dim)


class Alias(Tracked):
    def __init__(self, o, dim):
        if not (isinstance(dim, int) or (isinstance(dim, (list, tuple)) and len(dim) <= 1)):
            raise ValueError("Aliases must track a single dimension")
        super().__init__(o, dim)


class AliasLookup:
    def __init__(self, aliases):
        self._lookup = {}
        for _, alias in aliases.items():
            self.update(alias)

    def update(self, alias):
        if alias.dim is None or len(alias.dim) == 0:
            return

        dim = alias.dim[0]
        if dim not in self._lookup:
            self._lookup[dim] = {}

        dim_lookup = self._lookup[dim]
        itr = enumerate(alias.o) if isinstance(alias.o, list) else alias.o.items()
        for i, x in itr:
            if x not in dim_lookup:
                dim_lookup[x] = set()
            dim_lookup[x].add(i)

    def delete(self, alias):
        dim = alias.dim[0]
        dim_lookup = self._lookup[dim]
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


def resolve_dim(slicer_index: tuple, slicer_dim: list) -> list:
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


def reduced_o(tracked: list[Tracked]) -> list | Any:
    os = [t.o for t in tracked]
    os = os[0] if len(os) == 1 else os
    return os


class BaseHandler:
    @classmethod
    @abstractmethod
    def head_slice(cls, o, index_tup, max_dim):
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def tail_slice(cls, o, tail_index, max_dim, flatten=True):
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def max_dim(cls, o):
        raise NotImplementedError()

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
        cut_index = index_tup
        is_element = True if isinstance(cut_index[-1], int) else False
        sliced_o = o.iloc[cut_index]
        return is_element, sliced_o, 2

    @classmethod
    def tail_slice(cls, o, tail_index, max_dim, flatten=True):
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
        tail_index = index_tup[1:]
        cut = 1

        for sub_index in tail_index:
            if isinstance(sub_index, str) or cut == len(o.shape):
                break
            cut += 1

        cut_index = index_tup[:cut]
        is_element = any([True if isinstance(x, int) else False for x in cut_index])
        sliced_o = o[cut_index]

        return is_element, sliced_o, cut

    @classmethod
    def tail_slice(cls, o, tail_index, max_dim, flatten=True):
        if flatten:
            if (
                _safe_isinstance(o, "scipy.sparse.csc", "csc_matrix")
                or _safe_isinstance(o, "scipy.sparse.csr", "csr_matrix")
                or _safe_isinstance(o, "scipy.sparse.dok", "dok_matrix")
                or _safe_isinstance(o, "scipy.sparse.lil", "lil_matrix")
            ):
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

                return vstack(inner, format="csc")
            elif _safe_isinstance(o, "scipy.sparse.csr", "csr_matrix"):
                from scipy.sparse import vstack

                return vstack(inner, format="csr")
            elif _safe_isinstance(o, "scipy.sparse.dok", "dok_matrix"):
                from scipy.sparse import vstack

                return vstack(inner, format="dok")
            elif _safe_isinstance(o, "scipy.sparse.lil", "lil_matrix"):
                from scipy.sparse import vstack

                return vstack(inner, format="lil")
            else:
                raise ValueError(f"Cannot handle type {type(o)}.")

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
                results = [AtomicSlicer(o, max_dim=max_dim)[sub_index] for sub_index in head_index]
                results = tuple(results) if isinstance(o, tuple) else results
                return False, results, 1
        elif isinstance(head_index, slice):
            return False, o[head_index], 1
        elif isinstance(head_index, int):
            return True, o[head_index], 1
        else:
            raise ValueError(f"Invalid key {head_index} for {o}")

    @classmethod
    def tail_slice(cls, o, tail_index, max_dim, flatten=True):
        if flatten:
            return AtomicSlicer(o, max_dim=max_dim)[tail_index]
        else:
            results = [AtomicSlicer(e, max_dim=max_dim)[tail_index] for e in o]
            return tuple(results) if isinstance(o, tuple) else results

    @classmethod
    def max_dim(cls, o):
        return max([UnifiedDataHandler.max_dim(x) for x in o], default=-1) + 1


class UnifiedDataHandler:
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
        if isinstance(index_tup, (tuple, list)) and len(index_tup) == 0:
            return o

        o_type = _type_name(o)
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


def _type_name(o: object) -> tuple[str, str]:
    return _handle_module_aliases(o.__class__.__module__), o.__class__.__name__


def _safe_isinstance(o: object, module_name: str, type_name: str | set | tuple) -> bool:
    o_module, o_type = _type_name(o)
    if isinstance(type_name, str):
        return o_module == module_name and o_type == type_name
    else:
        return o_module == module_name and o_type in type_name


def _handle_module_aliases(module_name):
    module_map = {
        "scipy.sparse._csc": "scipy.sparse.csc",
        "scipy.sparse._csr": "scipy.sparse.csr",
        "scipy.sparse._dok": "scipy.sparse.dok",
        "scipy.sparse._lil": "scipy.sparse.lil",
    }
    return module_map.get(module_name, module_name)


class Slicer:
    """Provides unified slicing to tensor-like objects."""

    def __init__(self, *args, **kwargs):
        self.__class__._init_slicer(self, *args, **kwargs)

    @classmethod
    def from_slicer(cls, *args, **kwargs):
        slicer_instance = cls.__new__(cls)
        cls._init_slicer(slicer_instance, *args, **kwargs)
        return slicer_instance

    @classmethod
    def _init_slicer(cls, slicer_instance, *args, **kwargs):
        slicer_instance._max_dim = 0  # type: int
        slicer_instance._anon = []  # type: List[Tracked]
        slicer_instance._objects = {}  # type: dict
        slicer_instance._aliases = {}  # type: dict
        slicer_instance._alias_lookup = None  # type: Union[AliasLookup, None]

        slicer_instance.__setattr__("o", args)

        for key, value in kwargs.items():
            slicer_instance.__setattr__(key, value)

        objects_len = len(slicer_instance._objects)
        anon_len = len(slicer_instance._anon)
        aliases_len = len(slicer_instance._aliases)
        if ((objects_len == 1) ^ (anon_len == 1)) and aliases_len == 0:
            obj = None
            for _, t in slicer_instance._iter_tracked():
                obj = t

            generated_aliases = UnifiedDataHandler.default_alias(obj.o)
            for generated_alias in generated_aliases:
                slicer_instance.__setattr__(generated_alias._name, generated_alias)

    def __getitem__(self, item):
        index_tup = unify_slice(item, self._max_dim, self._alias_lookup)
        new_args = []
        new_kwargs = {}
        for name, tracked in self._iter_tracked(include_aliases=True):
            if len(tracked.dim) == 0:
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
        if item.startswith("_"):
            return object.__getattribute__(self, item)

        if item == "o":
            return reduced_o(self._anon)
        else:
            tracked = self._objects.get(item, None)
            if tracked is None:
                tracked = self._aliases.get(item, None)

            if tracked is None:
                raise AttributeError(f"Attribute '{item}' does not exist.")

            return tracked.o

    def __setattr__(self, key, value):
        if key.startswith("_"):
            return super().__setattr__(key, value)

        old_obj = self._objects.get(key, None)
        old_alias = self._aliases.get(key, None)

        if getattr(self, key, None) is not None and key != "o":
            if not isinstance(value, Tracked):
                if old_obj:
                    value = Obj(value, dim=old_obj.dim)
                elif old_alias:
                    value = Alias(value, dim=old_alias.dim)

        if isinstance(value, Alias):
            value._name = key
            self._aliases[key] = value

            if old_obj:
                del self._objects[key]

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
                super().__setattr__(key, os)
            else:
                if old_alias:
                    self._alias_lookup.delete(old_alias)
                    del self._aliases[key]

                value = Obj(value) if not isinstance(value, Obj) else value
                value._name = key
                self._objects[key] = value
                self._update_max_dim(value)
                super().__setattr__(key, value.o)

    def __delattr__(self, item):
        if item.startswith("_"):
            return super().__delattr__(item)

        self._objects.pop(item, None)
        self._aliases.pop(item, None)
        if item == "o":
            self._anon.clear()

        self._recompute_max_dim()
        self._alias_lookup = AliasLookup(self._aliases)
        super().__delattr__(item)

    def __repr__(self):
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

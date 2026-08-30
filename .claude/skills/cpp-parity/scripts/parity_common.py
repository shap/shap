"""Shared (de)serialization for C++/numba parity fixtures.

A fixture is a single observed call of the migrated Python-level entry point:
its arguments and its return value, canonicalized into plain arrays so the same
file can be replayed on any branch.

Extend ``encode``/``decode`` when a migration's entry point takes or returns a
type that is not covered here -- everything else in the harness is generic.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np


def _sparse(obj):
    """Return obj if it is a scipy sparse matrix, else None (scipy stays optional)."""
    mod = type(obj).__module__ or ""
    if mod.startswith("scipy.sparse"):
        return obj
    return None


def encode(name: str, obj) -> dict[str, np.ndarray]:
    """Flatten one value into a dict of arrays with an ``npz``-safe key prefix."""
    out: dict[str, np.ndarray] = {}
    if isinstance(obj, np.ndarray):
        out[f"{name}/kind"] = np.array("ndarray")
        out[f"{name}/value"] = obj
        out[f"{name}/dtype"] = np.array(str(obj.dtype))
    elif _sparse(obj) is not None:
        csr = obj.tocsr()
        out[f"{name}/kind"] = np.array("sparse_csr")
        out[f"{name}/data"] = csr.data
        out[f"{name}/indices"] = csr.indices
        out[f"{name}/indptr"] = csr.indptr
        out[f"{name}/shape"] = np.asarray(csr.shape)
    elif isinstance(obj, (list, tuple)):
        out[f"{name}/kind"] = np.array("sequence")
        out[f"{name}/len"] = np.array(len(obj))
        for i, item in enumerate(obj):
            out.update(encode(f"{name}/{i}", item))
    elif obj is None or isinstance(obj, (bool, int, float, str, np.generic)):
        out[f"{name}/kind"] = np.array("scalar")
        out[f"{name}/value"] = np.array(json.dumps(_jsonable(obj)))
    else:
        out[f"{name}/kind"] = np.array("unsupported")
        out[f"{name}/value"] = np.array(f"{type(obj).__module__}.{type(obj).__qualname__}")
    return out


def _jsonable(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def decode(name: str, blob) -> object:
    kind = str(blob[f"{name}/kind"])
    if kind == "ndarray":
        return blob[f"{name}/value"]
    if kind == "sparse_csr":
        import scipy.sparse

        return scipy.sparse.csr_matrix(
            (blob[f"{name}/data"], blob[f"{name}/indices"], blob[f"{name}/indptr"]),
            shape=tuple(blob[f"{name}/shape"].tolist()),
        )
    if kind == "sequence":
        return [decode(f"{name}/{i}", blob) for i in range(int(blob[f"{name}/len"]))]
    if kind == "scalar":
        return json.loads(str(blob[f"{name}/value"]))
    raise TypeError(f"cannot decode fixture entry {name!r} of kind {kind!r}")


def compare(expected, actual) -> list[str]:
    """Return a list of human-readable differences; empty list means exact parity."""
    diffs: list[str] = []
    if _sparse(expected) is not None or _sparse(actual) is not None:
        if _sparse(expected) is None or _sparse(actual) is None:
            return [f"type mismatch: {type(expected)} vs {type(actual)}"]
        e, a = expected.tocsr(), actual.tocsr()
        if e.shape != a.shape:
            return [f"shape {e.shape} != {a.shape}"]
        # compare densely: CSR internals may differ (sorting, explicit zeros)
        # while the represented matrix is identical
        ed, ad = np.asarray(e.todense()), np.asarray(a.todense())
        if ed.dtype != ad.dtype:
            diffs.append(f"dtype {ed.dtype} != {ad.dtype}")
        if not np.array_equal(ed, ad):
            n = int((ed != ad).sum())
            diffs.append(f"{n} of {ed.size} dense entries differ")
        if not np.array_equal(e.indptr, a.indptr):
            diffs.append("indptr differs")
        if not np.array_equal(e.indices, a.indices):
            diffs.append("indices differ (same dense value only if row sets are permuted)")
        return diffs
    if isinstance(expected, np.ndarray) or isinstance(actual, np.ndarray):
        e, a = np.asarray(expected), np.asarray(actual)
        if e.shape != a.shape:
            return [f"shape {e.shape} != {a.shape}"]
        if e.dtype != a.dtype:
            diffs.append(f"dtype {e.dtype} != {a.dtype}")
        if not np.array_equal(e, a, equal_nan=e.dtype.kind == "f"):
            diffs.append(f"{int((e != a).sum())} of {e.size} entries differ")
        return diffs
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"length {len(expected)} != {len(actual)}"]
        for i, (e, a) in enumerate(zip(expected, actual)):
            diffs += [f"[{i}] {d}" for d in compare(e, a)]
        return diffs
    if expected != actual:
        diffs.append(f"{expected!r} != {actual!r}")
    return diffs


def fingerprint(args: tuple, kwargs: dict) -> str:
    """Stable content hash of a call's inputs, used to dedupe fixtures."""
    h = hashlib.sha256()
    for obj in list(args) + [kwargs[k] for k in sorted(kwargs)]:
        if isinstance(obj, np.ndarray):
            h.update(str(obj.dtype).encode())
            h.update(str(obj.shape).encode())
            h.update(np.ascontiguousarray(obj).tobytes())
        elif _sparse(obj) is not None:
            csr = obj.tocsr()
            for part in (csr.data, csr.indices, csr.indptr, np.asarray(csr.shape)):
                h.update(np.ascontiguousarray(part).tobytes())
        else:
            h.update(repr(obj).encode())
    return h.hexdigest()[:16]


def describe_dims(args: tuple, kwargs: dict) -> str:
    """Shape signature of a call, e.g. ``2d(3,4)+scalar`` -- the coverage key."""
    parts = []
    for obj in list(args) + [kwargs[k] for k in sorted(kwargs)]:
        if isinstance(obj, np.ndarray):
            parts.append(f"{obj.ndim}d{tuple(obj.shape)}:{obj.dtype}")
        elif _sparse(obj) is not None:
            parts.append(f"sparse{tuple(obj.shape)}")
        else:
            parts.append("scalar")
    return "+".join(parts)

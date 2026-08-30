"""Pytest plugin: record calls of shap.utils._masked_model:_build_fixed_output.

Target-specific variant of ``parity_capture``: the entry point returns None and
mutates ``averaged_outs``/``last_outs`` in place, and takes a ``link``
*function* argument. So we snapshot the two output arrays before the call,
store the link by name (resolvable via ``shap.links`` on any branch), and store
the post-call state of both mutated arrays as the "result".

Canonical fixture layout (positional args, in order):
    arg0 averaged_outs   (pre-call copy)
    arg1 last_outs       (pre-call copy)
    arg2 outputs
    arg3 batch_positions
    arg4 varying_rows
    arg5 num_varying_rows
    arg6 link name       (str, e.g. "identity"/"logit")
    arg7 linearizing_weights (ndarray or None)
    result = [averaged_outs post-call, last_outs post-call]

Env: PARITY_OUT (required), PARITY_LABEL, PARITY_MAX_BYTES. PARITY_TARGET is
implied and recorded in the manifest for the replay script.
"""

from __future__ import annotations

import functools
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import parity_common as pc

_TARGET = "shap.utils._masked_model:_build_fixed_output"
_OUT = os.environ.get("PARITY_OUT", "")
_LABEL = os.environ.get("PARITY_LABEL", "")
_MAX_BYTES = int(os.environ.get("PARITY_MAX_BYTES", 20 * 1024 * 1024))
# per-row explainer loops produce a unique input per call; cap how many
# fixtures are *stored* per (dims, link, weighted) class, but keep counting
_MAX_PER_SIG = int(os.environ.get("PARITY_MAX_PER_SIG", 20))

_current_test = "<outside-test>"
_hits: dict[str, int] = {}
_cases: dict[str, dict] = {}
_sig_counts: dict[tuple, int] = {}
_bindings = 0


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def _link_name(link) -> str:
    name = getattr(link, "__name__", None)
    if name is None:  # numba dispatcher fallback
        name = getattr(getattr(link, "py_func", None), "__name__", None)
    if name is None:
        raise TypeError(f"cannot canonicalize link {link!r}")
    return name


def _record(canon_args, result) -> None:
    _hits[_current_test] = _hits.get(_current_test, 0) + 1
    fp = pc.fingerprint(tuple(canon_args), {})
    case = _cases.get(fp)
    if case is None:
        sig = (pc.describe_dims(tuple(canon_args), {}), canon_args[6], canon_args[7] is not None)
        _sig_counts[sig] = _sig_counts.get(sig, 0) + 1
        if _sig_counts[sig] > _MAX_PER_SIG:
            return
        size = sum(int(getattr(a, "nbytes", 0) or 0) for a in canon_args)
        case = {
            "fingerprint": fp,
            "dims": pc.describe_dims(tuple(canon_args), {}),
            "link": canon_args[6],
            "weighted": canon_args[7] is not None,
            "input_nbytes": size,
            "calls": 0,
            "tests": [],
            "file": f"case_{fp}.npz",
        }
        if size > _MAX_BYTES:
            case["file"] = None
            case["skipped_reason"] = f"input {size} bytes exceeds PARITY_MAX_BYTES={_MAX_BYTES}"
        else:
            blob: dict[str, np.ndarray] = {"n_args": np.array(len(canon_args))}
            for i, a in enumerate(canon_args):
                blob.update(pc.encode(f"arg{i}", a))
            blob["kwarg_names"] = np.array(json.dumps([]))
            blob.update(pc.encode("result", result))
            np.savez_compressed(Path(_OUT) / case["file"], **blob)
        _cases[fp] = case
    case["calls"] += 1
    if _current_test not in case["tests"]:
        case["tests"].append(_current_test)


def pytest_configure(config) -> None:
    global _bindings
    if not _OUT:
        raise SystemExit("parity_capture_bfo requires PARITY_OUT")
    Path(_OUT).mkdir(parents=True, exist_ok=True)

    mod_name, _, func_name = _TARGET.partition(":")
    module = importlib.import_module(mod_name)
    original = getattr(module, func_name)

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        try:
            assert not kwargs, f"unexpected kwargs: {sorted(kwargs)}"
            (averaged, last, outputs, bp, vr, nvr, link, lw) = args
            canon = [
                np.array(averaged, copy=True),
                np.array(last, copy=True),
                np.array(outputs, copy=True),
                np.array(bp, copy=True),
                np.array(vr, copy=True),
                np.array(nvr, copy=True),
                _link_name(link),
                None if lw is None else np.array(lw, copy=True),
            ]
        except Exception as exc:
            print(f"[parity_capture_bfo] failed to canonicalize: {exc!r}", file=sys.stderr)
            return original(*args, **kwargs)
        result = original(*args, **kwargs)
        assert result is None, "expected in-place target returning None"
        try:
            _record(canon, [np.array(args[0], copy=True), np.array(args[1], copy=True)])
        except Exception as exc:  # never let capture break the test run
            print(f"[parity_capture_bfo] failed to record call: {exc!r}", file=sys.stderr)
        return result

    wrapper.__parity_original__ = original

    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        for attr, value in list(vars(mod).items() if hasattr(mod, "__dict__") else []):
            if value is original:
                try:
                    setattr(mod, attr, wrapper)
                    _bindings += 1
                except Exception:
                    pass
    if _bindings == 0:
        raise SystemExit(f"parity_capture_bfo: found no live binding of {_TARGET}")


def pytest_runtest_setup(item) -> None:
    global _current_test
    _current_test = item.nodeid
    if os.environ.get("SHAP_CPP_TRACE"):
        print(f"[TEST] {item.nodeid}", file=sys.stderr, flush=True)


def pytest_sessionfinish(session, exitstatus) -> None:
    if not _OUT:
        return
    manifest = {
        "target": _TARGET,
        "adapter": "bfo",
        "label": _LABEL,
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "pytest_exitstatus": int(exitstatus),
        "bindings_patched": _bindings,
        "total_calls": sum(_hits.values()),
        "tests_that_hit": dict(sorted(_hits.items())),
        "dim_signatures": sorted({c["dims"] for c in _cases.values()}),
        "max_stored_per_class": _MAX_PER_SIG,
        "input_classes": {
            f"{dims} link={link} weighted={weighted}": n
            for (dims, link, weighted), n in sorted(_sig_counts.items())
        },
        "cases": [_cases[k] for k in sorted(_cases, key=lambda k: _cases[k]["dims"])],
    }
    (Path(_OUT) / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"\n[parity_capture_bfo] {manifest['total_calls']} calls from "
        f"{len(_hits)} tests, {len(_cases)} unique inputs -> {_OUT}",
        file=sys.stderr,
    )

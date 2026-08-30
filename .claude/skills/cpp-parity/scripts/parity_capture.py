"""Pytest plugin: record every call of a migrated entry point during a test run.

Usage (from the repo root, with the project's venv):

    PARITY_TARGET=shap.utils._masked_model:make_masks \
    PARITY_OUT=bugs/data/make_masks \
    .venv/bin/python -m pytest tests/... -p parity_capture

Add ``.claude/skills/cpp-parity/scripts`` to ``PYTHONPATH`` so ``-p
parity_capture`` resolves. Writes to ``PARITY_OUT``:

* ``case_<fingerprint>.npz`` -- one per unique input, holding args + return value
* ``manifest.json``          -- target, per-test hit counts, per-case dim signature

Environment:
    PARITY_TARGET     ``module:qualname`` of the Python function to wrap (required)
    PARITY_OUT        output directory (required)
    PARITY_LABEL      free-form label stored in the manifest, e.g. a branch name
    PARITY_MAX_BYTES  skip storing arrays for cases above this size (default 20 MiB)
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

_TARGET = os.environ.get("PARITY_TARGET", "")
_OUT = os.environ.get("PARITY_OUT", "")
_LABEL = os.environ.get("PARITY_LABEL", "")
_MAX_BYTES = int(os.environ.get("PARITY_MAX_BYTES", 20 * 1024 * 1024))

_current_test = "<outside-test>"
_hits: dict[str, int] = {}
_cases: dict[str, dict] = {}
_skipped: list[dict] = []
_bindings = 0


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return ""


def _nbytes(args, kwargs) -> int:
    total = 0
    for obj in list(args) + list(kwargs.values()):
        total += int(getattr(obj, "nbytes", 0) or 0)
    return total


def _record(args, kwargs, result) -> None:
    _hits[_current_test] = _hits.get(_current_test, 0) + 1
    fp = pc.fingerprint(args, kwargs)
    case = _cases.get(fp)
    if case is None:
        size = _nbytes(args, kwargs)
        case = {
            "fingerprint": fp,
            "dims": pc.describe_dims(args, kwargs),
            "input_nbytes": size,
            "calls": 0,
            "tests": [],
            "file": f"case_{fp}.npz",
        }
        if size > _MAX_BYTES:
            case["file"] = None
            case["skipped_reason"] = f"input {size} bytes exceeds PARITY_MAX_BYTES={_MAX_BYTES}"
            _skipped.append({"fingerprint": fp, "dims": case["dims"], "nbytes": size})
        else:
            blob: dict[str, np.ndarray] = {"n_args": np.array(len(args))}
            for i, a in enumerate(args):
                blob.update(pc.encode(f"arg{i}", a))
            blob["kwarg_names"] = np.array(json.dumps(sorted(kwargs)))
            for k in sorted(kwargs):
                blob.update(pc.encode(f"kwarg_{k}", kwargs[k]))
            blob.update(pc.encode("result", result))
            np.savez_compressed(Path(_OUT) / case["file"], **blob)
        _cases[fp] = case
    case["calls"] += 1
    if _current_test not in case["tests"]:
        case["tests"].append(_current_test)


def pytest_configure(config) -> None:
    global _bindings
    if not _TARGET or not _OUT:
        raise SystemExit("parity_capture requires PARITY_TARGET and PARITY_OUT")
    Path(_OUT).mkdir(parents=True, exist_ok=True)

    mod_name, _, func_name = _TARGET.partition(":")
    module = importlib.import_module(mod_name)
    original = getattr(module, func_name)

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        result = original(*args, **kwargs)
        try:
            _record(args, kwargs, result)
        except Exception as exc:  # never let capture break the test run
            print(f"[parity_capture] failed to record call: {exc!r}", file=sys.stderr)
        return result

    wrapper.__parity_original__ = original

    # ``from x import f`` copies the reference, so rebind every module that
    # already holds the original object, not just the defining module.
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
        raise SystemExit(f"parity_capture: found no live binding of {_TARGET}")


def pytest_runtest_setup(item) -> None:
    global _current_test
    _current_test = item.nodeid
    if os.environ.get("SHAP_CPP_TRACE"):
        # interleave with the C++ trace on the same stream (run pytest with -s)
        # so native prints can be attributed to a test by ordering alone
        print(f"[TEST] {item.nodeid}", file=sys.stderr, flush=True)


def pytest_sessionfinish(session, exitstatus) -> None:
    if not _OUT:
        return
    manifest = {
        "target": _TARGET,
        "label": _LABEL,
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "pytest_exitstatus": int(exitstatus),
        "bindings_patched": _bindings,
        "total_calls": sum(_hits.values()),
        "tests_that_hit": dict(sorted(_hits.items())),
        "dim_signatures": sorted({c["dims"] for c in _cases.values()}),
        "cases": [_cases[k] for k in sorted(_cases, key=lambda k: _cases[k]["dims"])],
        "skipped_large_cases": _skipped,
    }
    (Path(_OUT) / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"\n[parity_capture] {manifest['total_calls']} calls from "
        f"{len(_hits)} tests, {len(_cases)} unique inputs -> {_OUT}",
        file=sys.stderr,
    )

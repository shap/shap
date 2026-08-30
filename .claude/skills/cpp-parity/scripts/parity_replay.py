"""Replay captured fixtures against the currently checked-out implementation.

Run it on the migration branch (should trivially pass) and then on the baseline
branch (proves the C++ rewrite matches the numba original):

    PYTHONPATH=.claude/skills/cpp-parity/scripts \
    .venv/bin/python .claude/skills/cpp-parity/scripts/parity_replay.py \
        --dir bugs/data/make_masks

Exit code is 0 only if every fixture reproduces exactly. ``--report`` writes the
verdict as JSON next to the fixtures.
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import parity_common as pc


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return ""


def _load_call(blob):
    args = [pc.decode(f"arg{i}", blob) for i in range(int(blob["n_args"]))]
    kwargs = {k: pc.decode(f"kwarg_{k}", blob) for k in json.loads(str(blob["kwarg_names"]))}
    return args, kwargs, pc.decode("result", blob)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, help="fixture directory written by parity_capture")
    ap.add_argument("--target", help="module:qualname to call (default: manifest target)")
    ap.add_argument("--report", metavar="NAME", help="write JSON verdict to <dir>/<NAME>")
    args = ap.parse_args()

    fixtures = Path(args.dir)
    manifest = json.loads((fixtures / "manifest.json").read_text())
    target = args.target or manifest["target"]
    mod_name, _, func_name = target.partition(":")
    func = getattr(importlib.import_module(mod_name), func_name)

    branch, commit = _git("rev-parse", "--abbrev-ref", "HEAD"), _git("rev-parse", "--short", "HEAD")
    print(f"replaying {len(manifest['cases'])} case(s) of {target}")
    print(f"  fixtures captured on: {manifest['git_branch']} @ {manifest['git_commit']}")
    print(f"  replaying on:         {branch} @ {commit}")
    print(f"  implementation:       {getattr(func, '__module__', '?')}.{func_name}")

    results, failures = [], 0
    for case in manifest["cases"]:
        if not case.get("file"):
            results.append({**case, "status": "SKIPPED", "diffs": [case.get("skipped_reason", "")]})
            print(f"  SKIP  {case['dims']}: {case.get('skipped_reason')}")
            continue
        with np.load(fixtures / case["file"], allow_pickle=False) as blob:
            call_args, call_kwargs, expected = _load_call(blob)
        try:
            actual = func(*call_args, **call_kwargs)
            diffs = pc.compare(expected, actual)
        except Exception as exc:
            diffs = [f"raised {type(exc).__name__}: {exc}"]
        status = "PASS" if not diffs else "FAIL"
        failures += status == "FAIL"
        results.append({**case, "status": status, "diffs": diffs})
        print(f"  {status:5s} {case['dims']}  ({case['calls']} call(s), {len(case['tests'])} test(s))")
        for d in diffs:
            print(f"        -> {d}")

    dims = sorted({c["dims"] for c in manifest["cases"]})
    ndims = sorted({p.split("d(")[0] for c in dims for p in c.split("+") if "d(" in p})
    print(f"\ndim signatures: {len(dims)}   array ndims exercised: {', '.join(ndims) or 'none'}")
    verdict = "PARITY CONFIRMED" if failures == 0 else f"PARITY BROKEN ({failures} case(s))"
    print(verdict)

    if args.report:
        (fixtures / args.report).write_text(
            json.dumps(
                {
                    "target": target,
                    "captured_on": f"{manifest['git_branch']}@{manifest['git_commit']}",
                    "replayed_on": f"{branch}@{commit}",
                    "implementation_module": getattr(func, "__module__", None),
                    "verdict": verdict,
                    "failures": failures,
                    "dim_signatures": dims,
                    "cases": results,
                },
                indent=2,
            )
            + "\n"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

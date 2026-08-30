"""Replay _build_fixed_output fixtures captured by parity_capture_bfo.

The target mutates ``averaged_outs``/``last_outs`` in place, so the replay
passes fresh copies of the pre-call arrays and compares their post-call state
against the recorded result ``[averaged_outs, last_outs]``. The two arrays are
reported separately: only ``averaged_outs`` is observable by real callers
(``last_outs`` is a per-call scratch buffer in MaskedModel).

    PYTHONPATH=.claude/skills/cpp-parity/scripts \
    .venv/bin/python .claude/skills/cpp-parity/scripts/parity_replay_bfo.py \
        --dir bugs/data/build_fixed_output [--report NAME]
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
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--report", metavar="NAME")
    args = ap.parse_args()

    fixtures = Path(args.dir)
    manifest = json.loads((fixtures / "manifest.json").read_text())
    target = manifest["target"]
    mod_name, _, func_name = target.partition(":")
    func = getattr(importlib.import_module(mod_name), func_name)
    links = importlib.import_module("shap.links")

    branch, commit = _git("rev-parse", "--abbrev-ref", "HEAD"), _git("rev-parse", "--short", "HEAD")
    print(f"replaying {len(manifest['cases'])} case(s) of {target}")
    print(f"  fixtures captured on: {manifest['git_branch']} @ {manifest['git_commit']}")
    print(f"  replaying on:         {branch} @ {commit}")

    results, failures, scratch_only = [], 0, 0
    for case in manifest["cases"]:
        if not case.get("file"):
            results.append({**case, "status": "SKIPPED", "diffs": [case.get("skipped_reason", "")]})
            print(f"  SKIP  {case['dims']}")
            continue
        with np.load(fixtures / case["file"], allow_pickle=False) as blob:
            call_args = [pc.decode(f"arg{i}", blob) for i in range(int(blob["n_args"]))]
            expected = pc.decode("result", blob)
        averaged = np.array(call_args[0], copy=True)
        last = np.array(call_args[1], copy=True)
        link = getattr(links, call_args[6])
        lw = call_args[7]
        try:
            func(averaged, last, call_args[2], call_args[3], call_args[4], call_args[5], link, lw)
            avg_diffs = [f"averaged_outs: {d}" for d in pc.compare(expected[0], averaged)]
            last_diffs = [f"last_outs (scratch): {d}" for d in pc.compare(expected[1], last)]
        except Exception as exc:
            avg_diffs, last_diffs = [f"raised {type(exc).__name__}: {exc}"], []
        if avg_diffs:
            status = "FAIL"
            failures += 1
        elif last_diffs:
            status = "SCRATCH-DIFF"
            scratch_only += 1
        else:
            status = "PASS"
        results.append({**case, "status": status, "diffs": avg_diffs + last_diffs})
        print(
            f"  {status:12s} link={case['link']:8s} weighted={str(case['weighted']):5s} "
            f"{case['dims'].split('+')[2]}  ({case['calls']} call(s), {len(case['tests'])} test(s))"
        )
        for d in avg_diffs + last_diffs:
            print(f"        -> {d}")

    dims = sorted({c["dims"] for c in manifest["cases"]})
    print(f"\ndim signatures: {len(dims)}")
    verdict = "PARITY CONFIRMED" if failures == 0 else f"PARITY BROKEN ({failures} case(s))"
    if failures == 0 and scratch_only:
        verdict += f" ({scratch_only} case(s) differ only in the last_outs scratch buffer)"
    print(verdict)

    if args.report:
        (fixtures / args.report).write_text(
            json.dumps(
                {
                    "target": target,
                    "captured_on": f"{manifest['git_branch']}@{manifest['git_commit']}",
                    "replayed_on": f"{branch}@{commit}",
                    "verdict": verdict,
                    "failures": failures,
                    "scratch_only_diffs": scratch_only,
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

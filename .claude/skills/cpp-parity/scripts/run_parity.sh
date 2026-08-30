#!/usr/bin/env bash
# One-shot parity check for a Python -> C++ migration.
#
# Captures every call of an entry point while the given tests run on the current
# (migration) branch, then replays the captured inputs on a baseline branch and
# diffs the results. Exits non-zero if any case disagrees.
#
#   TARGET=shap.utils._masked_model:make_masks \
#   OUT=bugs/data/make_masks \
#   BASELINE=master \
#   .claude/skills/cpp-parity/scripts/run_parity.sh tests/explainers/test_partition.py::test_tabular_single_output ...
#
# Requires a clean-enough tree: the baseline checkout will fail if tracked files
# that differ between the branches are dirty. Remove trace instrumentation first.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

SCRIPTS=".claude/skills/cpp-parity/scripts"
PY="${PY:-.venv/bin/python}"
TARGET="${TARGET:?set TARGET=module:qualname of the migrated entry point}"
OUT="${OUT:?set OUT=fixture directory, e.g. bugs/data/<name>}"
BASELINE="${BASELINE:-master}"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <pytest target> [<pytest target> ...]" >&2
  exit 2
fi

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "!! tracked files are dirty; the $BASELINE checkout may fail" >&2
fi

echo "== 1/3 capture on $BRANCH (migration branch) =="
rm -rf "$OUT"
PYTHONPATH="$SCRIPTS" PARITY_TARGET="$TARGET" PARITY_OUT="$OUT" PARITY_LABEL="$BRANCH" \
  "$PY" -m pytest "$@" -q -p no:randomly -p parity_capture

echo "== 2/3 self-replay on $BRANCH =="
PYTHONPATH="$SCRIPTS" "$PY" "$SCRIPTS/parity_replay.py" --dir "$OUT" --report "replay_${BRANCH//\//_}.json"

echo "== 3/3 replay on $BASELINE (baseline implementation) =="
git checkout --quiet "$BASELINE"
trap 'git checkout --quiet "$BRANCH"' EXIT
PYTHONPATH="$SCRIPTS" "$PY" "$SCRIPTS/parity_replay.py" --dir "$OUT" --report "replay_${BASELINE//\//_}.json"

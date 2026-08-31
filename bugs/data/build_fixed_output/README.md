# _build_fixed_output numba -> nanobind parity (feature/nanobind-build-output)

Parity verification for the `_build_fixed_single_output`/`_build_fixed_multi_output`
migration to `shap/cutils/masked_model_utils.h` (bindings
`build_fixed_single_output`/`build_fixed_multi_output`, 8 overloads:
single/multi x weighted/unweighted x float/double), captured through the shared
entry point `shap.utils._masked_model:_build_fixed_output` with the generic
harness in `.claude/skills/cpp-parity/scripts/`.

The target mutates `averaged_outs`/`last_outs` in place and returns None, and
takes a link *function* argument — the harness gained pre-call snapshots,
post-call mutated-array comparison, and callable-by-reference encoding for
this migration.

## Standalone capture/replay flow (lower_credit style)

`run_parity.sh` in this directory runs the whole check with no branch switch
and no rebuild: it captures 245+ hypothesis-generated calls
(`test_parity_driver.py`, recorded by the `parity_capture_build_fixed_output`
pytest plugin into `fixtures_hypothesis/`, gitignored), self-replays them on
the current C++ build (must be bitwise), then replays them through the numba
baseline extracted from `git show master:shap/utils/_masked_model.py` and
jitted in isolation (`parity_replay_build_fixed_output.py`). Baseline results
must be bitwise-exact or fall into the documented divergence classes below
(`last-outs-linked`, `empty-first-carry`, `float32-rounding`); anything
unclassified fails the run. Requires `hypothesis` and `numba` in the env.

    ./bugs/data/build_fixed_output/run_parity.sh          # BASELINE=master
    BASELINE=df974a19 ./bugs/data/build_fixed_output/run_parity.sh

Latest run (2026-08-31): 245 unique cases; self-replay 245/245 exact;
baseline replay 165 exact + 50 last-outs-linked + 43 float32-rounding +
2 empty-first-carry, 0 broken.

## Files

- `test_hypothesis_parity.py` — 380+ hypothesis cases per run (single/multi,
  weighted/unweighted, identity/logit, float32/float64, empty-batch runs,
  stale-row layouts, large inputs up to 80000 samples x 500 batches), each
  compared bitwise against a pure-Python port of the numba baseline that
  mimics numba's sequential reductions via `np.cumsum(...)[-1]`. Plus
  deterministic tests pinning each documented divergence below.
- `fixtures/` (gitignored, 757MB, regenerable) — 13379 unique captured calls
  + manifest + replay reports.
- `test_parity_driver.py`, `parity_capture_build_fixed_output.py`,
  `parity_replay_build_fixed_output.py`, `run_parity.sh` — the standalone
  capture/replay flow above; `fixtures_hypothesis/` (gitignored, regenerable)
  holds its captured cases.
- `test_link_runtime_warnings.py` — pins the link boundary behavior
  (RuntimeWarning emission and the baseline's scalar `ZeroDivisionError`)
  against the numba implementation extracted from git.

## How to run

```bash
TARGET=shap.utils._masked_model:_build_fixed_output \
OUT=bugs/data/build_fixed_output/fixtures BASELINE=df974a19 \
PARITY_MAX_BYTES=200000000 \
  .claude/skills/cpp-parity/scripts/run_parity.sh \
    bugs/data/build_fixed_output/test_hypothesis_parity.py \
    tests/utils/test_masked_model.py \
    tests/explainers/test_partition.py::test_tabular_single_output \
    tests/explainers/test_partition.py::test_tabular_multi_output \
    tests/explainers/test_exact.py::test_tabular_single_output_partition_masker \
    tests/explainers/test_exact.py::test_tabular_multi_output_partition_masker \
    tests/explainers/test_coalition.py::test_tabular_coalition_single_output \
    -k "not diverges and not float16"
```

(The baseline is the merge-base `df974a19`, not local `master`, which is
behind it. The capture step needs a longer timeout than the default — the
baseline replay of 13k fixtures takes several minutes on its own.)

## Result (2026-08-30, after the fixes, branch @ fca56149)

Fixtures recaptured on the fixed build: 13379 unique inputs, 1406 dim
signatures. Self-replay on the branch: **13379/13379 exact**. Replay on the
numba baseline @ df974a19: **13247/13379 exact**; every one of the 132
remaining cases is classified:

1. **54 cases: `last_outs` contents only** (weighted + non-identity link) —
   the branch stores linked values where the baseline stored raw outputs.
   `averaged_outs` identical; callers treat `last_outs` as scratch.
2. **77 cases: float32 rounding** — abs diff <= 1.7e-07 (float32 eps), no
   inf/nan mismatches; float64 always bit-exact. Numba's float32 reduction
   rounds differently from the C++ float32-native accumulation. Production
   callers allocate with `np.zeros` (float64), so the real code path is in
   the bit-exact group.
3. **1 case: the pinned empty-first-batch residual** — unweighted logit with
   an empty first batch links the wraparound-carried value (`-inf`) where
   the baseline left it raw (`0.0`); see
   `test_empty_first_batch_wraparound_carry`.
4. **0 raised.**

A pre-fix cycle (13379 inputs) measured 13264/13379 with the stale-row
weighted divergence still present; the fixes moved those cases into the
exact group. Suite status: hypothesis parity file, `tests/utils/`, and the
partition/exact/coalition/permutation explainer tests all pass on the fixed
build (69 tests + 380 hypothesis cases per run).

## Fixes applied (2026-08-30, after the findings below)

All review findings were fixed while keeping the C++ a literal translation
of the numba code:

- `.noconvert()` on every argument of all 8 `m.def` overloads, plus
  docstrings, plus the leading underscore restored
  (`_build_fixed_single_output`/`_build_fixed_multi_output`). The stub
  (`shap/_cutils.pyi`, build-generated) omits them like the other
  underscore-private bindings — nanobind's stubgen skips private names.
- `nb::device::cpu` on every ndarray in `masked_model_utils.h`.
- Input validation moved to Python (`_validate_build_fixed_inputs` in
  `_build_fixed_output`): inconsistent `batch_positions`/`varying_rows`/
  `num_varying_rows`/shapes raise `ValueError` before the unchecked native
  code runs (the numba implementation's numpy fancy assignments raised on
  most of these; the C++ read out of bounds).
- **Stale-row weighted link restored to baseline semantics**: the carried
  initial state of `last_outs` is linked up front, so a never-written row
  contributes `link(0)` (`logit` -> -inf) to the weighted mean exactly like
  the numba code that linked `last_outs` on every batch. Elementwise
  identical, including the carried values.
- **Empty-batch carry is now numba's literal negative-index wraparound**
  (`averaged_outs[i - 1]` at `i == 0` reads the last element), making the
  identity-link and weighted paths bit-exact with the baseline even for an
  empty first batch. Residual divergence: unweighted non-identity link with
  an empty first batch links the carried value at the end (baseline left it
  raw) — pinned by `test_empty_first_batch_wraparound_carry`.
- Weighted-path link applications wrapped in `np.errstate` so `logit(0)`
  does not emit warnings the silent numba code never emitted.

Not restored: the baseline stored *raw* outputs in `last_outs` for weighted
links (the branch stores linked values), and float16 results are written
back (the baseline dropped them) — both pinned by tests as intentional.

Link boundary behavior (probed 2026-09-01, pinned by
`test_link_runtime_warnings.py` against the real numba baseline from git):

- unweighted logit, batch mean exactly 0.0: baseline silently produced
  `-inf` (numba scalar `np.log(0.0)`); the branch produces the bitwise-same
  `-inf` but now emits `RuntimeWarning: divide by zero encountered in log`
  from the unwrapped trailing link application.
- unweighted logit, batch mean exactly 1.0: the baseline RAISED
  `ZeroDivisionError` — inside `@njit` the scalar `x / (1 - x)` follows
  Python semantics, not numpy's. The branch returns `+inf` (with the same
  RuntimeWarning). The migration turned a crash into a value here.
- weighted paths apply the link to arrays on both stacks (numpy semantics in
  numba too): silent on both sides, `averaged_outs` bitwise identical.

## Review findings (probed on the pre-fix code; all fixed above)

1. **No `.noconvert()` on any binding argument**: a dtype-mismatched call via
   the raw binding converts the output arrays to temporaries and silently
   drops every write (probed: float32 `averaged_outs` + float64 `outputs`
   returns all zeros, no error).
2. **No shape/consistency validation anywhere**: `batch_positions` values are
   trusted; a malformed matrix reads far out of bounds (probed through the
   public `_build_fixed_output`: heap garbage returned in `averaged_outs`,
   no error).
3. **Missing `nb::device::cpu`** on every ndarray in `masked_model_utils.h`.
4. **API naming**: the bindings dropped the leading underscore the numba
   functions had.
5. **m.def had no docstrings** for the 8 overloads.
6. **Stale-row weighted link diverged from the baseline** (found by
   hypothesis): finite mean where the baseline produced -inf.
7. Native execution proof: the numba workers are deleted on this branch, so
   the bindings are the only implementation; the self-replay compares the
   mutated output buffers, which a silently-dropped conversion would leave
   zeroed (the exact failure mode probed in finding 1). No temporary C++
   trace was used.

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

## Files

- `test_hypothesis_parity.py` — 380+ hypothesis cases per run (single/multi,
  weighted/unweighted, identity/logit, float32/float64, empty-batch runs,
  stale-row layouts, large inputs up to 80000 samples x 500 batches), each
  compared bitwise against a pure-Python port of the numba baseline that
  mimics numba's sequential reductions via `np.cumsum(...)[-1]`. Plus
  deterministic tests pinning each documented divergence below.
- `fixtures/` (gitignored, 757MB, regenerable) — 13379 unique captured calls
  + manifest + replay reports.

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

## Result (2026-08-30, branch feature/nanobind-build-output @ 9df9a697)

43700 calls, 13379 unique inputs, 1406 dim signatures. Self-replay on the
branch: **13379/13379 exact**. Replay on the numba baseline @ df974a19:
**13264/13379 exact**; the 115 remaining split into three understood groups:

1. **48 cases: `last_outs` contents only** (weighted + logit) — the branch
   stores `link(outputs)` in `last_outs` where the baseline stored raw
   outputs. `averaged_outs` identical; callers treat `last_outs` as scratch.
2. **67 cases: float32 rounding** — abs diff <= 1.7e-07 (float32 eps),
   float64 always bit-exact. Numba's float32 reduction rounds differently
   from the C++ float32-native accumulation. Production callers allocate
   `averaged_outs`/`last_outs` with `np.zeros` (float64), so the real code
   path is in the bit-exact group.
3. **0 raised.**

## Documented behavior changes vs the baseline (each pinned by a test)

- **Stale-row weighted link**: a row never written by any batch contributes
  `logit(0) = -inf` to the baseline's weighted mean; the branch keeps the
  stale zero unlinked and stays finite. Real callers populate all rows in the
  first batch (zero-mask over the full background), but `_delta_masking_call`
  does not obviously guarantee it — worth confirming.
- **Empty first batch**: baseline carried `averaged_outs[-1]` (wraparound
  read, no link); the branch writes 0 then applies the link (`logit` -> -inf).
- **float16**: the baseline upcast into copies and silently dropped all
  results (caller buffers untouched); the branch writes results back.

## Review findings (probed)

1. **No `.noconvert()` on any binding argument**: a dtype-mismatched call via
   the raw binding converts the output arrays to temporaries and silently
   drops every write (probed: float32 `averaged_outs` + float64 `outputs`
   returns all zeros, no error). The Python entry always passes a consistent
   working dtype, so this is only reachable through `shap._cutils` directly.
2. **No shape/consistency validation anywhere**: `batch_positions` values are
   trusted; a malformed matrix reads far out of bounds (probed through the
   public `_build_fixed_output`: heap garbage returned in `averaged_outs`,
   no error). Same class as the make_masks findings; validation belongs in
   `_build_fixed_output` before the native call.
3. **Missing `nb::device::cpu`** on every ndarray in `masked_model_utils.h`
   (the directory convention in `grey_code_utils.h` etc. includes it): a CUDA
   tensor would be dereferenced on the host.
4. **API naming**: the bindings dropped the leading underscore the numba
   functions had (`build_fixed_single_output` vs `_build_fixed_single_output`)
   — inconsistent with `_init_masks`/`_rec_fill_masks`/`_delta_masking`.
5. **m.def has no docstrings** for the 8 overloads (every other binding in
   `cutils.cpp` documents itself).
6. Native execution proof: the numba workers are deleted on this branch, so
   the bindings are the only implementation; the self-replay compares the
   mutated output buffers, which a silently-dropped conversion would leave
   zeroed (the exact failure mode probed in finding 1). No temporary C++
   trace was used.

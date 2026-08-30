# make_masks numba -> nanobind parity (PR #5082)

Parity fixtures for the `_init_masks`/`_rec_fill_masks` migration on
`feature/nanobind-masks-2`, captured through the shared entry point
`shap.utils._masked_model:make_masks` with the generic harness in
`.claude/skills/cpp-parity/scripts/`. These are captured production calls, not
stale test data.

## Files

- `fixtures/case_*.npz` + `fixtures/manifest.json` — 6 unique inputs (M = 2, 4,
  6, 12, 30, 300) with the CSR result each produced; provenance per case.
- `fixtures/replay_*.json` — per-branch replay reports from the confirming run.
- `test_hypothesis_parity.py` — property-based parity: 700+ hypothesis-generated
  cases per run, each compared exactly (dense and/or CSR internals) against a
  verbatim pure-Python port of the numba baseline. Covers sizes M = 2..60000
  plus a fixed 2^19-leaf tree, shapes from balanced through caterpillars to
  chains in both orientations, input dtypes (float64/32/16, int64/32/16,
  uint32, object arrays mixing python ints and floats), fractional child ids,
  nan/inf/huge distance columns, and malformed inputs that must raise. Needs
  `uv pip install hypothesis` (not a project dependency; skips without it).
- `test_generate_cases.py` — synthetic clusterings fed to the capture run: the
  five production tests that reach `make_masks` all share one xgboost/adult
  scenario (a single (11, 4) cluster matrix), so this adds tiny/balanced/chain/
  scipy-linkage/image-scale trees for real size coverage.

## How to run

```bash
TARGET=shap.utils._masked_model:make_masks \
OUT=bugs/data/make_masks/fixtures BASELINE=master \
  .claude/skills/cpp-parity/scripts/run_parity.sh \
    tests/explainers/test_partition.py::test_tabular_single_output \
    tests/explainers/test_partition.py::test_tabular_multi_output \
    tests/explainers/test_exact.py::test_tabular_single_output_partition_masker \
    tests/explainers/test_exact.py::test_tabular_multi_output_partition_masker \
    tests/explainers/test_coalition.py::test_tabular_coalition_single_output \
    bugs/data/make_masks/test_generate_cases.py
```

## Result (2026-08-30, branch feature/nanobind-masks-2 @ 02d4a2e1)

**PARITY CONFIRMED** — 6/6 cases exact (dense values, CSR `indptr` and
`indices` all identical) on both the branch self-replay and the numba baseline
replay on master @ f290d210.

Re-confirmed 2026-08-30 after the C++ was aligned to the literal numba
mapping proposed in PR #5082 (direct recursion, no checks in the native
code; the only deviation kept is `nb::shape<-1, 4>` on `cluster_matrix`
instead of `ndim<2>`): 6/6 exact on these fixtures. Findings 2 and 3 below
are fixed on the Python side instead — `_validate_clustering` in
`make_masks` rejects out-of-range/forward-referencing children with
`IndexError` and clusterings deeper than `_MAX_CLUSTERING_DEPTH` (30000)
with `ValueError`. The raw bindings stay unguarded like the numba baseline:
probed via `_rec_fill_masks` directly, a degenerate chain is fine at
M=60000 and segfaults at M=80000, so the 30000 limit keeps a 2x margin.
The hypothesis suite also gained large cases: 40 random linkages with
M=20000..60000, a balanced tree with 2^19 leaves, and chains on both sides
of the depth limit (~13s for the large property alone).

## Findings (review + probes)

Reproduction tests live in `tests/utils/test_masked_model.py` and go through
the public API (`shap.PartitionExplainer`, `shap.CoalitionExplainer`,
`shap.utils.make_masks`); the ones marked FAILING are deliberate red tests that
reproduce a bug and go green once it is fixed. Crash reproductions run the
payload in a subprocess so a segfault fails the test instead of killing pytest.

1. **`make_masks` breaks on Windows (new regression).** Its index arrays are
   allocated `dtype=int` — int64 on Linux, int32 on Windows — while the
   bindings demand int64 with `.noconvert()`. On Windows every
   PartitionExplainer/CoalitionExplainer/partition-masker ExactExplainer with a
   static clustering raises `TypeError` at `_init_masks`. The numba baseline
   accepted any int dtype. Fix: `dtype=np.int64` on the three allocations.
   Test: `test_partition_explainer_with_custom_clustering` (green on Linux,
   fails on Windows runners).
2. **Out-of-range clustering index → segfault** through the public
   constructors (probed: exit -11). An unchecked `indices_row_pos(lind)` read
   with garbage `lind` from the cluster matrix. Inherited — the numba baseline
   (boundscheck off) segfaults on the same input — but should become a Python
   exception now that the code is C++, like the validation `lower_credit` got.
   Tests (FAILING until fixed):
   `test_partition_explainer_rejects_out_of_range_clustering`,
   `test_coalition_explainer_rejects_out_of_range_clustering`.
3. **100k-leaf degenerate chain → C++ stack overflow** (probed: exit -11), on
   *valid* input; `make_masks`'s docstring claims it exists for very large
   image trees. Inherited: the numba baseline segfaults identically.
   Test (FAILING until fixed): `test_make_masks_deep_chain_clustering`.
4. **Undersized output arrays via the raw `_init_masks` binding corrupt the
   heap** (probed: prints success, then segfaults at interpreter shutdown).
   No public path reaches this state — `make_masks` always allocates
   consistent sizes — so this is documented here rather than tested;
   validating `M` against the array extents in the binding closes it.
5. **Missing `nb::shape<-1, 4>` on `cluster_matrix`** (`ndim<2>` only): a
   <4-column matrix through the raw binding reads column 3 out of bounds
   (probed: silent garbage, no error). Publicly, numpy's own
   `cluster_matrix[:, 3]` raises first — pinned by
   `test_make_masks_rejects_narrow_cluster_matrix` (green) so the error never
   regresses into the native layer.
6. **`cluster_matrix.astype(np.double)` copies on every call**, even when the
   input is already float64 (probed); `np.asarray(cluster_matrix,
   dtype=np.float64)` copies only when needed. It is also what shields callers
   from the raw binding's float32 `TypeError`.
7. **Stale stub:** source-tree `shap/_cutils.pyi` lacks
   `_init_masks`/`_rec_fill_masks` (it does contain `lower_credit`), so
   mypy/IDEs see an outdated contract for this branch.
8. Minor: `_init_masks`'s binding docstring says "kernel explainer algorithm"
   (it builds partition mask matrices); the branch's `test_init_masks`
   docstring still claims `.noconvert()` is missing though it is present now.
9. Verified-correct: the `indptr` parameter dropped from `_rec_fill_masks` was
   unused in the numba baseline; `static_cast<int64_t>(double)` matches
   Python's `int()` truncation for these non-negative values; the exported
   names keep their leading underscore.

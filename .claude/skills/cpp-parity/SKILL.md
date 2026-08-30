---
name: cpp-parity
description: Verify a Python/numba-to-C++ (nanobind) migration in shap behaves identically to the pre-migration implementation. Use when a branch moves code from numba/@njit or pure Python into shap/cutils/*.h, when asked to "test the C++ rewrite", check migration parity, capture golden fixtures for a new binding, or review how a migration was done. Covers rebuilding the extension, tracing which tests reach the native code, capturing fixtures to bugs/data, replaying them on the baseline branch, checking dimension coverage, and reviewing nanobind pitfalls.
---

# C++ migration parity testing

Branch-agnostic procedure for proving a new `shap/cutils` binding matches the
implementation it replaced. Nothing here is specific to one function: everything
is parameterized by `TARGET` (the Python entry point that calls into C++) and the
baseline branch.

**Never claim parity from a passing test suite alone.** The old and new code paths
must be run on the *same recorded inputs* and produce byte-identical outputs. A
mask/index bug can leave the whole suite green (see `bugs/data/*/README.md` for
the case where `indptr` came back all zeros and nothing failed).

## Step 0 — scope the migration

```bash
BASELINE=master                        # or whatever the branch forked from
git diff --stat $BASELINE...HEAD
git diff $BASELINE...HEAD -- '*.py'    # what was deleted on the Python side
```

Identify, and write down:

- the new/changed `shap/cutils/*.h` functions and their `m.def(...)` bindings in `shap/cutils/cutils.cpp`;
- the **removed** Python/numba functions (usually `@njit`, e.g. `_init_masks`);
- the **surviving Python entry point** that now calls the binding (e.g.
  `shap.utils._masked_model:make_masks`). That entry point is `TARGET` — it exists
  on *both* branches, which is what makes cross-branch replay possible.

If the migration replaced the entry point itself (no shared Python-level
function), pick the nearest common caller instead.

## Step 1 — rebuild the extension

Per `CONTRIBUTING.md` ("Installing from source"), the project is installed
editable and the C++ is built by scikit-build-core:

```bash
uv sync --group test --group test-core --group plots --reinstall-package shap
# equivalent:  pip install -e . --group test-core --group plots
```

- `--reinstall-package shap` is **required**: editing a header does not trigger a
  rebuild on plain `uv sync`/`uv run`. The repo's `./rebuild.sh` wraps this.
- Pass every dependency group the env already had. `uv sync` prunes anything not
  requested, so a bare `--group test-core` silently uninstalls torch/tensorflow/etc.
- Confirm the rebuild actually landed before trusting any result:
  ```bash
  .venv/bin/python -c "import shap._cutils as c; print(c.__file__, dir(c))"
  cat shap/_cutils.pyi   # regenerated stub: the binding's real dtype/shape/device contract
  ```

## Step 2 — prove the native code runs (temporary trace)

Add an env-gated trace to the migrated header. Green tests do not prove the C++
ran — a silently dropped conversion can no-op the call.

```cpp
#include <cstdio>
#include <cstdlib>
// TEMPORARY MIGRATION TRACE -- remove before merging.
#define SHAP_TRACE_ON() (std::getenv("SHAP_CPP_TRACE") != nullptr)
...
    if (SHAP_TRACE_ON()) {
        std::fprintf(stderr, "[CPPTRACE] fname ndim=%zu shape=(%zu,%zu) M=%lld\n",
                     (size_t)arr.ndim(), (size_t)arr.shape(0), (size_t)arr.shape(1), (long long)M);
        std::fflush(stderr);
    }
```

For a recursive function, guard on a `static thread_local` depth counter and
trace only the outermost entry, otherwise one call floods the log with a line per
tree node. Rebuild (step 1) after adding the trace.

Print shape/ndim/dtype, not just a marker — step 5 needs it. Revert the header
(`git checkout -- <header>`) before any branch switch.

## Step 3 — find the hitting tests by call graph, not by sweeping

Do **not** run the whole suite. Work up the call graph statically, then run the
candidates individually — the suite has slow tests (coalition, exact, text/image
maskers download models) and a sweep costs 20+ minutes for information a grep
gives in seconds.

```bash
grep -rn "make_masks" --include='*.py' shap/          # who calls TARGET
grep -rln "Partition\|Exact\|Coalition" tests/        # tests touching those callers
.venv/bin/python -m pytest tests/explainers/test_partition.py --collect-only -q
```

Then run the shortlist with the trace and the capture plugin (step 4), and
attribute native entries to tests by interleaving both streams:

```bash
grep -E '^\[(TEST|CPPTRACE)\]' run.err \
  | awk '/^\[TEST\]/{t=$2} /^\[CPPTRACE\]/{print t, $3, $4, $5}' | sort | uniq -c
```

`[TEST] <nodeid>` lines come from the capture plugin when `SHAP_CPP_TRACE` is
set; run pytest with `-s` so the C++ writes to fd 2 are not swallowed.

Expect some candidates to miss (e.g. `ExactExplainer` with an auto masker never
reaches `make_masks`). Record hits *and* misses.

## Step 4 — capture fixtures into `bugs/data/<name>/`

```bash
PYTHONPATH=.claude/skills/cpp-parity/scripts \
PARITY_TARGET=shap.utils._masked_model:make_masks \
PARITY_OUT=bugs/data/make_masks \
PARITY_LABEL=cpp-nanobind SHAP_CPP_TRACE=1 \
  .venv/bin/python -m pytest <shortlist> -q -s -p no:randomly -p parity_capture
```

`scripts/parity_capture.py` wraps `PARITY_TARGET`, dedupes calls by input hash,
and writes `case_<hash>.npz` per unique input plus `manifest.json` (per-test hit
counts, per-case dim signature, git branch/commit). It rebinds *every* live
reference to the target, because `from ..utils import make_masks` copies the
reference — if it reports `bindings_patched: 1` for a widely imported function,
be suspicious.

Add a short `README.md` next to the fixtures saying what they are and how to
replay them, so they are not mistaken for stale test data.

## Step 5 — replay on the baseline branch

```bash
TARGET=shap.utils._masked_model:make_masks OUT=bugs/data/make_masks BASELINE=master \
  .claude/skills/cpp-parity/scripts/run_parity.sh <shortlist>
```

This captures on the current branch, self-replays, checks out `BASELINE`,
replays the same inputs through the old implementation, and restores the branch.
It exits non-zero on any mismatch. `parity_replay.py --dir <fixtures>` runs a
single side by itself.

Comparison is exact (`np.array_equal`); sparse results are compared densely
*and* on their CSR internals, so a permuted-but-equal matrix is flagged.

Caveats:
- Revert trace instrumentation first, or the baseline checkout fails on a dirty
  header that does not exist there.
- The replay uses the working-tree Python (editable install) but the *installed*
  `.so`. That is fine when the baseline path is pure Python/numba. If the
  baseline also needs C++ that the branch changed, rebuild after each checkout.
- Numba compiles on first call; the baseline replay is slower, not broken.

## Step 6 — dimension coverage

`manifest.json`'s `dim_signatures` lists every distinct
`ndim(shape):dtype` combination that reached the code. Two things to check:

1. **Binding overloads.** If `cutils.cpp` registers more than one overload
   (`compute_grey_code_row_values` has a 1D and a 2D form), each overload needs at
   least one hitting test. An unexercised overload is untested code.
2. **Input rank.** If the signature is fixed-rank (`nb::ndim<2>` on an `(n, 4)`
   cluster matrix), there is only one rank to cover — say so explicitly rather
   than implying multi-dimensional coverage. Then vary what *can* vary (size,
   degenerate `M`, dtype) and add a test per distinct class that is missing.

Report the count of signatures and which test covers each.

## Step 7 — review the migration

Check each item against the diff; most nanobind migration bugs are here.

**Argument contract (highest value — these fail silently):**

- **Missing `.noconvert()` on output arrays.** `nb::ndarray<int64_t>` accepts an
  int32 array via implicit conversion, writes into a temporary copy, and drops
  it. No exception, caller's array untouched. Any array the C++ *writes* must be
  `.noconvert()` in the `m.def`, or the dtype coupling must be enforced in Python.
- **`dtype=int` on the Python side.** Not a contract; it is platform/NumPy-version
  dependent. Allocate `np.int64` explicitly when the binding says int64.
- **`.astype(...)` added at the call site** to satisfy the binding: it copies on
  *every* call (even when the dtype already matches) and hides the mismatch
  instead of asserting it.
- **Missing `nb::device::cpu`** — compare against the conventions already used in
  the same directory (`grey_code_utils.h`, `kernel_explainer_utils.h` use
  `nb::shape<-1, -1>, nb::device::cpu`). Without it a CUDA tensor is accepted and
  its device pointer dereferenced on the host.
- **Missing shape constraint.** `nb::ndim<2>` fixes the rank but not the extents;
  `nb::shape<-1, 4>` states the real requirement.

**Memory safety:** `view()` indexing is unchecked. Every size implied by a scalar
argument (`M`, `ind`) or by a column of the input must be validated against the
actual array extents, and out-of-range values rejected with an exception. Verify
by probing from Python: an undersized output array or a wrong `M` should raise,
not corrupt the heap. Also check recursion depth against realistic input sizes
(image trees have tens of thousands of leaves) — the C++ stack has no
`RecursionError`.

**API surface:** newly exported names in `cutils.cpp` become part of
`shap._cutils`. Private helpers should keep the leading underscore they had in
Python (`_init_masks`), and the docstring should name the right algorithm.

**Semantics:** confirm each `static_cast<int64_t>(double)` matches the Python
`int()` truncation it replaced, that loop bounds and slice copies map 1:1, and
that dropped parameters really were unused.

**Tests:** a migration with no direct test of the new binding is untested; the
parity fixtures from step 4 should be committed as a regression test that calls
the binding directly with the dtypes the caller actually uses.

## Reporting

State, in this order:

1. the reproduction command (`run_parity.sh` invocation) and its verdict —
   `PARITY CONFIRMED` / `PARITY BROKEN`, with case count;
2. which tests hit the native code, and notable misses;
3. dimension coverage: number of signatures, per-overload coverage, gaps;
4. review findings, most severe first, each with the concrete failure mode;
5. that trace instrumentation was reverted.

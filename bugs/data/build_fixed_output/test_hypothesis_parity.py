"""Hypothesis parity checks for the _build_fixed_output numba -> nanobind migration.

Compares ``shap.utils._masked_model._build_fixed_output`` (the C++
``build_fixed_single_output``/``build_fixed_multi_output`` bindings) against a
pure-Python port of the numba baseline from the merge-base (df974a19), written
to mimic numba's semantics exactly: sequential same-dtype reductions
(``np.cumsum(...)[-1]``, bitwise-identical to numba's naive summation loop, not
numpy's pairwise ``np.sum``).

The target mutates ``averaged_outs`` and ``last_outs`` in place and returns
None, so every check compares the post-call state of both buffers. Per run the
properties generate 300+ cases covering single/multi output, weighted and
unweighted, identity and logit links, float32/float64, varying-row subsets,
empty batches, and large inputs.

The remaining intentional differences from the baseline get dedicated tests
instead of polluting the parity properties:

* weighted + non-identity link: the branch stores ``link(outputs)`` values in
  ``last_outs`` (the baseline stored raw outputs). Callers treat ``last_outs``
  as scratch, so only the buffer contents differ, never ``averaged_outs``.
  Stale (never-written) rows now match the baseline's averaging exactly: the
  carried initial state is linked up front, so a stale zero still contributes
  ``logit(0) = -inf`` to the weighted mean like the numba code did.
* an empty *first* batch carries via numba's negative-index wraparound
  (``averaged_outs[-1]``), now replicated literally in the C++; only the
  unweighted non-identity link still differs there, because the branch
  applies the link to the carried value at the end while the baseline never
  linked it.
* float16 inputs: the baseline upcast into copies and dropped the results
  (``averaged_outs`` stayed untouched); the branch writes real results back.
* malformed inputs (inconsistent batch_positions/varying_rows/shapes) raise
  ``ValueError`` from Python-side validation instead of reading out of
  bounds in the unchecked native code.

Run with:

    uv run pytest bugs/data/build_fixed_output/test_hypothesis_parity.py -q

``hypothesis`` is not a project dependency (install with
``uv pip install hypothesis``); the module skips cleanly without it.
"""

import numpy as np
import pytest

pytest.importorskip("hypothesis")
from hypothesis import given, settings
from hypothesis import strategies as st

from shap.links import identity, logit
from shap.utils._masked_model import _build_fixed_output

# --- pure-Python port of the numba baseline (merge-base df974a19) ------------


def _naive_mean(a):
    """Sequential same-dtype mean, matching numba's np.mean (not numpy's pairwise)."""
    return np.cumsum(a)[-1] / a.dtype.type(len(a))


def _ref_single(averaged_outs, last_outs, outputs, batch_positions, varying_rows, num_varying_rows, link, lw):
    sample_count = last_outs.shape[0]
    for i in range(len(averaged_outs)):
        if batch_positions[i] < batch_positions[i + 1]:
            if num_varying_rows[i] == sample_count:
                last_outs[:] = outputs[batch_positions[i] : batch_positions[i + 1]]
            else:
                last_outs[varying_rows[i]] = outputs[batch_positions[i] : batch_positions[i + 1]]
            if lw is not None:
                averaged_outs[i] = _naive_mean(lw * link(last_outs))
            else:
                averaged_outs[i] = link(_naive_mean(last_outs))
        else:
            averaged_outs[i] = averaged_outs[i - 1]


def _ref_multi(averaged_outs, last_outs, outputs, batch_positions, varying_rows, num_varying_rows, link, lw):
    sample_count = last_outs.shape[0]
    for i in range(len(averaged_outs)):
        if batch_positions[i] < batch_positions[i + 1]:
            if num_varying_rows[i] == sample_count:
                last_outs[:] = outputs[batch_positions[i] : batch_positions[i + 1]]
            else:
                last_outs[varying_rows[i]] = outputs[batch_positions[i] : batch_positions[i + 1]]
            if lw is not None:
                for j in range(last_outs.shape[-1]):
                    averaged_outs[i, j] = _naive_mean(lw[:, j] * link(last_outs[:, j]))
            else:
                for j in range(last_outs.shape[-1]):
                    averaged_outs[i, j] = link(_naive_mean(last_outs[:, j]))
        else:
            averaged_outs[i] = averaged_outs[i - 1]


def _ref_build_fixed_output(averaged_outs, last_outs, *args):
    if len(last_outs.shape) == 1:
        _ref_single(averaged_outs, last_outs, *args)
    else:
        _ref_multi(averaged_outs, last_outs, *args)


# --- strategies -------------------------------------------------------------

_LINKS = [identity, logit]


@st.composite
def build_fixed_cases(
    draw,
    multi,
    max_samples=12,
    max_batches=8,
    min_samples=1,
    min_batches=1,
    full_first=True,
    allow_weighted=True,
    seed_scale=2**32 - 1,
):
    """A consistent (outputs, batch_positions, varying_rows, ...) call, from a
    hypothesis-drawn seed and layout. ``full_first`` makes the first batch vary
    every row, the real-caller invariant that keeps weighted links comparable
    to the baseline (see module docstring).
    """
    n = draw(st.integers(min_samples, max_samples))
    out = draw(st.integers(1, 4)) if multi else None
    n_batches = draw(st.integers(min_batches, max_batches))
    dtype = draw(st.sampled_from([np.float32, np.float64]))
    weighted = draw(st.booleans()) if allow_weighted else False
    link = draw(st.sampled_from(_LINKS))
    rng = np.random.default_rng(draw(st.integers(0, seed_scale)))

    varying = np.zeros((n_batches, n), dtype=bool)
    for b in range(n_batches):
        varying[b] = rng.random(n) < rng.random()
    if full_first:
        varying[0] = True
    elif not varying[0].any():
        varying[0, rng.integers(n)] = True
    num_varying_rows = varying.sum(axis=1).astype(np.int64)
    batch_positions = np.zeros(n_batches + 1, dtype=np.int64)
    np.cumsum(num_varying_rows, out=batch_positions[1:])

    shape = (n,) if out is None else (n, out)
    out_shape = (int(batch_positions[-1]),) if out is None else (int(batch_positions[-1]), out)
    outputs = rng.uniform(0.05, 0.95, size=out_shape).astype(dtype)
    lw = rng.uniform(0.5, 1.5, size=shape).astype(dtype) if weighted else None
    averaged_outs = np.zeros((n_batches,) if out is None else (n_batches, out), dtype=dtype)
    last_outs = np.zeros(shape, dtype=dtype)
    return averaged_outs, last_outs, outputs, batch_positions, varying, num_varying_rows, link, lw


def _assert_parity(case):
    averaged, last, outputs, bp, vr, nvr, link, lw = case
    got_avg, got_last = averaged.copy(), last.copy()
    exp_avg, exp_last = averaged.copy(), last.copy()

    _build_fixed_output(got_avg, got_last, outputs, bp, vr, nvr, link, lw)
    _ref_build_fixed_output(exp_avg, exp_last, outputs, bp, vr, nvr, link, lw)

    np.testing.assert_array_equal(got_avg, exp_avg, strict=True)
    if lw is None or link is identity:
        np.testing.assert_array_equal(got_last, exp_last, strict=True)
    else:
        # documented difference: the branch stores link(outputs) in last_outs
        np.testing.assert_array_equal(got_last, logit(exp_last), strict=True)


# --- parity properties ------------------------------------------------------


@settings(max_examples=120, deadline=None)
@given(build_fixed_cases(multi=False, full_first=False))
def test_single_output_matches_numba_baseline(case):
    _assert_parity(case)


@settings(max_examples=120, deadline=None)
@given(build_fixed_cases(multi=True, full_first=False))
def test_multi_output_matches_numba_baseline(case):
    _assert_parity(case)


@settings(max_examples=60, deadline=None)
@given(st.booleans().flatmap(lambda m: build_fixed_cases(multi=m, max_batches=20, min_batches=10)))
def test_many_batches_with_empty_runs(case):
    """Long batch lists make empty-batch carry runs (bp[i] == bp[i+1]) common."""
    _assert_parity(case)


@settings(max_examples=25, deadline=None)
@given(build_fixed_cases(multi=False, min_samples=20_000, max_samples=80_000, min_batches=200, max_batches=500))
def test_large_single_output(case):
    _assert_parity(case)


@settings(max_examples=8, deadline=None)
@given(build_fixed_cases(multi=True, min_samples=2_000, max_samples=10_000, min_batches=50, max_batches=150))
def test_large_multi_output(case):
    _assert_parity(case)


@settings(max_examples=60, deadline=None)
@given(st.booleans().flatmap(lambda m: build_fixed_cases(multi=m, full_first=False)))
def test_stale_rows_parity(case):
    """Partial first batches (never-written rows) match the baseline in every
    mode, including weighted logit: the carried initial state is linked up
    front, so stale zeros contribute logit(0) = -inf exactly like numba did.
    """
    _assert_parity(case)


# --- documented divergences from the baseline, and the guards ----------------


def test_stale_row_weighted_logit_matches_baseline():
    """A never-written row + weighted logit: logit(0) = -inf poisons the mean
    exactly like the numba baseline (the carried initial state is linked).
    """
    bp = np.array([0, 1], dtype=np.int64)
    vr = np.array([[True, False]])
    nvr = vr.sum(axis=1).astype(np.int64)
    outputs = np.array([0.75])
    lw = np.array([1.0, 1.0])
    averaged = np.zeros(1)
    last = np.zeros(2)
    _build_fixed_output(averaged, last, outputs, bp, vr, nvr, logit, lw)
    assert averaged[0] == -np.inf  # matches the numba baseline


def test_empty_first_batch_wraparound_carry():
    """The carry replicates numba's negative-index wraparound. With identity
    (and the weighted path) that is bit-exact with the baseline; with an
    unweighted non-identity link the carried value still gets linked at the
    end (logit(0) = -inf) where the baseline left it raw.
    """
    bp = np.array([0, 0, 2], dtype=np.int64)
    vr = np.array([[False, False], [True, True]])
    nvr = vr.sum(axis=1).astype(np.int64)
    outputs = np.array([0.25, 0.75])

    averaged = np.zeros(2)
    last = np.zeros(2)
    _build_fixed_output(averaged, last, outputs, bp, vr, nvr, identity, None)
    assert averaged[0] == 0.0  # wraparound read of the initial last element, like numba

    averaged = np.zeros(2)
    last = np.zeros(2)
    _build_fixed_output(averaged, last, outputs, bp, vr, nvr, logit, None)
    assert averaged[0] == -np.inf  # residual divergence: baseline produced raw 0.0
    assert np.isfinite(averaged[1])


def test_malformed_inputs_raise_instead_of_reading_out_of_bounds():
    averaged = np.zeros(4)
    last = np.zeros(3)
    outputs = np.array([0.2, 0.4, 0.6])
    vr = np.ones((4, 3), dtype=bool)
    nvr = np.array([3, 3, 3, 3], dtype=np.int64)

    bad_bp = np.array([0, 3, 2_000_000, 2_000_003, 2_000_006], dtype=np.int64)
    with pytest.raises(ValueError):
        _build_fixed_output(averaged, last, outputs, bad_bp, vr, nvr, identity, None)

    bp = np.array([0, 3, 3, 3, 3], dtype=np.int64)
    bad_nvr = np.array([3, 0, 1, 0], dtype=np.int64)  # inconsistent with bp diffs
    with pytest.raises(ValueError):
        _build_fixed_output(averaged, last, outputs, bp, vr, bad_nvr, identity, None)

    with pytest.raises(ValueError):  # varying_rows shape mismatch
        _build_fixed_output(averaged, last, outputs, bp, np.ones((4, 2), dtype=bool), nvr, identity, None)

    vr_ok = np.zeros((4, 3), dtype=bool)
    vr_ok[0] = True
    nvr_ok = vr_ok.sum(axis=1).astype(np.int64)
    with pytest.raises(ValueError):  # weights shape mismatch
        _build_fixed_output(averaged, last, outputs, bp, vr_ok, nvr_ok, logit, np.ones(2))


def test_float16_results_are_written_back():
    """Baseline upcast float16 into copies and dropped the results entirely."""
    bp = np.array([0, 2], dtype=np.int64)
    vr = np.array([[True, True]])
    nvr = vr.sum(axis=1).astype(np.int64)
    outputs = np.array([0.25, 0.75], dtype=np.float16)
    averaged = np.zeros(1, dtype=np.float16)
    last = np.zeros(2, dtype=np.float16)
    _build_fixed_output(averaged, last, outputs, bp, vr, nvr, identity, None)
    assert averaged[0] == np.float16(0.5)  # baseline left this at 0.0

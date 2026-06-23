import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from shap import links
from shap.utils._masked_model import (
    MaskedModel,
    _build_fixed_multi_output,
    _build_fixed_output,
    _build_fixed_single_output,
    _convert_delta_mask_to_full,
    _init_masks,
    _rec_fill_masks,
    _upcast_array,
    link_reweighting,
    make_masks,
)


class BasicMasker:
    """Simple masker used for full-mask and converted-delta paths."""

    supports_delta_masking = False
    immutable_outputs = False
    shape = (2, 3)

    def __init__(self):
        self.background = np.array([[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]], dtype=float)
        self._state_mask = np.zeros(3, dtype=bool)
        self.reset_delta_masking = object()

    def invariants(self, x):
        return np.zeros((self.background.shape[0], x.shape[0]), dtype=bool)

    def __call__(self, mask, x):
        if np.isscalar(mask):
            self._state_mask[int(mask)] = ~self._state_mask[int(mask)]
            mask = self._state_mask.copy()
        out = self.background.copy()
        out[:, mask] = x[mask]
        return out


class DeltaMasker:
    """Masker that supports delta masking in a realistic packed form."""

    supports_delta_masking = True
    immutable_outputs = True
    shape = (2, 3)

    def __init__(self):
        self.background = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float)

    def invariants(self, x):
        return np.zeros((self.background.shape[0], x.shape[0]), dtype=bool)

    def __call__(self, masks, x):
        full_masks = np.zeros((int(np.sum(np.asarray(masks) >= 0)), x.shape[0]), dtype=bool)
        _convert_delta_mask_to_full(np.asarray(masks), full_masks)
        masked_rows = []
        varying_rows = []
        prev = np.zeros(x.shape[0], dtype=bool)
        for mask in full_masks:
            out = self.background.copy()
            out[:, mask] = x[mask]
            masked_rows.append(out)
            varying_rows.append(mask ^ prev)
            prev = mask
        # Broadcast feature-level variation to both background rows.
        varying_rows = np.array([np.any(v) * np.ones(2, dtype=bool) for v in varying_rows], dtype=bool)
        return (np.concatenate(masked_rows, axis=0),), varying_rows


class NoInvariantNoShapeMasker:
    """Masker without invariants/shape to cover fallback branches."""

    supports_delta_masking = False

    def __call__(self, mask, x, y):
        out = np.zeros((1, x.shape[0] + y.shape[0]), dtype=float)
        return out


class CallableShapeMasker(BasicMasker):
    """Masker exposing callable shape/mask_shapes."""

    def shape(self, x):
        return (2, x.shape[0])

    def mask_shapes(self, x):
        return [(x.shape[0],)]


class TwoArgSubsetMasker:
    """Two-input masker used to exercise len(args)>1 subset path."""

    supports_delta_masking = False
    shape = (2, 2)

    def invariants(self, a, b):
        # one row varies, one row is invariant
        return np.array([[False, False], [True, True]], dtype=bool)

    def __call__(self, mask, a, b):
        rows = np.array([[a[0], b[0]], [a[1], b[1]]], dtype=float)
        rows[:, ~mask] = 0.0
        return rows


def single_output_model(x):
    """Return one scalar output per row."""
    return x.sum(axis=1)


class TestHelpers:
    def test_convert_delta_mask_to_full_toggles_expected_columns(self):
        """Delta-encoded masks are converted into expected cumulative full masks."""
        masks = np.array([0, 2, -1, 1, MaskedModel.delta_mask_noop_value], dtype=int)
        full_masks = np.zeros((4, 3), dtype=bool)
        _convert_delta_mask_to_full(masks, full_masks)
        assert_array_equal(
            full_masks,
            np.array(
                [
                    [True, False, False],
                    [True, False, True],
                    [False, True, True],
                    [False, True, True],
                ],
                dtype=bool,
            ),
        )

    def test_upcast_array_float16_to_float32(self):
        """float16 arrays are upcast for numba-safe execution."""
        arr = np.array([1.0, 2.0], dtype=np.float16)
        out = _upcast_array(arr)
        assert out.dtype == np.float32

    def test_build_fixed_output_single_and_multi_shapes(self):
        """Output builder handles both single-output and multi-output arrays."""
        num_varying_rows = np.array([2], dtype=int)
        varying_rows = np.array([[True, True]], dtype=bool)
        batch_positions = np.array([0, 2], dtype=int)

        avg_single = np.zeros(1, dtype=np.float32)
        last_single = np.zeros(2, dtype=np.float32)
        outputs_single = np.array([1.0, 3.0], dtype=np.float16)
        _build_fixed_output(
            avg_single,
            last_single,
            outputs_single,
            batch_positions,
            varying_rows,
            num_varying_rows,
            links.identity,
            None,
        )
        assert_allclose(avg_single, np.array([2.0]), rtol=1e-5)

        avg_multi = np.zeros((1, 2), dtype=np.float32)
        last_multi = np.zeros((2, 2), dtype=np.float32)
        outputs_multi = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
        _build_fixed_output(
            avg_multi, last_multi, outputs_multi, batch_positions, varying_rows, num_varying_rows, links.identity, None
        )
        assert_allclose(avg_multi[0], np.array([2.0, 3.0]), rtol=1e-5)

    def test_make_masks_returns_expected_shape(self):
        """Tree clustering matrix is converted into a boolean CSR mask matrix."""
        cluster = np.array([[0, 1, 0.1, 2], [2, 3, 0.2, 3]], dtype=float)
        mask_matrix = make_masks(cluster)
        assert mask_matrix.shape == (5, 3)
        assert mask_matrix.dtype == bool

    def test_link_reweighting_returns_finite_weights(self):
        """Link reweighting returns finite, normalized weights."""
        p = np.array([[0.2, 0.6], [0.8, 0.4]], dtype=float)
        w = link_reweighting(p, links.logit)
        assert w.shape == p.shape
        assert np.all(np.isfinite(w))
        assert_allclose(np.sum(w, axis=0), np.array([2.0, 2.0]), rtol=1e-5)

    def test_numba_single_output_helper_py_func(self):
        """Single-output numba helper updates, averages, and carries forward."""
        averaged_outs = np.zeros(2, dtype=float)
        last_outs = np.array([10.0, 20.0], dtype=float)
        outputs = np.array([4.0], dtype=float)
        batch_positions = np.array([0, 1, 1], dtype=int)
        varying_rows = np.array([[True, False], [False, False]], dtype=bool)
        num_varying_rows = np.array([1, 0], dtype=int)

        _build_fixed_single_output.py_func(
            averaged_outs, last_outs, outputs, batch_positions, varying_rows, num_varying_rows, links.identity, None
        )
        assert_allclose(averaged_outs[0], np.mean([4.0, 20.0]))
        assert_allclose(averaged_outs[1], averaged_outs[0])

    def test_numba_single_output_helper_py_func_full_update_with_weights(self):
        """Single-output helper supports full-row update and weighted averaging."""
        averaged_outs = np.zeros(1, dtype=float)
        last_outs = np.zeros(2, dtype=float)
        outputs = np.array([0.2, 0.8], dtype=float)
        batch_positions = np.array([0, 2], dtype=int)
        varying_rows = np.array([[True, True]], dtype=bool)
        num_varying_rows = np.array([2], dtype=int)
        weights = np.array([0.5, 1.5], dtype=float)

        _build_fixed_single_output.py_func(
            averaged_outs, last_outs, outputs, batch_positions, varying_rows, num_varying_rows, links.identity, weights
        )
        assert_allclose(averaged_outs[0], np.mean(weights * outputs))

    def test_numba_multi_output_helper_py_func(self):
        """Multi-output numba helper handles partial updates and per-output means."""
        averaged_outs = np.zeros((2, 2), dtype=float)
        last_outs = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=float)
        outputs = np.array([[2.0, 4.0]], dtype=float)
        batch_positions = np.array([0, 1, 1], dtype=int)
        varying_rows = np.array([[False, True], [False, False]], dtype=bool)
        num_varying_rows = np.array([1, 0], dtype=int)

        _build_fixed_multi_output.py_func(
            averaged_outs, last_outs, outputs, batch_positions, varying_rows, num_varying_rows, links.identity, None
        )
        assert_allclose(averaged_outs[0], np.array([6.0, 12.0]))
        assert_allclose(averaged_outs[1], averaged_outs[0])

    def test_numba_multi_output_helper_py_func_full_update_with_weights(self):
        """Multi-output helper supports full-row update with per-output weights."""
        averaged_outs = np.zeros((1, 2), dtype=float)
        last_outs = np.zeros((2, 2), dtype=float)
        outputs = np.array([[0.2, 0.4], [0.8, 0.6]], dtype=float)
        batch_positions = np.array([0, 2], dtype=int)
        varying_rows = np.array([[True, True]], dtype=bool)
        num_varying_rows = np.array([2], dtype=int)
        weights = np.array([[1.0, 0.5], [1.0, 1.5]], dtype=float)

        _build_fixed_multi_output.py_func(
            averaged_outs, last_outs, outputs, batch_positions, varying_rows, num_varying_rows, links.identity, weights
        )
        assert_allclose(averaged_outs[0, 0], np.mean(weights[:, 0] * outputs[:, 0]))
        assert_allclose(averaged_outs[0, 1], np.mean(weights[:, 1] * outputs[:, 1]))

    def test_init_and_rec_fill_masks_py_func(self):
        """Mask-tree helper functions build expected root membership."""
        cluster = np.array([[0, 1, 0.1, 2], [2, 3, 0.2, 3]], dtype=float)
        M = cluster.shape[0] + 1
        indices_row_pos = np.zeros(2 * M - 1, dtype=int)
        indptr = np.zeros(2 * M, dtype=int)
        indices = np.zeros(int(np.sum(cluster[:, 3])) + M, dtype=int)

        _init_masks.py_func(cluster, M, indices_row_pos, indptr)
        _rec_fill_masks.py_func(cluster, indices_row_pos, indptr, indices, M, cluster.shape[0] - 1 + M)
        root_start, root_end = indptr[-2], indptr[-1]
        assert set(indices[root_start:root_end].tolist()) == {0, 1, 2}

    def test_rec_fill_masks_py_func_leaf_base_case(self):
        """Leaf base case writes the leaf index directly."""
        cluster = np.array([[0, 1, 0.1, 2]], dtype=float)
        M = 2
        indices_row_pos = np.array([0, 1, 2], dtype=int)
        indptr = np.array([0, 1, 2, 4], dtype=int)
        indices = np.zeros(4, dtype=int)
        _rec_fill_masks.py_func(cluster, indices_row_pos, indptr, indices, M, 1)
        assert indices[indices_row_pos[1]] == 1


class TestMaskedModel:
    def test_init_len_and_varying_inputs(self):
        """MaskedModel initializes expected dimensions and varying input indices."""
        x = np.array([0.3, 0.4, 0.5], dtype=float)
        mm = MaskedModel(single_output_model, BasicMasker(), links.identity, False, x)
        assert len(mm) == 3
        assert mm._masker_rows == 2
        assert mm._masker_cols == 3
        assert_array_equal(mm.varying_inputs(), np.array([0, 1, 2]))

    def test_init_fallback_without_invariants_or_shape(self):
        """Fallback init path infers feature length from args when no shape is available."""
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0, 5.0])
        mm = MaskedModel(single_output_model, NoInvariantNoShapeMasker(), links.identity, False, x, y)
        assert len(mm) == 5
        assert mm._variants is None
        assert_array_equal(mm.varying_inputs(), np.array([0, 1, 2, 3, 4]))
        assert mm.mask_shapes == [x.shape, y.shape]

    def test_init_with_callable_shape_and_mask_shapes(self):
        """Callable shape/mask_shapes are respected when provided by masker."""
        x = np.array([0.3, 0.4, 0.5], dtype=float)
        mm = MaskedModel(single_output_model, CallableShapeMasker(), links.identity, False, x)
        assert mm._masker_rows == 2
        assert mm._masker_cols == 3
        assert mm.mask_shapes == [(3,)]

    def test_call_with_full_masks_returns_one_output_per_mask(self):
        """2D mask input uses full masking path and returns expected shape."""
        x = np.array([0.6, 0.2, 0.9], dtype=float)
        mm = MaskedModel(single_output_model, BasicMasker(), links.identity, False, x)
        masks = np.array([[True, False, False], [True, True, False]], dtype=bool)
        out = mm(masks)
        assert out.shape == (2,)

    def test_call_with_delta_masks_converts_when_not_supported(self):
        """1D delta masks are converted to full masks if masker lacks delta support."""
        x = np.array([0.6, 0.2, 0.9], dtype=float)
        mm = MaskedModel(single_output_model, BasicMasker(), links.identity, False, x)
        masks = np.array([0, 1, -1, 2, MaskedModel.delta_mask_noop_value], dtype=int)
        out = mm(masks, batch_size=2)
        assert out.shape[0] == int(np.sum(masks >= 0))

    def test_call_with_delta_masking_supported_path(self):
        """1D delta masks use the dedicated delta-masking path when supported."""
        x = np.array([0.1, 0.7, 0.5], dtype=float)
        mm = MaskedModel(single_output_model, DeltaMasker(), links.identity, False, x)
        masks = np.array([0, 2, -1, 1], dtype=int)
        out = mm(masks, zero_index=0)
        assert out.shape[0] == int(np.sum(masks >= 0))

    def test_linearize_link_in_full_masking_path(self):
        """Full masking path computes background outputs and linearizing weights."""
        x = np.array([0.2, 0.4, 0.8], dtype=float)

        def prob_model(arr):
            return np.clip(arr.mean(axis=1), 1e-3, 1 - 1e-3)

        mm = MaskedModel(prob_model, BasicMasker(), links.logit, True, x)
        masks = np.array([[False, False, False], [True, False, False]], dtype=bool)
        out = mm._full_masking_call(masks, zero_index=0, batch_size=2)
        assert out.shape == (2,)
        assert mm._linearizing_weights is not None
        assert mm.background_outputs.shape[0] == mm._masker_rows

    def test_linearize_link_in_delta_masking_path(self):
        """Delta masking path computes linearizing weights when requested."""
        x = np.array([0.1, 0.7, 0.5], dtype=float)

        def prob_model(arr):
            return np.clip(arr.mean(axis=1), 1e-3, 1 - 1e-3)

        mm = MaskedModel(prob_model, DeltaMasker(), links.logit, True, x)
        masks = np.array([0, 2, -1, 1], dtype=int)
        out = mm(masks, zero_index=0)
        assert out.shape[0] == int(np.sum(masks >= 0))
        assert mm._linearizing_weights is not None

    def test_main_effects_returns_full_length_vector(self):
        """main_effects returns a vector aligned to the full feature space."""
        x = np.array([0.5, 0.1, 0.9], dtype=float)
        mm = MaskedModel(single_output_model, DeltaMasker(), links.identity, False, x)
        effects = mm.main_effects(batch_size=2)
        assert effects.shape == (3,)

    def test_len_args_gt_one_subset_branch_current_behavior(self):
        """Two-input subset branch is currently not indexable with bool masks (documented behavior)."""
        a = np.array([1.0, 2.0], dtype=float)
        b = np.array([3.0, 4.0], dtype=float)
        mm = MaskedModel(single_output_model, TwoArgSubsetMasker(), links.identity, False, a, b)
        masks = np.array([[False, False], [True, False]], dtype=bool)
        with pytest.raises(TypeError):
            mm._full_masking_call(masks, batch_size=2)

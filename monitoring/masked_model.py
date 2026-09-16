import numpy as np

from shap.links import identity
from shap.utils._masked_model import _build_fixed_output


class MaskedModelSuite:
    sample_count = 100
    mask_count = 100
    output_count = 4

    def setup(self):
        rng = np.random.default_rng(0)
        self.batch_positions = np.arange(self.mask_count + 1, dtype=np.int64) * self.sample_count
        self.varying_rows = np.ones((self.mask_count, self.sample_count), dtype=bool)
        self.num_varying_rows = np.full(self.mask_count, self.sample_count, dtype=np.int64)

        self.single_averaged = np.zeros(self.mask_count)
        self.single_last = np.zeros(self.sample_count)
        self.single_outputs = rng.random(self.mask_count * self.sample_count)

        self.multi_averaged = np.zeros((self.mask_count, self.output_count))
        self.multi_last = np.zeros((self.sample_count, self.output_count))
        self.multi_outputs = rng.random((self.mask_count * self.sample_count, self.output_count))

        # Warm up the Numba implementation when comparing against older commits.
        self.time_single_output()
        self.time_multi_output()

    def time_single_output(self):
        _build_fixed_output(
            self.single_averaged,
            self.single_last,
            self.single_outputs,
            self.batch_positions,
            self.varying_rows,
            self.num_varying_rows,
            identity,
            None,
        )

    def time_multi_output(self):
        _build_fixed_output(
            self.multi_averaged,
            self.multi_last,
            self.multi_outputs,
            self.batch_positions,
            self.varying_rows,
            self.num_varying_rows,
            identity,
            None,
        )

import numpy as np

from shap.datasets import adult
from shap.maskers import Independent, Partition
from shap.utils import MaskedModel


class TabularMaskerSuite:
    # Adapted from tests/maskers/test_tabular.py and monitoring/permutation_explainer.py
    max_samples = 100

    def setup(self):
        X, _ = adult()
        if self.max_samples is not None:
            X = X.iloc[: self.max_samples]
        self.X = X.values
        self.x = self.X[0]

        self.mask = np.ones(self.X.shape[1], dtype=bool)
        self.mask[::2] = False

        feature_indices = np.arange(self.X.shape[1], dtype=np.int64)
        self.delta_masks = np.concatenate(
            (
                np.array([MaskedModel.delta_mask_noop_value], dtype=np.int64),
                feature_indices,
                feature_indices,
            )
        )

        self.independent_masker = Independent(self.X)
        self.partition_masker = Partition(self.X, clustering=None)

        # Dry run to avoid numba jit compilation time in the benchmark
        _ = self.independent_masker(self.delta_masks, self.x)
        _ = self.partition_masker(self.delta_masks, self.x)

    def time_independent_masking(self):
        _ = self.independent_masker(self.mask, self.x)

    def time_independent_delta_masking(self):
        _ = self.independent_masker(self.delta_masks, self.x)

    def time_partition_masking(self):
        _ = self.partition_masker(self.mask, self.x)

    def time_partition_delta_masking(self):
        _ = self.partition_masker(self.delta_masks, self.x)

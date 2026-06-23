"""Tests for the Coefficient explainer."""

from __future__ import annotations

import numpy as np
import pytest

from shap.explainers.other._coefficient import Coefficient


class _LinearLikeModel:
    def __init__(self, coef):
        self.coef_ = np.asarray(coef)


def test_coefficient_requires_model_with_coef_attribute():
    with pytest.raises(AssertionError, match="does not have a coef_ attribute"):
        Coefficient(model=object())


def test_coefficient_attributions_tiles_coefficients_by_num_rows():
    model = _LinearLikeModel([0.5, -1.0, 2.0])
    explainer = Coefficient(model)

    X = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ]
    )

    attributions = explainer.attributions(X)

    expected = np.tile(model.coef_, (X.shape[0], 1))
    np.testing.assert_allclose(attributions, expected)

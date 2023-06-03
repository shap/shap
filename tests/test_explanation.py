"""This file contains tests for the `shap._explanation` module.
"""

import numpy as np
import pytest
import shap


def test_explanation_hstack():
    """Checks that `hstack` works as expected with two valid Explanation objects.
    And that it returns an Explanation object.
    """
    # generate 2 Explanation objects for stacking
    rs = np.random.RandomState(0)
    base_vals = np.ones(20) * 0.123
    exp1 = shap.Explanation(
        values=rs.randn(20, 7),
        base_values=base_vals,
    )
    exp2 = shap.Explanation(
        values=rs.randn(20, 5),
        base_values=base_vals,
    )
    new_exp = exp1.hstack(exp2)

    assert isinstance(new_exp, shap.Explanation)
    assert new_exp.values.shape == (20, 12)


def test_explanation_hstack_errors():
    """Checks that `hstack` throws errors on invalid input.
    """
    # generate 2 Explanation objects for stacking
    rs = np.random.RandomState(1)
    base_vals = np.ones(20) * 0.123
    base_exp = shap.Explanation(
        values=rs.randn(20, 5),
        base_values=base_vals,
    )

    with pytest.raises(
        AssertionError,
        match="Can't hstack explanations with different numbers of rows",
    ):
        exp2 = shap.Explanation(
            values=rs.randn(7, 5),
            base_values=np.ones(7),
        )
        _ = base_exp.hstack(exp2)

    with pytest.raises(
        AssertionError,
        match="Can't hstack explanations with different base values",
    ):
        exp2 = shap.Explanation(
            values=rs.randn(20, 5),
            base_values=np.ones(20) * 0.987,
        )
        _ = base_exp.hstack(exp2)

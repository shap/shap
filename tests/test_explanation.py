"""This file contains tests for the `shap._explanation` module.
"""

import numpy as np
import pytest

import shap


def test_explanation_hstack(random_seed):
    """Checks that `hstack` works as expected with two valid Explanation objects.
    And that it returns an Explanation object.
    """
    # generate 2 Explanation objects for stacking
    rs = np.random.RandomState(random_seed)
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


def test_explanation_hstack_errors(random_seed):
    """Checks that `hstack` throws errors on invalid input.
    """
    # generate 2 Explanation objects for stacking
    rs = np.random.RandomState(random_seed)
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


@pytest.mark.parametrize("N", [4, 5, 6])
def test_feature_names_slicing_for_square_arrays(random_seed, N):
    """Checks that feature names in Explanations are properly sliced with "square"
    arrays (N==k).

    For 2D arrays, there is an ambiguity in how to assign the feature names to the
    slicer index. E.g. if feature_names is a list of 5 elements, and the shap_values is
    a (5,5) array, it's ambiguous whether the axis=0 or axis=1 refers to the "feature
    columns".

    This test ensures that we give higher priority to axis=1 for the feature_names for
    square arrays. Since most of the time, the 2D shap values arrays are assembled as
    (# samples, # features).

    cf. GH Issue #2722, #2699.
    """
    rs = np.random.RandomState(random_seed)
    featnames = list("abcde")

    exp = shap.Explanation(
        # an array of this shape typically arises as the shap values of N samples, k=5 features
        values=rs.rand(N, 5),
        feature_names=featnames,
        output_names=featnames,
    )
    first_sample = exp[0]
    # exp[0] used to return "a" incorrectly when N=5 here, instead of ["a","b","c","d","e"]
    assert first_sample.feature_names == first_sample.output_names == featnames
    column_e = exp[..., "e"]
    assert column_e.feature_names == "e"

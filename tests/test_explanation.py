"""This file contains tests for the `shap._explanation` module."""

from textwrap import dedent

import numpy as np
import pandas as pd
import pytest
from pytest import param
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier

import shap
from shap._explanation import OpHistoryItem


def test_explanation_repr():
    exp = shap.Explanation(values=np.arange(5))
    assert (
        exp.__repr__()
        == dedent(
            """
            .values =
            array([0, 1, 2, 3, 4])
            """
        ).strip()
    )

    exp = shap.Explanation(values=np.arange(5), base_values=0.5, data=np.ones(5))
    assert (
        exp.__repr__()
        == dedent(
            """
            .values =
            array([0, 1, 2, 3, 4])

            .base_values =
            0.5

            .data =
            array([1., 1., 1., 1., 1.])
            """
        ).strip()
    )


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
    """Checks that `hstack` throws errors on invalid input."""
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
        ValueError,
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

    cf. GH #2722, GH #2699.
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


def test_populating_op_history():
    """Tests whether the Explanation.op_history attribute is populated properly after operations have been applied."""
    values = np.arange(-18, 17).reshape(7, 5)

    # apply some operations
    exp = shap.Explanation(values=values).abs.sample(5, random_state=0).flip[..., :3].mean(axis=0)
    exp += 2
    expected_op_names = [
        "abs",
        "__getitem__",
        "flip",
        "__getitem__",
        "mean",
        "__add__",
    ]

    op_history = exp.op_history
    # sanity check for op_history
    assert len(op_history) == 6
    assert all(isinstance(op, OpHistoryItem) for op in op_history)
    assert [op.name for op in op_history] == expected_op_names

    # check that operations have been applied and produce the correct output
    assert np.allclose(exp.values, [10.8, 11.0, 11.6])


@pytest.mark.parametrize(
    "inp",
    [
        param(None, id="None"),
        param([1, 2, 3], id="list[int]"),
        param({"a": 10}, id="dict[int]"),
    ],
)
def test_cohorts_invalid_input(inp):
    with pytest.raises(TypeError):
        _ = shap.Cohorts(test_grp=inp)

    with pytest.raises(TypeError):
        ch = shap.Cohorts()
        ch.cohorts = inp


def test_cohorts_magic_methods(random_seed):
    rs = np.random.RandomState(random_seed)
    e_size = (1_000, 5)
    exp = shap.Explanation(
        values=rs.uniform(low=-1, high=1, size=e_size),
        data=rs.normal(loc=1, scale=3, size=e_size),
        feature_names=list("abcde"),
    )

    exp_neg = exp[exp[:, "a"].data < 0]
    exp_pos = exp[exp[:, "a"].data >= 0]
    ch = shap.Cohorts(col_a_neg=exp_neg, col_a_pos=exp_pos)

    # normal attribute access AND method access -> should be dispatched to Explanation objects
    new_ch = ch.abs.mean(axis=0)
    assert isinstance(new_ch, shap.Cohorts)
    assert np.allclose(
        new_ch.cohorts["col_a_neg"].values,
        exp_neg.abs.mean(axis=0).values,
    )
    assert np.allclose(
        new_ch.cohorts["col_a_pos"].values,
        exp_pos.abs.mean(axis=0).values,
    )

    # getitem access -> should be dispatched to Explanation objects
    new_ch = ch[..., "a"]
    assert isinstance(new_ch, shap.Cohorts)
    assert np.allclose(
        new_ch.cohorts["col_a_neg"].values,
        exp_neg[..., "a"].values,
    )
    assert np.allclose(
        new_ch.cohorts["col_a_pos"].values,
        exp_pos[..., "a"].values,
    )


def test_cohorts_magic_methods_errors():
    """We don't support dispatching __call__ to the Explanation objects in Cohorts.
    The only valid use case for cohorts.__call__() is for invoking Explanation methods (see above test).
    """
    ch = shap.Cohorts()
    with pytest.raises(ValueError, match=r"No methods"):
        ch(axis=0)


def test_cohorts_multi_class():
    # Load dataset
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    Y = data.target
    model = RandomForestClassifier(random_state=42)
    model.fit(X, Y)

    explainer = shap.TreeExplainer(model)

    shap_values = explainer(X[:100])

    with pytest.raises(ValueError, match="Cohorts cannot be calculated on multiple outputs at once."):
        shap_values.cohorts(2)

    cohorts = shap_values[..., 0].cohorts(2)
    isinstance(cohorts, shap.Cohorts)

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


def test_explanation_vstack_basic(random_seed):
    """Test standard 2D tabular vstacking with compute_time tracking and value verification."""
    rs = np.random.RandomState(random_seed)
    exp1 = shap.Explanation(
        values=rs.normal(size=(10, 5)),
        base_values=rs.normal(size=(10,)),
        data=rs.normal(size=(10, 5)),
        feature_names=["A", "B", "C", "D", "E"],
        compute_time=0.5,
    )
    exp2 = shap.Explanation(
        values=rs.normal(size=(5, 5)),
        base_values=rs.normal(size=(5,)),
        data=rs.normal(size=(5, 5)),
        feature_names=["A", "B", "C", "D", "E"],
        compute_time=0.3,
    )

    stacked = exp1.vstack(exp2)

    # 1. Shape Verification
    assert stacked.values.shape == (15, 5)
    assert stacked.base_values.shape == (15,)
    assert stacked.data.shape == (15, 5)
    assert np.all(np.array(stacked.feature_names) == np.array(["A", "B", "C", "D", "E"]))

    # 2. Value Verification (crucial for concatenation logic)
    assert np.allclose(stacked.values[:10], exp1.values)
    assert np.allclose(stacked.values[10:], exp2.values)
    assert np.allclose(stacked.base_values[:10], exp1.base_values)
    assert np.allclose(stacked.base_values[10:], exp2.base_values)

    # 3. Float and Tracking Verification
    assert np.isclose(stacked.compute_time, 0.8)
    assert stacked.op_history[-1].name == "vstack"
    assert len(stacked.op_history) == len(exp1.op_history) + 1


def test_explanation_vstack_feature_name_mismatch(random_seed):
    """Ensure vstack fails cleanly when feature names do not match."""
    rs = np.random.RandomState(random_seed)
    exp1 = shap.Explanation(values=rs.normal(size=(10, 2)), feature_names=["A", "B"])
    exp2 = shap.Explanation(values=rs.normal(size=(5, 2)), feature_names=["A", "C"])  # Mismatch

    with pytest.raises(ValueError, match="different feature names"):
        exp1.vstack(exp2)


def test_explanation_vstack_dimension_mismatch(random_seed):
    """Ensure vstack fails if feature dimensions do not match."""
    rs = np.random.RandomState(random_seed)
    exp1 = shap.Explanation(values=rs.normal(size=(10, 3)))
    exp2 = shap.Explanation(values=rs.normal(size=(5, 4)))  # Mismatch in columns

    with pytest.raises(ValueError, match="different feature dimensions"):
        exp1.vstack(exp2)


def test_explanation_vstack_inconsistent_metadata(random_seed):
    """Ensure vstack fails if one explanation has metadata and the other does not."""
    rs = np.random.RandomState(random_seed)
    # exp2 completely omits 'data' to test the strict None check
    exp1 = shap.Explanation(values=rs.normal(size=(10, 2)), data=rs.normal(size=(10, 2)))
    exp2 = shap.Explanation(values=rs.normal(size=(5, 2)))

    with pytest.raises(ValueError, match="inconsistent metadata"):
        exp1.vstack(exp2)


def test_explanation_vstack_multiclass(random_seed):
    """Test vstacking on 3D (multiclass) explanations and verify values."""
    rs = np.random.RandomState(random_seed)
    exp1 = shap.Explanation(values=rs.normal(size=(10, 4, 3)), data=rs.normal(size=(10, 4)))
    exp2 = shap.Explanation(values=rs.normal(size=(8, 4, 3)), data=rs.normal(size=(8, 4)))

    stacked = exp1.vstack(exp2)

    # Shape Checks
    assert stacked.values.shape == (18, 4, 3)
    assert stacked.data.shape == (18, 4)

    # Value Checks
    assert np.allclose(stacked.values[:10], exp1.values)
    assert np.allclose(stacked.values[10:], exp2.values)


def test_explanation_vstack_scalar_base_values(random_seed):
    """Ensure scalar base_values are correctly broadcasted to arrays matching the row counts."""
    rs = np.random.RandomState(random_seed)
    exp1 = shap.Explanation(values=rs.normal(size=(10, 3)), base_values=0.5)
    exp2 = shap.Explanation(values=rs.normal(size=(5, 3)), base_values=0.5)

    stacked = exp1.vstack(exp2)

    # The scalars should be broadcasted to a 1D array of length (10 + 5)
    assert stacked.base_values.shape == (15,)

    # Verify the first 10 elements belong to exp1's broadcast, and the next 5 to exp2's
    assert np.allclose(stacked.base_values[:10], np.full(10, 0.5))
    assert np.allclose(stacked.base_values[10:], np.full(5, 0.5))


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
        "sample",
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


def test_cohorts_generation_with_one_feature():
    exp = shap.Explanation(
        values=np.random.uniform(low=-1, high=1, size=(500, 1)),
        data=np.random.normal(loc=1, scale=3, size=(500, 1)),
        feature_names=list("a"),
    )
    cohorts = exp.cohorts(3)
    assert isinstance(cohorts, shap.Cohorts)
    assert len(cohorts.cohorts) == 3
